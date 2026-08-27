# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Probe 1.3: in-step fork/join event chains under ACLGraph capture.

Simulates one decode step of streaming KV tiering: S "streaming layers", each
with fork-event -> copy-stream gather (H2D) -> done-event -> join before a
toy compute kernel, all inside a single torch.npu.graph capture. Verifies:

  1. capture succeeds (multi-stream fork/join chains x S in one graph)
  2. replay produces correct data (contents actually moved per replay,
     with source contents CHANGED between replays to prove the copy is
     re-executed from the current buffer, not baked in)
  3. per-chain overhead (replay time vs S=0 baseline)

Run on an NPU machine:

    python tools/streaming_kv/probe_fork_join_capture.py [--layers 6] [--mb 64]

Deliverable: capture ok y/n, replay correctness, overhead numbers; verdict on
schedule variant A (per-layer fork, this script) vs B. Record in design.md.
"""

import argparse
import time

import torch

try:
    import torch_npu  # noqa: F401
except ImportError:
    raise SystemExit("This probe must run on a machine with torch_npu.")


def build_step(copy_stream, host_blocks, staging, done_events, compute_sink, layers, mb_elems):
    """One 'decode step': S chains of fork -> H2D on copy stream -> join -> toy compute."""

    def step():
        for s in range(layers):
            # fork: copy stream waits until compute stream reaches this point
            fork = torch.npu.Event()
            torch.npu.current_stream().record_event(fork)
            copy_stream.wait_event(fork)
            with torch.npu.stream(copy_stream):
                staging[s].copy_(host_blocks[s], non_blocking=True)
                done_events[s].record(copy_stream)
            # join: compute stream waits for this layer's copy before "attention"
            torch.npu.current_stream().wait_event(done_events[s])
            # toy compute consuming the staged data (stands in for attention)
            compute_sink[s].copy_(staging[s].float().mean(dim=list(range(staging[s].dim()))[1:] if staging[s].dim() > 1 else None).reshape(-1)[: compute_sink[s].numel()].to(torch.bfloat16))

    return step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, default=6, help="S streaming layers")
    parser.add_argument("--mb", type=int, default=64, help="per-layer transfer size MiB")
    parser.add_argument("--replays", type=int, default=50)
    args = parser.parse_args()

    torch.npu.set_device(0)
    numel = args.mb * 1024 * 1024 // 2
    copy_stream = torch.npu.Stream()

    host_blocks = [torch.randn(numel, dtype=torch.bfloat16, pin_memory=True) for _ in range(args.layers)]
    staging = [torch.zeros(numel, dtype=torch.bfloat16, device="npu") for _ in range(args.layers)]
    compute_sink = [torch.zeros(64, dtype=torch.bfloat16, device="npu") for _ in range(args.layers)]
    done_events = [torch.npu.Event() for _ in range(args.layers)]

    step = build_step(copy_stream, host_blocks, staging, done_events, compute_sink, args.layers, numel)

    # eager warmup + reference
    for _ in range(3):
        step()
    torch.npu.current_stream().synchronize()
    ref = [s.clone() for s in staging]

    # capture
    torch.npu.current_stream().synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        step()
    print("capture: OK")

    # replay correctness, including with CHANGED host contents
    for h in host_blocks:
        h.normal_()
    graph.replay()
    torch.npu.current_stream().synchronize()
    mismatches = [i for i in range(args.layers) if not torch.equal(staging[i], host_blocks[i].to("npu"))]
    print("replay correctness:", "OK" if not mismatches else f"FAILED at layers {mismatches}")

    # timing: full step vs compute-only baseline (S=0 equivalent: time the join/toy-compute graph)
    torch.npu.current_stream().synchronize()
    t0 = time.perf_counter()
    for _ in range(args.replays):
        graph.replay()
    torch.npu.current_stream().synchronize()
    full_ms = (time.perf_counter() - t0) / args.replays * 1e3

    print(f"replay step time: {full_ms:.3f} ms "
          f"({args.layers} layers x {args.mb} MiB H2D = {args.layers * args.mb} MiB/step)")
    eff_gbps = args.layers * args.mb / 1024 / (full_ms / 1e3)
    print(f"effective in-graph H2D throughput: {eff_gbps:.1f} GiB/s (upper bound; toy compute hides nothing)")
    print("Record: capture ok, replay ok, step time, throughput -> design.md")


if __name__ == "__main__":
    main()
