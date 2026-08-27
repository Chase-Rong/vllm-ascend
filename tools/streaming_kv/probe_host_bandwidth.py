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
"""Probe 1.1: host<->NPU H2D bandwidth for streaming KV tiering.

Measures, at block sizes matching MLA pages (512k-context K3: per-request
per-layer KV is large and contiguous):
  (a) SDMA path: pinned host tensor -> NPU tensor via copy_(non_blocking=True)
  (b) device direct-read path: a Triton/device kernel reading host memory
      through the unified address space (gvas pattern used by
      sparse_kv_offload's offload.sparse_copy)

Run on an Ascend 950 machine with torch_npu installed:

    python tools/streaming_kv/probe_host_bandwidth.py

Deliverable: a GB/s table. Feed the (b) number (or (a) if (b) fails) into
streaming_kv policy as B_h2d, and record the verdict in design.md.
"""

import argparse
import time

import torch

try:
    import torch_npu  # noqa: F401
except ImportError:
    raise SystemExit("This probe must run on a machine with torch_npu.")

try:
    from vllm.triton_utils import tl, triton
except Exception:  # triton-ascend not available
    tl, triton = None, None


def bench_sdma_h2d(host: torch.Tensor, dev: torch.Tensor, iters: int) -> float:
    """SDMA H2D bandwidth in GB/s via async copy."""
    stream = torch.npu.current_stream()
    # warmup
    for _ in range(3):
        dev.copy_(host, non_blocking=True)
    stream.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        dev.copy_(host, non_blocking=True)
    stream.synchronize()
    elapsed = time.perf_counter() - start
    return host.numel() * host.element_size() * iters / elapsed / 1e9


if triton is not None:

    @triton.jit
    def _direct_read_kernel(src_ptr, dst_ptr, numel, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < numel
        vals = tl.load(src_ptr + offs, mask=mask)
        tl.store(dst_ptr + offs, vals, mask=mask)


def bench_direct_read(host: torch.Tensor, dev: torch.Tensor, iters: int) -> float | None:
    """Device-side direct read of host memory, GB/s. None if unsupported."""
    if triton is None:
        print("  [direct-read] triton-ascend unavailable, skipped")
        return None
    numel = host.numel()
    block = 1024
    grid = ((numel + block - 1) // block,)
    try:
        # warmup (also the support check: can a device kernel dereference a
        # host pointer at all?)
        _direct_read_kernel[grid](host, dev, numel, BLOCK=block)
        torch.npu.current_stream().synchronize()
    except Exception as exc:
        print(f"  [direct-read] UNSUPPORTED: {type(exc).__name__}: {exc}")
        return None
    start = time.perf_counter()
    for _ in range(iters):
        _direct_read_kernel[grid](host, dev, numel, BLOCK=block)
    torch.npu.current_stream().synchronize()
    elapsed = time.perf_counter() - start
    return numel * host.element_size() * iters / elapsed / 1e9


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes-mb", type=int, nargs="+", default=[64, 256, 590, 1024],
                        help="transfer sizes in MiB (590 MiB ~= one MLA layer of one 512k request)")
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    torch.npu.set_device(0)
    print(f"{'size':>10} | {'SDMA H2D GB/s':>14} | {'direct-read GB/s':>16}")
    print("-" * 50)
    for size_mb in args.sizes_mb:
        nbytes = size_mb * 1024 * 1024
        numel = nbytes // 2  # bf16
        host = torch.empty(numel, dtype=torch.bfloat16, pin_memory=True)
        host.normal_()
        dev = torch.empty(numel, dtype=torch.bfloat16, device="npu")
        sdma = bench_sdma_h2d(host, dev, args.iters)
        direct = bench_direct_read(host, dev, args.iters)
        direct_str = f"{direct:16.1f}" if direct is not None else f"{'N/A':>16}"
        print(f"{size_mb:>8}MB | {sdma:>14.1f} | {direct_str}")
        del host, dev
        torch.npu.empty_cache()

    print()
    print("Record both columns in design.md. Policy B_h2d := min column that works.")


if __name__ == "__main__":
    main()
