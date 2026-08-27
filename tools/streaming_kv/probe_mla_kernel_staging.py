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
"""Probe 1.2: does the MLA decode (FIA) kernel accept staging-style KV buffers?

Three variants for one small decode case (identical logical KV contents):
  (a) baseline : standard paged KV cache + block table (reference output)
  (b) staging  : a dense [num_reqs * max_blocks, block_size, dim] buffer whose
                 block table is the identity mapping per request — i.e. the
                 layout our HBM staging buffer uses after the gather compacts
                 host blocks per-request-contiguously. Verifies the gather ->
                 staging -> kernel data path end to end.
  (c) zero-copy: k_nope/k_pe base pointers into PINNED HOST memory (the
                 zero-copy fast path from design.md Open Questions).

Run on an NPU machine:

    python tools/streaming_kv/probe_mla_kernel_staging.py

Deliverable: (b) max abs diff vs (a) [expect bf16-identical], (c) works y/n.
Record the verdict in design.md.
"""

import torch

try:
    import torch_npu
except ImportError:
    raise SystemExit("This probe must run on a machine with torch_npu.")

# Mirrors the decode call in vllm_ascend/attention/mla_v1.py::_forward_decode.
# If your CANN version renames kwargs, adjust here only.
NUM_HEADS = 16
NUM_KV_HEADS = 1
KV_LORA_RANK = 512
QK_ROPE_DIM = 64
BLOCK_SIZE = 128
NUM_REQS = 2
SEQ_LENS = [200, 97]
NUM_TOKENS = NUM_REQS  # 1 decode token per request


def run_fia(q_nope, q_pe, k_nope, k_pe, block_table, seq_lens):
    return torch_npu.npu_fused_infer_attention_score(
        q_nope,
        k_nope,
        k_pe,  # v is folded into k_pe for MLA (kv_lora path)
        q_pe,
        num_heads=NUM_HEADS,
        num_key_value_heads=NUM_KV_HEADS,
        input_layout="TND",
        softmax_scale=1.0 / ((KV_LORA_RANK + QK_ROPE_DIM) ** 0.5),
        block_table=block_table,
        block_size=BLOCK_SIZE,
        actual_seq_lengths=seq_lens,
        actual_seq_lengths_kv=seq_lens,
        sparse_mode=3,
    )[0]


def main():
    torch.npu.set_device(0)
    torch.manual_seed(0)
    dtype = torch.bfloat16

    max_blocks_per_req = max((s + BLOCK_SIZE - 1) // BLOCK_SIZE for s in SEQ_LENS)
    total_blocks = NUM_REQS * max_blocks_per_req

    q_nope = torch.randn(NUM_TOKENS, NUM_HEADS, KV_LORA_RANK, dtype=dtype, device="npu")
    q_pe = torch.randn(NUM_TOKENS, NUM_HEADS, QK_ROPE_DIM, dtype=dtype, device="npu")

    # Logical KV contents (per request, per token).
    logical_k = torch.randn(NUM_REQS, max(SEQ_LENS), KV_LORA_RANK, dtype=dtype)
    logical_pe = torch.randn(NUM_REQS, max(SEQ_LENS), QK_ROPE_DIM, dtype=dtype)

    def fill_paged(buf_k, buf_pe, block_table_rows):
        """Write logical KV into a paged buffer according to block table."""
        for r, seq in enumerate(SEQ_LENS):
            for b in range((seq + BLOCK_SIZE - 1) // BLOCK_SIZE):
                blk = int(block_table_rows[r][b])
                lo, hi = b * BLOCK_SIZE, min((b + 1) * BLOCK_SIZE, seq)
                buf_k[blk, 0, : hi - lo] = logical_k[r, lo:hi].to("npu")
                buf_pe[blk, 0, : hi - lo] = logical_pe[r, lo:hi].to("npu")

    # (a) baseline: paged cache with shuffled block ids
    perm = torch.randperm(total_blocks)
    block_table_a = perm.view(NUM_REQS, max_blocks_per_req).to("npu", torch.int32)
    k_a = torch.zeros(total_blocks, NUM_KV_HEADS, BLOCK_SIZE, KV_LORA_RANK, dtype=dtype, device="npu")
    pe_a = torch.zeros(total_blocks, NUM_KV_HEADS, BLOCK_SIZE, QK_ROPE_DIM, dtype=dtype, device="npu")
    fill_paged(k_a, pe_a, block_table_a.cpu())
    out_a = run_fia(q_nope, q_pe, k_a, pe_a, block_table_a, SEQ_LENS)

    # (b) staging layout: identity block table (per-request contiguous)
    block_table_b = torch.arange(total_blocks, dtype=torch.int32).view(NUM_REQS, max_blocks_per_req).to("npu")
    k_b = torch.zeros_like(k_a)
    pe_b = torch.zeros_like(pe_a)
    fill_paged(k_b, pe_b, block_table_b.cpu())
    out_b = run_fia(q_nope, q_pe, k_b, pe_b, block_table_b, SEQ_LENS)

    diff = (out_a.float() - out_b.float()).abs().max().item()
    print(f"(b) staging identity-layout vs (a) paged baseline: max abs diff = {diff}")
    print("    EXPECT: 0.0 (same bytes, same kernel) -> staging layout OK" if diff == 0.0 else
          "    UNEXPECTED mismatch: investigate layout before building gather")

    # (c) zero-copy: host-pinned base pointers
    try:
        k_c = torch.zeros(total_blocks, NUM_KV_HEADS, BLOCK_SIZE, KV_LORA_RANK, dtype=dtype, pin_memory=True)
        pe_c = torch.zeros(total_blocks, NUM_KV_HEADS, BLOCK_SIZE, QK_ROPE_DIM, dtype=dtype, pin_memory=True)
        fill_paged(k_c, pe_c, block_table_b.cpu())
        out_c = run_fia(q_nope, q_pe, k_c, pe_c, block_table_b, SEQ_LENS)
        diff_c = (out_a.float() - out_c.float()).abs().max().item()
        print(f"(c) zero-copy host KV: WORKS, max abs diff vs baseline = {diff_c}")
    except Exception as exc:
        print(f"(c) zero-copy host KV: UNSUPPORTED ({type(exc).__name__}: {exc})")
        print("    -> staging + gather path (b) is the way; zero-copy fast path is off")


if __name__ == "__main__":
    main()
