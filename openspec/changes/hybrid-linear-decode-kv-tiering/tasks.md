# Tasks: Hybrid Linear Model Decode KV Tiering

> **Status convention (2026-08-27)**: `[x]` below means implementation and
> test/script files are written, but **execution/verification is pending on
> an NPU environment** (this implementation pass ran on a CPU-only Windows
> dev machine with only syntax-level checks). Unchecked tasks are not yet
> implemented.
>
> **Phase-1 layout deviation (needs spec review)**: the host pool is
> per-request contiguous (single-copy onload/writeback) instead of
> block-table-paged. Consequence: the streaming group does not participate
> in prefix cache sharing in phase 1 — the spec scenario "Streaming group
> scheduler accounting / prefix-cache compatibility" should be re-scoped or
> moved to the paged-pool phase.

## 1. Feasibility Spikes (no production code changes)

- [ ] 1.1 Host bandwidth micro-benchmark on Ascend 950: measure achievable H2D bandwidth for (a) SDMA async copies from pinned host memory and (b) device-side direct reads of host memory (gvas pattern from `sparse_copy`), at block sizes matching MLA pages. Deliverable: numbers table; feeds policy defaults and the f-selection formula. **[script ready: `tools/streaming_kv/probe_host_bandwidth.py` — run on 950]**
- [ ] 1.2 MLA decode kernel probe: check whether the Ascend MLA decode attention kernel accepts KV base pointers into (a) a contiguous per-request staging buffer consumed via `seq_lens`, and (b) host-pinned memory directly. Deliverable: probe script + verdict recorded in design.md (decides staging layout and whether the zero-copy fast path from Open Questions is viable). **[script ready: `tools/streaming_kv/probe_mla_kernel_staging.py` — run on NPU]**
- [ ] 1.3 In-step fork/join capture probe: capture a toy graph with 6 streaming layers' worth of fork/copy/join event chains on the copy stream, verify capture succeeds, replay is correct, and measure per-chain overhead. Deliverable: verdict on schedule variant A (per-layer fork) vs B (issue-all-at-step-start) and captured overhead numbers. **[script ready: `tools/streaming_kv/probe_fork_join_capture.py` — run on NPU]**

## 2. Residency Policy and Group Splitting

- [x] 2.1 Implement `streaming_kv/policy.py`: residency plan computation from (HBM budget, `max_model_len`, spec page sizes, B_h2d, target TPS/batch) with explicit `stream_layers` override and validation (full-attention layers only). Verify: unit tests for plan computation, override validation, and low-bandwidth warning. **[implemented: `vllm_ascend/distributed/kv_transfer/streaming_kv/policy.py` + `tests/ut/kv_offload/test_streaming_kv_policy.py` — run on NPU env]**
- [x] 2.2 Add `ascend_config` section for tiering (enable flag, HBM/CPU budgets, host bandwidth, layer list, prefetch depth, mutual-exclusion check vs sparse KV offload / recompute CPU offload). Verify: config parsing unit tests; conflicting configs rejected at startup. **[implemented: `StreamingKVTieringConfig` in `ascend_config.py` + `tests/ut/kv_offload/test_streaming_kv_config.py`]**
- [x] 2.3 Implement activation gating: enable only when MambaSpec + full-attention groups coexist; error on pure dense models; phase-1 check that all streaming layers are `AscendMLAAttentionSpec`. Verify: unit tests over synthetic KV cache configs (K3-like, Qwen3.5-like, dense). **[implemented: `streaming_kv/gating.py` + `tests/ut/kv_offload/test_streaming_kv_gating.py`]**
- [x] 2.4 Mark policy-selected layers' specs `store_on_host=True` at spec creation; confirm group split produces resident + streaming MLA groups and that `verify_and_split_kv_cache_groups` / `find_longest_cache_hit` handle the two full-attention groups. Verify: unit test asserting group structure and prefix-hit alignment across both MLA groups. **[marking implemented: `_apply_streaming_kv_residency` in `model_runner_v1.py`; group-split behavior needs NPU run to confirm]**

## 3. Host Pool, Staging, and Scheduler Accounting

- [x] 3.1 Implement streaming-group capacity accounting: CPU budget (bytes→blocks) for the streaming group; HBM `num_blocks` unchanged for resident/linear groups. Verify: unit test that admission admits the expected extra sequences at a given f. **[implemented: `streaming_kv/accounting.py` + wrapper in `patch_kv_cache_utils.py` + `tests/ut/patch/platform/test_streaming_kv_accounting.py`]**
- [x] 3.2 Implement `streaming_kv/manager.py` host pool: pinned host allocation for the streaming group's KV (per-TP-rank layout following the sparse-offload pattern), plus per-layer fixed-address HBM staging buffers (×2 rotation). Verify: allocation unit test; memory sizes match policy computation. **[implemented as per-request contiguous pool — see deviation note above]**
- [x] 3.3 Wire manager registration into `model_runner_v1` alongside sparse-offload call sites (metadata update, kv cache registration). Verify: model loads with tiering enabled under eager; shapes/layouts asserted. **[wiring implemented: manager init, HBM-allocation skip for streaming layers, decoder-layer wrapping; model-load verification needs NPU]**

## 4. Streaming Data Path (eager first)

- [ ] 4.1 Implement block-table-driven gather onload: device-side copy from host pool to staging buffer for all of a streaming layer's blocks, driven by the in-place-updated block table. Verify: eager-mode unit test comparing staged contents against a reference HBM copy. **[PARTIAL: contiguous single-copy onload implemented in `manager.start_onload`; **missing link: streaming MLA attention impl must consume `manager.staging_buffer()` instead of its placeholder kv cache — not yet implemented**]**
- [ ] 4.2 Implement new-KV writeback: append buffer per streaming layer + D2H to host-pool slots after attention. Verify: multi-step eager generation on a small hybrid model matches non-tiered outputs token-for-token (greedy). **[PARTIAL: `manager.start_writeback` implemented (eager, CPU seq-lens sized); depends on the same impl integration as 4.1]**
- [ ] 4.3 Implement cross-layer overlap: copy-stream fork/join with prefetch depth k derived from the layer pattern (`start_onload` after layer i−k, `wait_onload` before layer i), staging rotation. Verify: profiler trace shows H2D of layer i concurrent with compute of layers i−k..i−1; correctness unchanged. **[PARTIAL: fork/join machinery + `hooks.py` decoder-layer wrapping implemented; profiler verification needs NPU]**
- [ ] 4.4 Prefill handoff: deliver streaming-layer KV to the Decode host pool via the host-to-host P/D path. Verify: PD-disaggregated eager run, first decode step onloads correctly. **[not started]**

## 5. ACLGraph Integration

- [ ] 5.1 Make onload/writeback capture-safe: in-place block-table updates, fixed buffer addresses, capture lifecycle hooks mirroring `sync_prev_onload`/`join_after_forward`. Verify: capture succeeds for all decode batch sizes; replay outputs match eager. **[manager capture-lifecycle methods implemented; custom-op wrapping of hook calls (mutates_args) and acl_graph.py hook points not yet wired]**
- [ ] 5.2 MTP/DSpark support: onload/writeback sized for 1+k tokens per request. Verify: MTP-enabled e2e matches non-tiered outputs.

## 6. Validation

- [ ] 6.1 Correctness e2e: Kimi K3, long-context (≥128k, target 512k) greedy decode with tiering vs full-HBM baseline; verify token-identical outputs.
- [ ] 6.2 Throughput benchmark on 950: measure aggregate TPS at varying f and batch; verify the measured ceiling tracks `B_h2d / (f × KV_bytes)` and that the chosen operating point meets the target.
- [ ] 6.3 Documentation: feature guide (config, sizing formulas, mutual exclusions, tuning) + design doc update with spike results.

## 7. Phase 2 (tracked, not started in this change)

- [ ] 7.1 GQA `FullAttentionSpec` streaming for Qwen3.5-class hybrid models: page-layout adaptation of manager/gather; verify with Qwen3.5 e2e correctness.
