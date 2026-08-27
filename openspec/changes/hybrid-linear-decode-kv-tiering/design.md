# Design: Hybrid Linear Model Decode KV Tiering

## Context

See `proposal.md` — Why. Key facts established during exploration that shape this design:

- Hybrid linear models (K3/Qwen3.5) already produce KV cache groups typed by spec: MambaSpec groups (constant state) vs full-attention groups (dense, grows with seq len). Spec equality drives group merging (`AscendMLAAttentionSpec.merge` requires uniform `store_on_host` per group).
- `AscendMLAAttentionSpec` already carries a `store_on_host` field, currently used only by sparse KV offload in an all-or-nothing way. Specs differing in this field naturally land in separate groups.
- The coordinator already tolerates multiple full-attention groups (DeepSeek-V4 C4/C128 precedent in `patch_kv_cache_coordinator.py:find_longest_cache_hit`).
- Two in-graph data-movement precedents exist and run in production:
  1. **Weight prefetch offloader** (`vllm_ascend/model_executor/offloader/prefetch.py` + `vllm.model_executor.offloader.prefetch`): fork/join multi-stream capture inside ACLGraph, `StaticBufferPool` fixed-address rotation, `prefetch_step` cross-layer prefetch depth. `compilation/acl_graph.py` already handles capture lifecycle (`sync_prev_onload`, `join_after_forward` for the dangling last fork).
  2. **Sparse KV offload** (`sparse_kv_offload_manager.py:onload_topk_kv` with `capturing=True`): in-graph async D2H of metadata, `_launch_host_func` host nodes, device-side indirect copy (`offload.sparse_copy` over gvas host addresses). Proves host pinned memory is device-addressable and per-step-varying source addresses are handled by indirection (block table updated in place at a fixed address).
- Sparse offload does NOT overlap load with compute (copies are inline on the compute stream); its LRU/miss machinery is unnecessary for dense streaming.
- Layerwise prefill offload solves the prefill-side write path (KV streams D2H during prefill, reusable layer buffers) and `SfaRemoteD2HConnector` demonstrates prefill-host → decode-host delivery.

## Goals / Non-Goals

**Goals:**

- Decode-side only: stream a statically chosen subset of full-attention layers' KV from Decode host DRAM every step, overlapped with linear-layer compute, under eager and ACLGraph.
- Model-generic: driven entirely by KV cache spec types and measured/configured bandwidth; no model-name or layer-table hardcoding.
- Correctness identical to full-HBM decode (bf16 tolerance), including MTP/DSpark.
- Startup-time residency policy with explicit layer-list override; residency fixed at runtime.

**Non-Goals:**

- Per-request dynamic residency, staging-buffer-as-cache for short sequences, pressure demotion of resident layers (recorded as future optimizations).
- Prefill-side read backflow (chunked prefill re-read from host).
- GQA `FullAttentionSpec` streaming (phase 2; design must not preclude it).
- Unifying with the three existing offload connectors.

## Decisions

### D1: Residency expressed by spec-level `store_on_host`, splitting one MLA group into two

Mark the policy-selected full-attention layers' specs with `store_on_host=True` at spec creation time. Because `merge()` requires uniform `store_on_host`, layers split naturally into a resident group and a streaming group; no scheduler surgery is needed to *form* the groups. Rationale: group is the scheduler's unit of capacity accounting and block-table management, and residency is fundamentally a capacity-accounting property (streaming blocks must not consume HBM `num_blocks`).

Alternatives considered: per-layer residency bit inside one group — rejected: punches holes in the group's uniform-storage assumption across scheduler, block pool, and prefix cache.

### D2: Streaming group accounting — CPU budget, HBM block ids unused

The streaming group's single-type manager allocates block tables as usual (same granularity, same prefix-cache compatibility), but its capacity check runs against a configured CPU budget (bytes → blocks), and the worker allocates its "KV cache tensor" as pinned host memory instead of NPU memory. On the NPU side, each streaming layer gets only a fixed staging buffer (see D3), independent of `num_blocks`. Rationale: keeps block-table/prefix-cache machinery intact while relocating the storage.

Consequence for prefix cache: hit blocks in the streaming group live in the host pool; a later request hitting the prefix streams them like any other block. No special handling required beyond the existing multi-full-attention-group hit alignment.

### D3: In-graph indirect gather into double-buffered staging, fork/join copy stream

Per streaming layer, the decode-step onload is a device-side gather: a copy kernel reads the layer's block table (fixed address, updated in place by the model runner before replay) and the host pool base address, and moves all of the layer's blocks into a fixed HBM staging buffer. Two mechanisms from the precedents are composed:

- **Addressing** (from sparse offload): indirection via in-place-updated block table; host memory device-addressable (gvas). No host-func round trip needed — dense streaming has no miss logic, the copy plan is a pure function of the block table.
- **Overlap** (from weight prefetch offloader): per streaming layer `i`, `start_onload(i)` forks the copy stream right after layer `i-k`'s attention (prefetch depth `k` ≈ number of linear layers between two full layers + 1, derived from the layer pattern); `wait_onload(i)` joins before layer `i`'s attention. Staging uses `num_in_group = 2` buffer rotation. Capture lifecycle hooks (`sync_prev_onload` / `join_after_forward` pattern) are reused.

**Critical constraint — no cross-step prefetch**: the weight offloader's circular schedule (last layers prefetch the *next* forward's first layers) is incorrect for KV. The block table is only updated by the CPU before each replay; a gather issued during step N−1's tail would resolve addresses from step N−1's block table and deliver stale KV to step N. All onloads for step N must be issued inside step N's forward; prefetch depth is bounded by intra-step layer distance. The static onload schedule (issue/join/writeback hook points derived from the layer pattern, handling adjacent full layers like K3's 92/93) is generated once at startup.

**Staging buffer cost is charged to the policy**: each staging slot holds `max_batch × W_max × page_bytes` (e.g., 2 slots × 32 × 590 MB ≈ 38 GB @512k). Net HBM saving is `(f − SLOTS) × per-layer-batch-KV`, so f must exceed SLOTS (=2) for positive net gain. Staging uses a dense per-request layout consumed via `seq_lens` (no block table), with the gather kernel doing paged→dense compaction (pending probe P2; fallback: paged staging + synthesized block table).

Alternatives considered: (a) zero-copy direct host addressing by the attention kernel — simplest, but requires the MLA decode kernel to accept host pointers; kept as a fallback/fast path pending a kernel probe (Open Questions). (b) Same-stream inline copies — no overlap, rejected. (c) Segmented graph replay with eager copies between segments — viable fallback if fork/join capture hits a limitation with this copy kernel, adds per-segment replay overhead.

Detailed function-level design (manager API, hook points, event timelines for eager/capture/replay, MTP and padding handling) lives in `docs/hybrid_linear_kv_tiering_graph_mode_design.md`.

### D4: Writeback — decode-side D2H on the copy stream; prefill-side host-to-host delivery

New KV rows for streaming layers are written during forward into a small fixed NPU append buffer, then D2H'd to their host-pool slots on the copy stream after the layer's attention (also captured in-graph; source addresses derive from slot mapping). On the prefill side, streaming layers' KV follows the existing host-to-host P/D pattern (prefill host pool → decode host pool, SfaRemoteD2HConnector-style), so streaming-layer KV never occupies persistent NPU memory on either node.

### D5: Static residency policy

Startup policy inputs: per-rank HBM budget for KV (after weights/activations), `max_model_len`, per-layer page sizes from specs, host bandwidth (config value, optionally probed by a micro-benchmark at init), and a target aggregate TPS or target batch. It picks the smallest set of full-attention layers whose removal from HBM satisfies the budget, subject to `TPS_ceiling = B_h2d / (f × KV_bytes_per_seq) ≥ target`. Layers are chosen from the tail of the network first (later layers free the same bytes but keep early-layer latency low; ties are arbitrary). An explicit `stream_layers` config list overrides the computation (validated: full-attention layers only). If the ceiling is below target and the user has not acknowledged, startup warns loudly.

Rationale for static: dynamic per-request residency conflicts with ACLGraph (mutating memory plan, per-request data paths) and scheduler accounting. The main dynamic benefit (short sequences shouldn't pay streaming cost) is deferred to a documented future optimization (staging-buffer block cache).

### D6: Activation gating

Enabled only when (a) config flag on, (b) model KV config mixes MambaSpec and full-attention groups, (c) phase-1: streaming layers are all `AscendMLAAttentionSpec`. Pure dense models → startup error. GQA hybrid models (Qwen3.5) → accepted by gating but streaming list must be empty/MLA-only in phase 1; full GQA streaming is phase 2 (same data path, different page layout).

### D7: Module placement

New `vllm_ascend/distributed/kv_transfer/streaming_kv/` package: `policy.py` (residency plan), `manager.py` (host pool, staging buffers, block-table-driven onload/writeback, lifecycle hooks), `copy_pipeline.py` (copy-stream fork/join, capture integration). Spec marking hooks into the model's spec-creation path; worker hooks into `model_runner_v1` alongside the existing sparse-offload call sites. No changes to the three existing offload connectors.

## Risks / Trade-offs

- [950 host bandwidth is an assumption, not a measurement] → Micro-benchmark at init (and a standalone spike first); policy refuses/warns below the configured ceiling. All sizing formulas parameterized by measured B_h2d.
- [MLA decode kernel may not tolerate staging-buffer layouts that differ from the paged cache] → Stage buffers replicate the paged layout 1:1 (block-major); kernel consumes staging via a synthesized contiguous block table. Kernel probe is task T1.
- [In-graph D2H writeback + H2D onload on one copy stream could exceed the overlap window at high batch] → Prefetch depth `k` configurable; profile-driven tuning; fallback is increasing f (more layers streaming spreads the same bytes thinner per layer) or reducing batch.
- [MTP/DSpark changes per-step tokens and slot mapping] → Onload/writeback sizing uses `1 + num_spec_tokens` rows per request; covered by spec scenario; e2e test with MTP enabled.
- [Host pool capacity is a new admission dimension operators must size] → Startup log prints derived per-sequence host bytes and max sequences; docs with sizing formula; default CPU budget conservative.
- [Feature interacts with existing offload connectors if both configured] → Validation: streaming tiering is mutually exclusive with sparse KV offload and recompute CPU offload at startup (they solve overlapping problems differently); mixing is rejected with a clear error.

## Migration Plan

Feature is opt-in and off by default; no migration. Rollout: (1) land behind config flag with eager mode; (2) validate correctness + bandwidth on 950; (3) enable ACLGraph path; (4) phase 2 GQA. Rollback: remove flag — groups, accounting, and data paths revert exactly to current behavior.

## Open Questions

- Can the MLA decode attention kernel read host-pinned memory directly (zero-copy path)? If yes, staging buffers and gather become optional for sufficiently high B_h2d. Answerable by a small kernel probe without changing specs or task breakdown (it adds a fast path, not a redesign).
- What is the actual fork/join capture overhead and achievable copy-stream concurrency at 24 streaming layers? Affects prefetch-depth tuning only.
