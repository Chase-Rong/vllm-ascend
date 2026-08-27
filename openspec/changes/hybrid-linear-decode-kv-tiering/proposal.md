# Proposal: Hybrid Linear Model Decode KV Tiering

## Why

Hybrid linear-attention models (Kimi K3 / KDA, Qwen3.5 / GDN, MiniMax M3) keep most sequence state in constant-size linear-attention states, but their full-attention layers still accumulate dense KV that grows with sequence length. On P/D-disaggregated Decode nodes, this dense KV dominates HBM (Kimi K3 @512k: ~14.2 GB per sequence across 24 MLA layers), capping batch size and throughput. With the high host↔NPU bandwidth of Ascend 950-class hardware, a fraction of full-attention layers can be hosted in CPU DRAM and streamed H2D during decode, overlapped with the KV-free compute of the surrounding linear layers — trading host bandwidth for HBM capacity with no throughput loss. Today vllm-ascend has no mechanism for this: existing offload paths are prefix-cache L2 (inactive blocks), recompute preemption preservation (per-request, all layers), or sparse top-k fetch (sparse-attention models only).

## What Changes

- **Layer-residency group splitting**: at startup, full-attention layers of a hybrid linear model are split into an HBM-resident KV cache group and a host-streaming KV cache group, using the existing `store_on_host` spec marker. The scheduler accounts the streaming group against a CPU budget instead of the HBM block pool. Linear-attention (MambaSpec) groups always stay HBM-resident.
- **Dense streaming data path**: a new decode-side onload path that, every decode step, gathers each streaming layer's full KV from host memory into a fixed HBM staging buffer, via an indirect copy driven by the in-place-updated block table. Cross-layer double buffering with copy-stream fork/join overlaps the H2D transfer of layer `i` with the compute of the preceding linear layers. Works under ACLGraph capture using the patterns proven by the weight prefetch offloader (fork/join capture, static buffer pool) and sparse KV offload (in-graph indirect host↔device movement).
- **Static residency policy**: a startup-time policy computes which full-attention layers are marked streaming from the HBM budget, `max_model_len`, measured/configured host bandwidth, and target batch; an explicit per-layer config list can override it. Residency is fixed for the process lifetime.
- **New-token writeback**: KV produced during decode for streaming layers is written through to the host pool (D2H) instead of a persistent NPU paged cache.
- **Gating**: the mechanism activates only when the model's KV cache config mixes MambaSpec groups with full-attention groups (the machine-checkable definition of a hybrid linear model). Phase 1 supports MLA full-attention layers (Kimi K3); GQA `FullAttentionSpec` streaming (Qwen3.5) is phase 2.
- **Prefill/P-D handoff (decode-side scope)**: prefill-side write-path reuse only — streaming layers' KV is shipped to the Decode host pool via the existing host-to-host P/D pattern (SfaRemoteD2HConnector-style); no prefill read-backflow.

Explicitly **not** in scope (non-goals): per-request dynamic residency (short-sequence HBM caching of streamed blocks), pressure-driven demotion of resident layers, unified abstraction with the existing three offload connectors, prefill-side read backflow, sparse/quantized KV for streaming layers.

## Capabilities

### New Capabilities

- `hybrid-linear-kv-tiering`: Decode-side KV cache tiering for hybrid linear-attention models — residency group splitting, static residency policy, dense per-layer streaming onload with cross-layer overlap, streaming-layer writeback, host pool management, and activation gating.

### Modified Capabilities

<!-- No existing specs in this repo yet; nothing to modify. -->

## Impact

- **KV cache spec/config**: `vllm_ascend/core/kimi_k3`-style spec creation paths and `vllm_ascend/core/kv_cache_interface.py` — marking a configurable subset of full-attention layers with `store_on_host=True`; group merge invariants already keep differing specs in separate groups.
- **Scheduler / coordinator**: `patch_kv_cache_coordinator.py` — streaming group accounted against CPU budget; prefix-cache hit lookup across two full-attention groups (precedent: DeepSeek-V4 C4/C128).
- **Worker / model runner**: `model_runner_v1.py` — host pool + HBM staging allocation, per-layer streaming onload/writeback hooks, ACLGraph lifecycle integration (`sync_prev_onload`/`join_after_forward` pattern in `compilation/acl_graph.py`).
- **New module**: `vllm_ascend/distributed/kv_transfer/streaming_kv/` (manager, residency policy, copy pipeline).
- **P/D connectors**: reuse of host-to-host transfer pattern for streaming-group KV delivery to Decode host pool.
- **Config surface**: new `ascend_config` section (enable flag, HBM/CPU budgets, host bandwidth, explicit layer list, prefetch depth).
- **Hardware assumption**: targets Ascend 950-class host↔NPU bandwidth; on lower-bandwidth hosts the policy must refuse or warn with the computed TPS ceiling.
- **Tests**: unit tests for policy/group splitting/manager; e2e correctness (Kimi K3 long-context) and throughput benchmarks on 950.
