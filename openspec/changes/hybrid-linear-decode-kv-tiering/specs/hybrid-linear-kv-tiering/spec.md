# Spec: hybrid-linear-kv-tiering

## Purpose

Decode-side KV cache tiering for hybrid linear-attention models: a configurable subset of full-attention layers hosts its KV in CPU DRAM and streams it H2D every decode step, freeing HBM to raise batch size on long-context P/D Decode nodes.

## ADDED Requirements

### Requirement: Activation gating for hybrid linear models

The system SHALL enable KV tiering only when the model's KV cache configuration contains at least one linear-attention (MambaSpec) group AND at least one full-attention group, and the user has explicitly enabled the feature via configuration. When the model has no linear-attention group, the system SHALL refuse to enable tiering at startup with an error naming the missing precondition.

#### Scenario: Hybrid linear model with feature enabled
- **WHEN** a model whose KV cache config mixes MambaSpec and full-attention groups (e.g. Kimi K3, Qwen3.5) is started with tiering enabled
- **THEN** the system initializes tiering and reports the computed residency plan at startup

#### Scenario: Pure dense model rejected
- **WHEN** a model with only full-attention groups is started with tiering enabled
- **THEN** startup fails with an error stating tiering requires a hybrid linear model

#### Scenario: Feature disabled by default
- **WHEN** any model is started without the tiering config flag
- **THEN** KV cache groups and data paths are identical to current behavior, with zero host pool allocation

### Requirement: Static residency plan

At startup the system SHALL compute a residency plan assigning each full-attention layer to either HBM-resident or host-streaming, derived from the HBM budget, `max_model_len`, host bandwidth, and per-layer page sizes. The system SHALL accept an explicit per-layer list overriding the automatic plan. The plan SHALL NOT change for the lifetime of the process.

#### Scenario: Automatic plan from budgets
- **WHEN** tiering is enabled with an HBM budget but no explicit layer list
- **THEN** the startup log shows which layers are HBM-resident vs host-streaming and the estimated per-sequence HBM savings

#### Scenario: Explicit layer override
- **WHEN** the user provides an explicit streaming layer list
- **THEN** exactly those full-attention layers are marked streaming, and the list is validated to contain only full-attention layers

#### Scenario: Insufficient host bandwidth warning
- **WHEN** the configured/measured host bandwidth implies an aggregate TPS ceiling below the configured target
- **THEN** the system warns at startup with the computed ceiling, and continues only if the user has acknowledged low-bandwidth operation

### Requirement: Streaming group scheduler accounting

The scheduler SHALL account streaming-group blocks against a CPU budget and resident/linear groups against the HBM block pool. A request SHALL be admittable only when both budgets can satisfy its allocation. Streaming-group block tables SHALL track per-request token-to-host-block mapping with the same granularity as resident groups.

#### Scenario: HBM budget freed by streaming layers
- **WHEN** a hybrid linear model runs with f of its full-attention layers streaming
- **THEN** per-sequence HBM consumption excludes those f layers' KV, and admission control admits correspondingly more concurrent sequences

#### Scenario: CPU budget exhaustion
- **WHEN** the CPU host pool cannot fit a new request's streaming-layer KV
- **THEN** the request is not admitted (or preempted per existing recompute policy) and the event is logged

### Requirement: Dense per-layer streaming onload

Every decode step, for each streaming layer, the system SHALL gather that layer's complete KV for all running requests from the host pool into a fixed-address HBM staging buffer before that layer's attention consumes it, using the request block tables updated in place. The onload of a streaming layer SHALL be overlapped with the compute of preceding layers via cross-layer double buffering on a dedicated copy stream, under both eager and ACLGraph execution.

#### Scenario: Correct attention over streamed KV
- **WHEN** a decode step runs with streaming layers active
- **THEN** each streaming layer's attention output is numerically identical (within bf16 tolerance) to the same run with all layers HBM-resident

#### Scenario: Overlapped transfer under ACLGraph
- **WHEN** decode runs under ACLGraph capture with tiering enabled
- **THEN** graph capture succeeds, replay produces correct outputs, and the H2D transfer of each streaming layer executes concurrently with preceding layers' compute (verifiable via profiling)

#### Scenario: MTP / speculative decode
- **WHEN** speculative decoding (MTP/DSpark) is enabled with k speculative tokens
- **THEN** streaming onload covers all 1+k tokens per request per step and outputs match the non-tiered run

### Requirement: Streaming-layer KV writeback

New KV produced during decode for a streaming layer SHALL be written to the host pool (D2H) instead of a persistent NPU paged cache, and SHALL be visible to subsequent steps' onload. Prefill-produced KV for streaming layers SHALL be delivered to the Decode host pool via the host-to-host P/D transfer path, never occupying persistent Decode HBM.

#### Scenario: Multi-step generation consistency
- **WHEN** a request generates N tokens with tiering enabled
- **THEN** tokens generated at every step attend to the full history including KV written back by earlier steps

#### Scenario: P/D handoff to host pool
- **WHEN** a prefill node finishes a request's prefill in a disaggregated deployment
- **THEN** the request's streaming-layer KV arrives in the Decode host pool and the first decode step onloads it correctly

### Requirement: Linear-attention state exclusion

Linear-attention (MambaSpec) groups SHALL remain entirely HBM-resident and SHALL NOT be affected by tiering in allocation, transfer, or compute paths.

#### Scenario: KDA state untouched
- **WHEN** tiering is enabled for a hybrid linear model
- **THEN** linear-attention layers use the same HBM state tensors and kernels as without tiering, with no additional copies
