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
"""Static residency policy for streaming KV tiering.

Computes which full-attention layers of a hybrid linear model keep their KV
HBM-resident and which stream from the Decode host pool, once at startup.
The plan is fixed for the process lifetime (design.md D5).

Core accounting (per decode step, per sequence):

    tps_ceiling   = B_h2d / (f * KV_bytes_per_seq)     # aggregate, batch-independent
    hbm_per_seq   = (1 - f) * KV_bytes_per_seq
    staging_bytes = SLOTS * max_batch * per_layer_bytes(max_model_len)
    net saving    = f * per_layer_bytes - staging_bytes  (f > SLOTS for gain)
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

from vllm.logger import logger

# Number of HBM staging buffer slots rotated across streaming layers.
# slot_capacity = 2 is the minimum for producer/consumer overlap; deeper
# rotation buys little because the copy stream is consumed in layer order.
STAGING_SLOTS = 2


@dataclass(frozen=True)
class FullAttentionLayerInfo:
    """Per-layer inputs the policy needs, extracted from KV cache specs."""

    layer_name: str
    page_size_bytes: int  # spec.page_size_bytes (one block)
    block_size: int  # tokens per block

    def kv_bytes_for_tokens(self, num_tokens: int) -> int:
        return math.ceil(num_tokens / self.block_size) * self.page_size_bytes


@dataclass
class ResidencyPlan:
    """Output of the policy: which layers stream, and the derived numbers."""

    stream_layers: list[str]
    resident_layers: list[str]
    per_sequence_hbm_bytes_before: int
    per_sequence_hbm_bytes_after: int
    per_sequence_host_bytes: int
    staging_bytes: int
    tps_ceiling: float  # aggregate tokens/s bound from host bandwidth
    warnings: list[str] = field(default_factory=list)

    @property
    def num_stream_layers(self) -> int:
        return len(self.stream_layers)

    @property
    def per_sequence_hbm_savings(self) -> int:
        return self.per_sequence_hbm_bytes_before - self.per_sequence_hbm_bytes_after


def compute_residency_plan(
    full_attn_layers: Sequence[FullAttentionLayerInfo],
    *,
    max_model_len: int,
    max_batch_size: int,
    hbm_budget_bytes: int,
    host_bandwidth_bytes_per_s: float,
    target_tps: float | None = None,
    stream_layers_override: Sequence[str] | None = None,
    staging_slots: int = STAGING_SLOTS,
) -> ResidencyPlan:
    """Compute the static residency plan.

    Args:
        full_attn_layers: all full-attention layers (any attention spec type;
            callers pre-filter to streamable spec types for phase-1 gating).
        max_model_len: used for per-sequence KV sizing.
        max_batch_size: used for staging sizing.
        hbm_budget_bytes: per-rank HBM budget available for full-attention KV.
        host_bandwidth_bytes_per_s: measured or configured B_h2d.
        target_tps: optional target aggregate throughput; a warning is emitted
            when the plan's ceiling falls below it.
        stream_layers_override: explicit streaming layer names; skips the
            automatic selection (validated below).
        staging_slots: staging buffer rotation depth.

    Returns:
        ResidencyPlan. Empty ``stream_layers`` means "tiering degenerates to
        all-resident" (callers should treat that as feature-off).
    """
    if not full_attn_layers:
        raise ValueError("compute_residency_plan requires at least one full-attention layer")
    if host_bandwidth_bytes_per_s <= 0:
        raise ValueError("host_bandwidth_bytes_per_s must be positive; run probe_host_bandwidth.py or configure it")

    names = [info.layer_name for info in full_attn_layers]
    info_by_name = dict(zip(names, full_attn_layers))
    per_seq_total = sum(info.kv_bytes_for_tokens(max_model_len) for info in full_attn_layers)
    per_layer_staging = max(info.kv_bytes_for_tokens(max_model_len) for info in full_attn_layers)
    staging_bytes = staging_slots * max_batch_size * per_layer_staging
    warnings: list[str] = []

    if stream_layers_override is not None:
        unknown = [n for n in stream_layers_override if n not in info_by_name]
        if unknown:
            raise ValueError(f"stream_layers contains non-full-attention or unknown layers: {unknown}")
        stream_layers = list(stream_layers_override)
    else:
        stream_layers = _auto_select(
            full_attn_layers,
            max_model_len=max_model_len,
            hbm_budget_bytes=hbm_budget_bytes,
            staging_bytes=staging_bytes,
        )

    if not stream_layers:
        warnings.append("residency plan selected zero streaming layers; tiering has no effect")

    streamed_per_seq = sum(info_by_name[n].kv_bytes_for_tokens(max_model_len) for n in stream_layers)
    resident_per_seq = per_seq_total - streamed_per_seq
    tps_ceiling = host_bandwidth_bytes_per_s / streamed_per_seq if streamed_per_seq else float("inf")

    if len(stream_layers) <= staging_slots:
        warnings.append(
            f"only {len(stream_layers)} streaming layer(s) vs {staging_slots} staging slots: "
            "staging cost exceeds what offloading frees; consider more streaming layers"
        )
    if target_tps is not None and tps_ceiling < target_tps:
        warnings.append(
            f"TPS ceiling {tps_ceiling:.1f} < target {target_tps:.1f} "
            f"(B_h2d={host_bandwidth_bytes_per_s / 1e9:.0f}GB/s, "
            f"streamed {streamed_per_seq / 1e9:.2f}GB/seq). "
            "Reduce streaming layers or raise host bandwidth."
        )

    plan = ResidencyPlan(
        stream_layers=stream_layers,
        resident_layers=[n for n in names if n not in set(stream_layers)],
        per_sequence_hbm_bytes_before=per_seq_total,
        per_sequence_hbm_bytes_after=resident_per_seq,
        per_sequence_host_bytes=streamed_per_seq,
        staging_bytes=staging_bytes,
        tps_ceiling=tps_ceiling,
        warnings=warnings,
    )
    for w in warnings:
        logger.warning("streaming-kv policy: %s", w)
    logger.info(
        "streaming-kv residency plan: %d/%d full-attn layers streaming, "
        "per-seq HBM %.2fGB -> %.2fGB (host %.2fGB), staging %.2fGB, TPS ceiling %.1f",
        plan.num_stream_layers,
        len(names),
        per_seq_total / 1e9,
        resident_per_seq / 1e9,
        streamed_per_seq / 1e9,
        staging_bytes / 1e9,
        tps_ceiling,
    )
    return plan


def _auto_select(
    full_attn_layers: Sequence[FullAttentionLayerInfo],
    *,
    max_model_len: int,
    hbm_budget_bytes: int,
    staging_bytes: int,
) -> list[str]:
    """Pick the smallest set of layers whose streaming satisfies the HBM budget.

    Selection is from the tail of the network first (later layers free the
    same bytes; keeps early-layer latency low).
    """
    per_seq_total = sum(info.kv_bytes_for_tokens(max_model_len) for info in full_attn_layers)
    # Budget must cover resident KV + staging.
    resident_allowance = hbm_budget_bytes - staging_bytes
    if resident_allowance >= per_seq_total:
        return []
    if resident_allowance < 0:
        # Staging alone busts the budget: still stream as many as needed.
        resident_allowance = 0

    stream_layers: list[str] = []
    resident_per_seq = per_seq_total
    for info in reversed(full_attn_layers):
        if resident_per_seq <= resident_allowance:
            break
        stream_layers.append(info.layer_name)
        resident_per_seq -= info.kv_bytes_for_tokens(max_model_len)
    position = {info.layer_name: i for i, info in enumerate(full_attn_layers)}
    stream_layers.sort(key=position.__getitem__)  # restore network order
    return stream_layers
