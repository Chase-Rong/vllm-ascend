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
"""Streaming KV manager: host pool, HBM staging, and the onload/writeback
event machinery (design doc §3-§5).

Phase-1 layout decision (recorded in the change's design.md): the host pool
is *per-request contiguous* — host_pool[layer][req_slot] holds that request's
whole KV for the layer as one contiguous region. Onload is then a single
strided H2D copy and writeback a tail D2H copy, instead of a block-table-
driven gather over a globally paged host pool. Consequences:

  + one big copy per layer per step (best bandwidth, trivially correct)
  - the streaming group does not participate in prefix cache sharing
    (hybrid prefix hits are capped by the streaming group's miss); the
    paged, block-table-driven pool (probe-dependent, G1/G2 in design.md)
    restores that in a later phase.

Staging slots rotate across streaming layers (SLOTS=2): at any moment only
SLOTS layers' KV is in flight on HBM, which is what makes the whole scheme a
net HBM win (net saving = (f - SLOTS) x per-layer-batch-KV).
"""

import torch
from vllm.config import VllmConfig
from vllm.logger import logger
from vllm.v1.kv_cache_interface import KVCacheConfig

from vllm_ascend.distributed.kv_transfer.streaming_kv.policy import (
    STAGING_SLOTS,
    ResidencyPlan,
)

_STREAMING_KV_MANAGER: "StreamingKVManager | None" = None


class StreamingKVManager:
    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        plan: ResidencyPlan,
        max_num_reqs: int,
        staging_max_tokens: int,
        prefetch_depth: int,
    ):
        self.vllm_config = vllm_config
        self.plan = plan
        self.stream_layers: list[str] = list(plan.stream_layers)
        self.layer_index: dict[str, int] = {n: i for i, n in enumerate(self.stream_layers)}
        self.max_num_reqs = max_num_reqs
        self.staging_max_tokens = staging_max_tokens
        self.prefetch_depth = prefetch_depth

        # All phase-1 streaming layers are AscendMLAAttentionSpec: single
        # latent vector per token, identical layout across layers.
        spec = self._find_streaming_spec(kv_cache_config)
        self.head_size = spec.head_size
        self.dtype = spec.dtype

        # Host pool: [num_layers][max_reqs, staging_max_tokens, head_size]
        # pinned, per-request contiguous (see module docstring).
        pool_shape = (max_num_reqs, staging_max_tokens, self.head_size)
        self.host_pool: list[torch.Tensor] = []
        for name in self.stream_layers:
            buf = torch.empty(pool_shape, dtype=self.dtype, pin_memory=True)
            buf.zero_()
            self.host_pool.append(buf)
        host_gb = len(self.host_pool) * self.host_pool[0].numel() * self.host_pool[0].element_size() / 1e9

        # HBM staging slots, shared by all streaming layers via rotation.
        self.staging: list[torch.Tensor] = [
            torch.zeros(pool_shape, dtype=self.dtype, device="npu") for _ in range(STAGING_SLOTS)
        ]
        staging_gb = sum(s.numel() * s.element_size() for s in self.staging) / 1e9

        self.copy_stream = torch.npu.Stream()
        self._done_events: list[torch.npu.Event] = [torch.npu.Event() for _ in self.stream_layers]
        # eager/capture bookkeeping mirrors vllm's PrefetchOffloader:
        self._in_capture: list[bool] = [False] * len(self.stream_layers)
        self._event_valid_for_eager: list[bool] = [False] * len(self.stream_layers)

        logger.info(
            "streaming-kv manager: %d streaming layers, host pool %.2fGB, "
            "staging %d slots %.2fGB, per-req max tokens %d",
            len(self.stream_layers), host_gb, STAGING_SLOTS, staging_gb, staging_max_tokens,
        )

    def _find_streaming_spec(self, kv_cache_config: KVCacheConfig):
        stream_set = set(self.stream_layers)
        for group in kv_cache_config.kv_cache_groups:
            spec = group.kv_cache_spec
            specs = getattr(spec, "kv_cache_specs", None)
            if isinstance(specs, dict):
                for name, s in specs.items():
                    if name in stream_set:
                        return s
            elif any(n in stream_set for n in group.layer_names):
                return spec
        raise RuntimeError("streaming layers not found in kv_cache_config")

    # ------------------------------------------------------------------
    # attention-facing accessors
    # ------------------------------------------------------------------
    def slot_of(self, layer_name: str) -> int:
        return self.layer_index[layer_name] % STAGING_SLOTS

    def staging_buffer(self, layer_name: str) -> torch.Tensor:
        """The HBM buffer the layer's attention reads/writes this step."""
        return self.staging[self.slot_of(layer_name)]

    # ------------------------------------------------------------------
    # onload: host pool -> staging (fork on copy stream)
    # ------------------------------------------------------------------
    def start_onload(self, layer_name: str, num_reqs: int) -> None:
        idx = self.layer_index[layer_name]
        slot = idx % STAGING_SLOTS
        capturing = torch.npu.is_current_stream_capturing()
        self._in_capture[idx] = capturing

        fork = torch.npu.Event()
        torch.npu.current_stream().record_event(fork)
        self.copy_stream.wait_event(fork)
        with torch.npu.stream(self.copy_stream):
            dst = self.staging[slot][:num_reqs]
            dst.copy_(self.host_pool[idx][:num_reqs], non_blocking=True)
            self._done_events[idx].record(self.copy_stream)
        self._event_valid_for_eager[idx] = not capturing

    def wait_onload(self, layer_name: str) -> None:
        idx = self.layer_index[layer_name]
        if torch.npu.is_current_stream_capturing():
            if not self._in_capture[idx]:
                return  # pre-capture onload already synced by sync_before_capture
            torch.npu.current_stream().wait_event(self._done_events[idx])
            self._in_capture[idx] = False
        else:
            if self._event_valid_for_eager[idx]:
                torch.npu.current_stream().wait_event(self._done_events[idx])
            else:
                torch.npu.current_stream().wait_stream(self.copy_stream)

    # ------------------------------------------------------------------
    # writeback: staging tail -> host pool (D2H on copy stream)
    # ------------------------------------------------------------------
    def start_writeback(self, layer_name: str, prev_seq_lens: list[int], cur_seq_lens: list[int]) -> None:
        """Copy the newly appended rows of each request back to the host pool.

        Phase-1 eager path: per-request tail copies sized by CPU-side seq
        lens. Graph-mode writeback (device-computed regions) is part of the
        capture task (5.1).
        """
        idx = self.layer_index[layer_name]
        slot = idx % STAGING_SLOTS
        fork = torch.npu.Event()
        torch.npu.current_stream().record_event(fork)
        self.copy_stream.wait_event(fork)
        with torch.npu.stream(self.copy_stream):
            for req, (prev, cur) in enumerate(zip(prev_seq_lens, cur_seq_lens)):
                if cur > prev:
                    self.host_pool[idx][req, prev:cur].copy_(
                        self.staging[slot][req, prev:cur], non_blocking=True
                    )

    # ------------------------------------------------------------------
    # capture lifecycle (called from acl_graph.py hook points)
    # ------------------------------------------------------------------
    def sync_before_capture(self) -> None:
        torch.npu.current_stream().wait_stream(self.copy_stream)

    def join_after_forward(self) -> None:
        for idx, in_cap in enumerate(self._in_capture):
            if in_cap:
                torch.npu.current_stream().wait_event(self._done_events[idx])
                self._in_capture[idx] = False


def init_streaming_kv_manager(
    vllm_config: VllmConfig,
    kv_cache_config: KVCacheConfig,
    plan: ResidencyPlan,
    streaming_config,
    max_num_reqs: int,
) -> StreamingKVManager:
    global _STREAMING_KV_MANAGER
    manager = StreamingKVManager(
        vllm_config,
        kv_cache_config,
        plan,
        max_num_reqs=max_num_reqs,
        staging_max_tokens=streaming_config.staging_max_tokens,
        prefetch_depth=streaming_config.prefetch_depth,
    )
    _STREAMING_KV_MANAGER = manager
    return manager


def get_streaming_kv_manager() -> StreamingKVManager:
    assert _STREAMING_KV_MANAGER is not None, "StreamingKVManager is not initialized"
    return _STREAMING_KV_MANAGER
