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
"""Streaming-group capacity accounting helpers (design.md D2).

The streaming group's block ids come from the shared block pool like any
other group (keeping block-table / prefix-cache machinery intact), but its
pages are interpreted as rows of the Decode host pool: the group contributes
zero to the HBM bytes-per-block computation, and the CPU budget is validated
to cover num_blocks x streaming pages.
"""

from vllm.v1.kv_cache_interface import KVCacheGroupSpec, UniformTypeKVCacheSpecs


def is_host_streaming_group(group: KVCacheGroupSpec) -> bool:
    """True when the group's spec(s) are marked store_on_host."""
    spec = group.kv_cache_spec
    if isinstance(spec, UniformTypeKVCacheSpecs):
        return any(getattr(s, "store_on_host", False) for s in spec.kv_cache_specs.values())
    return getattr(spec, "store_on_host", False)


def _per_layer_spec(group: KVCacheGroupSpec, layer_name: str):
    spec = group.kv_cache_spec
    if isinstance(spec, UniformTypeKVCacheSpecs):
        return spec.kv_cache_specs[layer_name]
    return spec


def group_bytes_per_block(group: KVCacheGroupSpec) -> int:
    """Sum of per-layer page sizes in the group (bytes per block row)."""
    return sum(_per_layer_spec(group, name).page_size_bytes for name in group.layer_names)


def validate_cpu_budget(
    host_groups: list[KVCacheGroupSpec],
    num_blocks: int,
    cpu_budget_bytes: int,
) -> None:
    """Raise when the CPU budget cannot back num_blocks of the host groups.

    Must be checked after num_blocks is finalized: excluding the host groups
    from HBM sizing *increases* num_blocks, which in turn raises the host
    pool requirement.
    """
    required = sum(group_bytes_per_block(g) for g in host_groups) * num_blocks
    if required > cpu_budget_bytes:
        raise ValueError(
            "streaming_kv_config.cpu_budget_bytes is too small: "
            f"need {required / 1e9:.1f}GB for {num_blocks} blocks of "
            f"{sum(len(g.layer_names) for g in host_groups)} streaming layers, "
            f"configured {cpu_budget_bytes / 1e9:.1f}GB. "
            "Raise the CPU budget, or reduce streaming layers / max blocks."
        )
