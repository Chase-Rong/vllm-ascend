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
"""Activation gating for streaming KV tiering (design.md D6).

Tiering activates only for hybrid linear models: the KV cache config must
mix linear-attention (MambaSpec) groups with full-attention groups. This is
the machine-checkable condition; no model names are involved. Phase 1
additionally requires every streaming layer to use AscendMLAAttentionSpec.
"""

from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheSpec, MambaSpec

from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec


def is_hybrid_linear_spec_dict(kv_cache_spec: dict[str, KVCacheSpec]) -> bool:
    """Spec-dict variant of :func:`is_hybrid_linear_config`, usable at spec
    creation time before KV cache groups are formed."""
    has_linear = any(isinstance(spec, MambaSpec) for spec in kv_cache_spec.values())
    has_full_attn = any(
        isinstance(spec, FullAttentionSpec) and not isinstance(spec, MambaSpec)
        for spec in kv_cache_spec.values()
    )
    return has_linear and has_full_attn


def is_hybrid_linear_config(kv_cache_config: KVCacheConfig) -> bool:
    """True when the model mixes linear-attention and full-attention groups."""
    has_linear = False
    has_full_attn = False
    for group in kv_cache_config.kv_cache_groups:
        spec = group.kv_cache_spec
        specs = getattr(spec, "kv_cache_specs", None)
        flat = list(specs.values()) if isinstance(specs, dict) else [spec]
        for s in flat:
            if isinstance(s, MambaSpec):
                has_linear = True
            elif isinstance(s, FullAttentionSpec):
                # AscendMLAAttentionSpec derives from MLAAttentionSpec which
                # derives from FullAttentionSpec; MambaSpec is disjoint.
                has_full_attn = True
    return has_linear and has_full_attn


def assert_streamable_layers(kv_cache_config: KVCacheConfig, stream_layers: list[str]) -> None:
    """Phase-1 gate: every streaming layer must be an AscendMLAAttentionSpec."""
    spec_by_layer: dict[str, object] = {}
    for group in kv_cache_config.kv_cache_groups:
        spec = group.kv_cache_spec
        specs = getattr(spec, "kv_cache_specs", None)
        if isinstance(specs, dict):
            spec_by_layer.update(specs)
        else:
            for name in group.layer_names:
                spec_by_layer[name] = spec

    bad = [
        name for name in stream_layers
        if not isinstance(spec_by_layer.get(name), AscendMLAAttentionSpec)
    ]
    if bad:
        raise ValueError(
            "Streaming KV tiering phase-1 supports only AscendMLAAttentionSpec layers; "
            f"these streaming layers have another spec: {bad}"
        )


def assert_gating(kv_cache_config: KVCacheConfig, stream_layers: list[str]) -> None:
    """Full startup gate. Raises ValueError with an actionable message."""
    if not is_hybrid_linear_config(kv_cache_config):
        raise ValueError(
            "Streaming KV tiering requires a hybrid linear-attention model "
            "(MambaSpec + full-attention KV cache groups). "
            "Refusing to enable for this model."
        )
    assert_streamable_layers(kv_cache_config, stream_layers)
