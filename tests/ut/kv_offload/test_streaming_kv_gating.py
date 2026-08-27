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

import pytest
import torch
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)

from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec
from vllm_ascend.distributed.kv_transfer.streaming_kv.gating import (
    assert_gating,
    assert_streamable_layers,
    is_hybrid_linear_config,
)

MLA_LAYERS = [f"model.layers.{i}.self_attn.attn" for i in range(0, 96, 4)]
KDA_LAYERS = [f"model.layers.{i}.linear_attn" for i in range(96) if i % 4 != 0]


def mla_spec(store_on_host: bool = False) -> AscendMLAAttentionSpec:
    return AscendMLAAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=576,
        dtype=torch.bfloat16,
        store_on_host=store_on_host,
    )


def mamba_spec() -> MambaSpec:
    return MambaSpec(
        block_size=128,
        shapes=((96, 128, 128),),
        dtypes=(torch.bfloat16,),
    )


def gqa_spec() -> FullAttentionSpec:
    return FullAttentionSpec(
        block_size=128,
        num_kv_heads=8,
        head_size=128,
        dtype=torch.bfloat16,
    )


def make_config(groups: list[KVCacheGroupSpec]) -> KVCacheConfig:
    return KVCacheConfig(num_blocks=1000, kv_cache_tensors=[], kv_cache_groups=groups)


def k3_like() -> KVCacheConfig:
    return make_config([
        KVCacheGroupSpec(KDA_LAYERS, mamba_spec()),
        KVCacheGroupSpec(MLA_LAYERS, mla_spec()),
    ])


class TestHybridDetection:
    def test_k3_like_is_hybrid(self):
        assert is_hybrid_linear_config(k3_like())

    def test_qwen35_like_is_hybrid(self):
        cfg = make_config([
            KVCacheGroupSpec(KDA_LAYERS, mamba_spec()),
            KVCacheGroupSpec(MLA_LAYERS, gqa_spec()),
        ])
        assert is_hybrid_linear_config(cfg)

    def test_dense_model_is_not_hybrid(self):
        cfg = make_config([KVCacheGroupSpec(MLA_LAYERS, gqa_spec())])
        assert not is_hybrid_linear_config(cfg)
        with pytest.raises(ValueError, match="hybrid linear"):
            assert_gating(cfg, [])

    def test_pure_linear_model_is_not_hybrid(self):
        cfg = make_config([KVCacheGroupSpec(KDA_LAYERS, mamba_spec())])
        assert not is_hybrid_linear_config(cfg)


class TestStreamableLayers:
    def test_mla_layers_pass(self):
        assert_streamable_layers(k3_like(), MLA_LAYERS[-6:])

    def test_gqa_layers_rejected_in_phase1(self):
        cfg = make_config([
            KVCacheGroupSpec(KDA_LAYERS, mamba_spec()),
            KVCacheGroupSpec(MLA_LAYERS, gqa_spec()),
        ])
        with pytest.raises(ValueError, match="AscendMLAAttentionSpec"):
            assert_gating(cfg, MLA_LAYERS[-6:])

    def test_kda_layer_in_stream_list_rejected(self):
        with pytest.raises(ValueError, match="AscendMLAAttentionSpec"):
            assert_streamable_layers(k3_like(), [KDA_LAYERS[0]])

    def test_split_groups_both_streamable(self):
        # resident + streaming groups as produced after store_on_host marking
        cfg = make_config([
            KVCacheGroupSpec(KDA_LAYERS, mamba_spec()),
            KVCacheGroupSpec(MLA_LAYERS[:-6], mla_spec(store_on_host=False)),
            KVCacheGroupSpec(MLA_LAYERS[-6:], mla_spec(store_on_host=True)),
        ])
        assert_gating(cfg, MLA_LAYERS[-6:])
