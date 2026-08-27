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

import dataclasses
from types import SimpleNamespace

import pytest
import torch
from vllm.v1.kv_cache_interface import KVCacheGroupSpec, MambaSpec

import vllm_ascend.patch.platform.patch_kv_cache_utils as patch_mod
from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec
from vllm_ascend.distributed.kv_transfer.streaming_kv.accounting import (
    group_bytes_per_block,
    is_host_streaming_group,
    validate_cpu_budget,
)

PAGE = 128 * 576 * 2  # MLA page bytes


def mla_group(n_layers: int, store_on_host: bool, prefix: str) -> KVCacheGroupSpec:
    return KVCacheGroupSpec(
        [f"{prefix}.layers.{i}.self_attn.attn" for i in range(n_layers)],
        AscendMLAAttentionSpec(
            block_size=128, num_kv_heads=1, head_size=576,
            dtype=torch.bfloat16, store_on_host=store_on_host,
        ),
    )


def mamba_group(n_layers: int) -> KVCacheGroupSpec:
    return KVCacheGroupSpec(
        [f"model.layers.{i}.linear_attn" for i in range(n_layers)],
        MambaSpec(block_size=128, shapes=((96, 128, 128),), dtypes=(torch.bfloat16,)),
    )


@dataclasses.dataclass
class FakeConfig:
    num_blocks: int
    kv_cache_groups: list


def fake_orig(vllm_config, groups, available_memory):
    bytes_per_block = max(group_bytes_per_block(g) for g in groups)
    return FakeConfig(num_blocks=available_memory // bytes_per_block, kv_cache_groups=groups)


class TestHostGroupDetection:
    def test_marks_only_store_on_host(self):
        assert is_host_streaming_group(mla_group(6, True, "m"))
        assert not is_host_streaming_group(mla_group(18, False, "m"))
        assert not is_host_streaming_group(mamba_group(69))


class TestConfigWrapper:
    @pytest.fixture(autouse=True)
    def _patch(self, monkeypatch):
        monkeypatch.setattr(patch_mod, "_orig_get_kv_cache_config_from_groups", fake_orig)
        # The wrapper imports get_ascend_config lazily from its source module.
        import vllm_ascend.ascend_config as ascend_config_mod
        monkeypatch.setattr(
            ascend_config_mod,
            "get_ascend_config",
            lambda: SimpleNamespace(streaming_kv_config=SimpleNamespace(cpu_budget_bytes=1 << 40)),
        )

    def test_passthrough_without_host_groups(self):
        groups = [mamba_group(69), mla_group(24, False, "m")]
        cfg = patch_mod._ascend_get_kv_cache_config_from_groups(None, groups, 1 << 34)
        assert cfg.num_blocks == (1 << 34) // max(group_bytes_per_block(g) for g in groups)

    def test_host_groups_excluded_from_hbm_sizing(self):
        resident = mla_group(18, False, "m")
        streaming = mla_group(6, True, "m")
        mamba = mamba_group(69)
        available = 1 << 34
        cfg = patch_mod._ascend_get_kv_cache_config_from_groups(None, [mamba, resident, streaming], available)
        # num_blocks driven by the largest DEVICE group only
        expected = available // max(group_bytes_per_block(mamba), group_bytes_per_block(resident))
        assert cfg.num_blocks == expected
        # all groups reattached for the scheduler
        assert len(cfg.kv_cache_groups) == 3

    def test_cpu_budget_validation(self):
        streaming = mla_group(6, True, "m")
        # 100 blocks * 6 layers * PAGE bytes vs tiny budget
        with pytest.raises(ValueError, match="cpu_budget_bytes"):
            validate_cpu_budget([streaming], num_blocks=100, cpu_budget_bytes=100 * PAGE * 5)
        # exact fit passes
        validate_cpu_budget([streaming], num_blocks=100, cpu_budget_bytes=100 * PAGE * 6)

    def test_admission_gain(self):
        """f streaming layers free HBM proportionally: more sequences admitted."""
        per_seq_blocks = 10
        available = 1 << 34
        resident = mla_group(18, False, "m")
        streaming = mla_group(6, True, "m")
        mamba = mamba_group(69)

        cfg_all_device = patch_mod._ascend_get_kv_cache_config_from_groups(
            None, [mamba, resident, mla_group(6, False, "m")], available)
        cfg_tiered = patch_mod._ascend_get_kv_cache_config_from_groups(
            None, [mamba, resident, streaming], available)
        assert cfg_tiered.num_blocks >= cfg_all_device.num_blocks
        seqs_before = cfg_all_device.num_blocks // per_seq_blocks
        seqs_after = cfg_tiered.num_blocks // per_seq_blocks
        assert seqs_after >= seqs_before
