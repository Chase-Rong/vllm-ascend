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

from vllm_ascend.distributed.kv_transfer.streaming_kv.policy import (
    STAGING_SLOTS,
    FullAttentionLayerInfo,
    compute_residency_plan,
)

# K3-like: 24 MLA layers, block 128, page = 128 * 576 * 2B = 147456 B
PAGE = 128 * 576 * 2
BLOCK = 128
LAYERS = [f"model.layers.{i}.self_attn.attn" for i in range(0, 96, 4)]  # 24 layers
MAX_LEN = 512 * 1024
BATCH = 32
GB = 1 << 30


def make_infos() -> list[FullAttentionLayerInfo]:
    return [FullAttentionLayerInfo(layer_name=n, page_size_bytes=PAGE, block_size=BLOCK) for n in LAYERS]


def per_layer_seq_bytes() -> int:
    return (MAX_LEN + BLOCK - 1) // BLOCK * PAGE


class TestAutoSelection:
    def test_budget_fits_all_resident_selects_nothing(self):
        plan = compute_residency_plan(
            make_infos(),
            max_model_len=MAX_LEN,
            max_batch_size=BATCH,
            hbm_budget_bytes=10 * GB * BATCH,
            host_bandwidth_bytes_per_s=1e12,
        )
        assert plan.stream_layers == []
        assert plan.num_stream_layers == 0
        assert any("zero streaming layers" in w for w in plan.warnings)

    def test_selects_tail_layers_until_budget_met(self):
        per_layer = per_layer_seq_bytes()
        staging = STAGING_SLOTS * BATCH * per_layer
        total = 24 * per_layer
        # Allow resident of exactly 20 layers + staging => 4 layers streamed.
        budget = 20 * per_layer + staging
        plan = compute_residency_plan(
            make_infos(),
            max_model_len=MAX_LEN,
            max_batch_size=BATCH,
            hbm_budget_bytes=budget,
            host_bandwidth_bytes_per_s=1e12,
        )
        assert len(plan.stream_layers) == 4
        # tail-first selection, reported in network order
        assert plan.stream_layers == LAYERS[-4:]
        assert plan.per_sequence_hbm_bytes_after == 20 * per_layer
        assert plan.per_sequence_host_bytes == 4 * per_layer
        assert plan.staging_bytes == staging

    def test_tps_ceiling_math(self):
        per_layer = per_layer_seq_bytes()
        plan = compute_residency_plan(
            make_infos(),
            max_model_len=MAX_LEN,
            max_batch_size=BATCH,
            hbm_budget_bytes=0,  # stream everything
            host_bandwidth_bytes_per_s=1e12,
        )
        assert plan.num_stream_layers == 24
        assert plan.tps_ceiling == pytest.approx(1e12 / (24 * per_layer))


class TestOverride:
    def test_explicit_layers_respected(self):
        chosen = LAYERS[2:5]
        plan = compute_residency_plan(
            make_infos(),
            max_model_len=MAX_LEN,
            max_batch_size=BATCH,
            hbm_budget_bytes=10 * GB * BATCH,  # budget irrelevant under override
            host_bandwidth_bytes_per_s=1e12,
            stream_layers_override=chosen,
        )
        assert plan.stream_layers == chosen

    def test_unknown_layer_rejected(self):
        with pytest.raises(ValueError, match="unknown layers"):
            compute_residency_plan(
                make_infos(),
                max_model_len=MAX_LEN,
                max_batch_size=BATCH,
                hbm_budget_bytes=0,
                host_bandwidth_bytes_per_s=1e12,
                stream_layers_override=["no.such.layer"],
            )


class TestWarnings:
    def test_low_bandwidth_target_warning(self):
        plan = compute_residency_plan(
            make_infos(),
            max_model_len=MAX_LEN,
            max_batch_size=BATCH,
            hbm_budget_bytes=0,
            host_bandwidth_bytes_per_s=1e9,  # 1 GB/s: ceiling far below target
            target_tps=100.0,
        )
        assert any("TPS ceiling" in w for w in plan.warnings)

    def test_few_layers_vs_staging_slots_warning(self):
        plan = compute_residency_plan(
            make_infos(),
            max_model_len=MAX_LEN,
            max_batch_size=BATCH,
            hbm_budget_bytes=0,
            host_bandwidth_bytes_per_s=1e12,
            stream_layers_override=LAYERS[:1],
        )
        assert any("staging slots" in w for w in plan.warnings)

    def test_nonpositive_bandwidth_rejected(self):
        with pytest.raises(ValueError, match="bandwidth"):
            compute_residency_plan(
                make_infos(),
                max_model_len=MAX_LEN,
                max_batch_size=BATCH,
                hbm_budget_bytes=0,
                host_bandwidth_bytes_per_s=0,
            )
