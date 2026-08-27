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

from types import SimpleNamespace

import pytest

from vllm_ascend.ascend_config import AscendConfig, StreamingKVTieringConfig


def fake_vllm_config(max_model_len=512 * 1024):
    return SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=max_model_len),
        use_v2_model_runner=False,
    )


class TestStreamingKVTieringConfig:
    def test_disabled_by_default(self):
        cfg = StreamingKVTieringConfig(fake_vllm_config(), {})
        assert not cfg.enabled

    def test_enabled_defaults(self):
        cfg = StreamingKVTieringConfig(fake_vllm_config(), {"enabled": True})
        assert cfg.enabled
        assert cfg.cpu_budget_bytes > 0
        assert cfg.prefetch_depth == 3
        assert cfg.staging_max_tokens == 512 * 1024
        assert cfg.stream_layers is None

    def test_explicit_values(self):
        cfg = StreamingKVTieringConfig(
            fake_vllm_config(),
            {
                "enabled": True,
                "hbm_budget_bytes": 1 << 33,
                "cpu_budget_bytes": 1 << 34,
                "host_bandwidth_bytes_per_s": 5e11,
                "target_tps": 200,
                "stream_layers": ["a", "b"],
                "prefetch_depth": 4,
                "staging_max_tokens": 128 * 1024,
                "allow_low_bandwidth": True,
            },
        )
        assert cfg.hbm_budget_bytes == 1 << 33
        assert cfg.stream_layers == ["a", "b"]
        assert cfg.prefetch_depth == 4
        assert cfg.allow_low_bandwidth

    def test_stream_layers_type_validated(self):
        with pytest.raises(ValueError, match="list of layer names"):
            StreamingKVTieringConfig(fake_vllm_config(), {"enabled": True, "stream_layers": "not-a-list"})

    def test_prefetch_depth_validated(self):
        with pytest.raises(ValueError, match="prefetch_depth"):
            StreamingKVTieringConfig(fake_vllm_config(), {"enabled": True, "prefetch_depth": 0})


def bare_ascend_config(**attrs) -> AscendConfig:
    cfg = AscendConfig.__new__(AscendConfig)
    cfg.sparse_kv_offload_config = SimpleNamespace(enabled=False)
    cfg.scheduler_config = SimpleNamespace(recompute_scheduler_enable=False)
    cfg.vllm_config = fake_vllm_config()
    for k, v in attrs.items():
        setattr(cfg, k, v)
    return cfg


class TestMutualExclusion:
    def _cfg(self, enabled=True):
        return SimpleNamespace(enabled=enabled)

    def test_rejects_sparse_kv_offload_combination(self):
        cfg = bare_ascend_config(streaming_kv_config=self._cfg(True))
        cfg.sparse_kv_offload_config = SimpleNamespace(enabled=True)
        with pytest.raises(ValueError, match="sparse_kv_offload_config"):
            cfg._validate_streaming_kv_compatibility()

    def test_rejects_recompute_scheduler_combination(self):
        cfg = bare_ascend_config(streaming_kv_config=self._cfg(True))
        cfg.scheduler_config = SimpleNamespace(recompute_scheduler_enable=True)
        with pytest.raises(ValueError, match="recompute scheduler"):
            cfg._validate_streaming_kv_compatibility()

    def test_rejects_v2_runner(self):
        cfg = bare_ascend_config(streaming_kv_config=self._cfg(True))
        cfg.vllm_config = SimpleNamespace(use_v2_model_runner=True)
        with pytest.raises(ValueError, match="model_runner_v2"):
            cfg._validate_streaming_kv_compatibility()

    def test_disabled_skips_validation(self):
        cfg = bare_ascend_config(streaming_kv_config=self._cfg(False))
        cfg.sparse_kv_offload_config = SimpleNamespace(enabled=True)
        cfg._validate_streaming_kv_compatibility()  # no raise
