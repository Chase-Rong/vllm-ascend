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
"""Init-time host<->NPU bandwidth micro-benchmark.

Used when streaming_kv_config.host_bandwidth_bytes_per_s is 0. For stable,
reproducible plans prefer configuring the value explicitly from
tools/streaming_kv/probe_host_bandwidth.py output.
"""

import time

import torch

_BENCH_BYTES = 256 * 1024 * 1024  # 256 MiB
_BENCH_ITERS = 10


def measure_host_bandwidth(num_bytes: int = _BENCH_BYTES, iters: int = _BENCH_ITERS) -> float:
    """Measured SDMA H2D bandwidth in bytes/s. Requires an initialized NPU."""
    numel = num_bytes // 2
    host = torch.empty(numel, dtype=torch.bfloat16, pin_memory=True)
    dev = torch.empty(numel, dtype=torch.bfloat16, device="npu")
    stream = torch.npu.current_stream()
    for _ in range(3):
        dev.copy_(host, non_blocking=True)
    stream.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        dev.copy_(host, non_blocking=True)
    stream.synchronize()
    elapsed = time.perf_counter() - start
    del host, dev
    torch.npu.empty_cache()
    return num_bytes * iters / elapsed
