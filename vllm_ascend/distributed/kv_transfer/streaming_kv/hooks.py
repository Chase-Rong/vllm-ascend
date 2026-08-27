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
"""Decoder-layer forward hooks for streaming KV tiering (design doc §4.2).

Wraps every decoder layer's forward so that:
  - on ENTRY of a streaming layer: join its onload (wait_onload);
  - on EXIT of an issue-point layer: fork the onload of the streaming layer
    scheduled at this point (start_onload);
  - on EXIT of a streaming layer: writeback its new KV rows to the host pool.

Issue points are intra-step only (design doc §5.2: prefetch must never cross
a step boundary, because the block table / request set can change between
steps).

Model-generic: layers are discovered by the `layer_idx` attribute that
decoder-layer classes carry (KimiK3DecoderLayer, Qwen3.5 decoder layer,
...). Streaming layer names are expected to embed the decoder index as
`...layers.<idx>.<attn-suffix>`.
"""

import re
from collections.abc import Callable

import torch
import torch.nn as nn
from vllm.forward_context import get_forward_context
from vllm.logger import logger

from vllm_ascend.distributed.kv_transfer.streaming_kv.manager import (
    StreamingKVManager,
)

_LAYER_IDX_RE = re.compile(r"layers\.(\d+)\.")


def _decoder_index_of_layer_name(layer_name: str) -> int:
    m = _LAYER_IDX_RE.search(layer_name)
    if m is None:
        raise ValueError(f"cannot extract decoder index from streaming layer name: {layer_name}")
    return int(m.group(1))


def build_onload_schedule(
    stream_layers: list[str],
    num_decoder_layers: int,
    prefetch_depth: int,
) -> dict[int, list[str]]:
    """Map issue-point decoder index -> streaming layer names to start there.

    issue(s) = decoder_idx(s) - prefetch_depth, clamped to layer 0. Layers
    whose issue point clamps to 0 are issued together at step start (variant
    B for the head of the network; the copy stream serializes them safely).
    """
    schedule: dict[int, list[str]] = {}
    for name in stream_layers:
        idx = _decoder_index_of_layer_name(name)
        issue = max(0, idx - prefetch_depth)
        schedule.setdefault(issue, []).append(name)
    return schedule


def _current_seq_lens() -> tuple[list[int], list[int]] | None:
    """(prev, cur) per-request sequence lengths from the forward context.

    Best-effort eager path; returns None when unavailable (callers skip
    writeback sizing and fall back to a full-length copy decision upstream).
    """
    try:
        attn_metadata = get_forward_context().attn_metadata
    except Exception:
        return None
    if attn_metadata is None:
        return None
    decode = getattr(attn_metadata, "decode", None)
    seq_lens = getattr(decode, "seq_lens_list", None) or getattr(attn_metadata, "seq_lens_list", None)
    if seq_lens is None:
        return None
    cur = [int(x) for x in seq_lens]
    prev = [max(0, c - 1) for c in cur]
    return prev, cur


def wrap_decoder_layers_for_streaming(
    model: nn.Module,
    manager: StreamingKVManager,
    prefetch_depth: int,
    layer_name_prefix: str = "model.layers.",
) -> int:
    """Wrap decoder layer forwards with onload/writeback hooks.

    Returns the number of wrapped layers (0 means "nothing matched", which
    the caller should treat as a configuration error).
    """
    stream_by_idx = {
        _decoder_index_of_layer_name(n): n for n in manager.stream_layers
    }
    num_layers = (max(stream_by_idx) + 1) if stream_by_idx else 0
    schedule = build_onload_schedule(manager.stream_layers, num_layers, prefetch_depth)

    wrapped = 0
    for module in model.modules():
        layer_idx = getattr(module, "layer_idx", None)
        if layer_idx is None or not hasattr(module, "self_attn"):
            continue
        original_forward = module.forward
        is_streaming = layer_idx in stream_by_idx
        issue_targets = schedule.get(layer_idx, [])
        if not is_streaming and not issue_targets:
            continue

        def make_wrapped(orig: Callable, idx: int, streaming_name: str | None, targets: list[str]):
            def wrapped_forward(*args, **kwargs):
                if streaming_name is not None:
                    manager.wait_onload(streaming_name)
                output = orig(*args, **kwargs)
                hidden = args[0] if args else kwargs.get("hidden_states")
                num_reqs = hidden.shape[0] if isinstance(hidden, torch.Tensor) and hidden.dim() > 1 else 0
                for target in targets:
                    manager.start_onload(target, num_reqs=num_reqs)
                if streaming_name is not None:
                    seq_lens = _current_seq_lens()
                    if seq_lens is not None:
                        manager.start_writeback(streaming_name, *seq_lens)
                return output

            return wrapped_forward

        module.forward = make_wrapped(
            original_forward,
            layer_idx,
            stream_by_idx.get(layer_idx),
            issue_targets,
        )
        wrapped += 1

    logger.info(
        "streaming-kv: wrapped %d decoder layers (%d streaming, %d issue points)",
        wrapped, len(stream_by_idx), len(schedule),
    )
    return wrapped
