# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Reference adaptive-INT8 attention over common fixed-byte KV pages."""

from __future__ import annotations

import math

import torch

from vllm.v1.worker.experimental.cachegen_int8_adaptive_page_codec import (
    CacheGenInt8AdaptivePageCodec,
)
from vllm.v1.worker.experimental.cachegen_kv_page_pool import (
    CacheGenKVPageHandle,
    CacheGenKVPagePool,
)


def cachegen_int8_common_page_decode_attention(
    *,
    query: torch.Tensor,
    pool: CacheGenKVPagePool,
    codec: CacheGenInt8AdaptivePageCodec,
    page_handles: list[CacheGenKVPageHandle],
    sequence_length: int,
    scale: float | None = None,
) -> torch.Tensor:
    """Reference decode attention over byte-slab adaptive INT8 K/V pages.

    This function is intentionally for correctness validation only. It
    reconstructs K/V pages through the shared ABI and computes attention with
    PyTorch. The production Triton kernel must consume the same metadata and
    payload bytes directly without materializing decoded pages in HBM.
    """
    if query.ndim != 2:
        raise ValueError(
            f"query must have shape [num_heads, head_size], got "
            f"{tuple(query.shape)}"
        )
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if not page_handles:
        raise ValueError("page_handles must not be empty")
    if query.device != pool.page_bytes.device:
        raise ValueError("query must be on the common page-pool device")
    if not query.dtype.is_floating_point:
        raise ValueError("query must be floating point")

    layout = codec.layout
    if query.shape[1] != layout.head_size:
        raise ValueError(
            f"query head size must be {layout.head_size}, "
            f"got {query.shape[1]}"
        )

    required_pages = (
        sequence_length + layout.tokens_per_page - 1
    ) // layout.tokens_per_page
    if len(page_handles) < required_pages:
        raise ValueError(
            "page_handles do not cover sequence length: "
            f"required={required_pages}, got={len(page_handles)}"
        )

    keys: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    for handle in page_handles[:required_pages]:
        page_keys, page_values = codec.read_page(
            pool=pool,
            handle=handle,
            dtype=torch.float32,
        )
        keys.append(page_keys)
        values.append(page_values)

    all_keys = torch.cat(keys, dim=0)[:sequence_length]
    all_values = torch.cat(values, dim=0)[:sequence_length]

    num_query_heads = query.shape[0]
    num_kv_heads = all_keys.shape[1]
    if num_query_heads % num_kv_heads:
        raise ValueError(
            "num_query_heads must be divisible by num_kv_heads; got "
            f"{num_query_heads} and {num_kv_heads}"
        )

    queries_per_kv_head = num_query_heads // num_kv_heads
    expanded_keys = torch.repeat_interleave(
        all_keys,
        queries_per_kv_head,
        dim=1,
    )
    expanded_values = torch.repeat_interleave(
        all_values,
        queries_per_kv_head,
        dim=1,
    )

    attention_scale = (
        scale if scale is not None else 1.0 / math.sqrt(layout.head_size)
    )
    logits = torch.einsum(
        "hd,thd->ht",
        query.float(),
        expanded_keys,
    ) * attention_scale
    probabilities = torch.softmax(logits, dim=-1)
    return torch.einsum(
        "ht,thd->hd",
        probabilities,
        expanded_values,
    ).to(query.dtype)
