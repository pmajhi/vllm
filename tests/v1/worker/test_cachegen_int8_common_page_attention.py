# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import pytest
import torch

from vllm.v1.worker.experimental.cachegen_int8_adaptive_page_codec import (
    CacheGenInt8AdaptivePageCodec,
    CacheGenInt8AdaptivePageLayout,
)
from vllm.v1.worker.experimental.cachegen_int8_common_page_attention import (
    cachegen_int8_common_page_decode_attention,
)
from vllm.v1.worker.experimental.cachegen_kv_page_format_registry import (
    CacheGenKVPageFormatRegistry,
)
from vllm.v1.worker.experimental.cachegen_kv_page_pool import (
    CacheGenKVPageHandle,
    CacheGenKVPagePool,
)
from vllm.v1.worker.experimental.cachegen_quantizer_types import (
    CacheGenKVQuantizer,
)


def make_common_pages() -> tuple[
    CacheGenKVPagePool,
    CacheGenInt8AdaptivePageCodec,
    list[CacheGenKVPageHandle],
]:
    layout = CacheGenInt8AdaptivePageLayout(
        tokens_per_page=4,
        num_kv_heads=2,
        head_size=8,
    )
    registry = CacheGenKVPageFormatRegistry(
        page_nbytes=256,
        tokens_per_page=layout.tokens_per_page,
    )
    pool = CacheGenKVPagePool(
        num_pages=2,
        page_nbytes=256,
        device=torch.device("cpu"),
        page_formats=registry,
    )
    codec = CacheGenInt8AdaptivePageCodec(
        page_format=registry.get(CacheGenKVQuantizer.INT8_ADAPTIVE),
        layout=layout,
    )

    handles = []
    for page_index in range(2):
        handle = pool.allocate(CacheGenKVQuantizer.INT8_ADAPTIVE)
        assert handle is not None

        keys = torch.linspace(
            -1.0 + page_index,
            1.0 + page_index,
            steps=64,
        ).reshape(4, 2, 8)
        values = torch.linspace(
            1.0 - page_index,
            -1.0 - page_index,
            steps=64,
        ).reshape(4, 2, 8)
        codec.write_page(
            pool=pool,
            handle=handle,
            keys=keys,
            values=values,
        )
        handles.append(handle)

    return pool, codec, handles


def reference_attention(
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
) -> torch.Tensor:
    expanded_keys = torch.repeat_interleave(keys, 2, dim=1)
    expanded_values = torch.repeat_interleave(values, 2, dim=1)
    logits = torch.einsum("hd,thd->ht", query, expanded_keys)
    logits *= 1.0 / math.sqrt(query.shape[-1])
    weights = torch.softmax(logits, dim=-1)
    return torch.einsum("ht,thd->hd", weights, expanded_values)


def test_common_page_attention_matches_reference() -> None:
    pool, codec, handles = make_common_pages()
    query = torch.linspace(-0.5, 0.5, steps=32).reshape(4, 8)

    output = cachegen_int8_common_page_decode_attention(
        query=query,
        pool=pool,
        codec=codec,
        page_handles=handles,
        sequence_length=6,
    )

    all_keys = []
    all_values = []
    for handle in handles:
        keys, values = codec.read_page(
            pool=pool,
            handle=handle,
            dtype=torch.float32,
        )
        all_keys.append(keys)
        all_values.append(values)

    expected = reference_attention(
        query,
        torch.cat(all_keys, dim=0)[:6],
        torch.cat(all_values, dim=0)[:6],
    )

    assert torch.allclose(output, expected, atol=1e-6, rtol=1e-6)


def test_common_page_attention_respects_sequence_length() -> None:
    pool, codec, handles = make_common_pages()
    query = torch.ones((4, 8))

    output_four = cachegen_int8_common_page_decode_attention(
        query=query,
        pool=pool,
        codec=codec,
        page_handles=handles,
        sequence_length=4,
    )
    output_eight = cachegen_int8_common_page_decode_attention(
        query=query,
        pool=pool,
        codec=codec,
        page_handles=handles,
        sequence_length=8,
    )

    assert output_four.shape == (4, 8)
    assert output_eight.shape == (4, 8)
    assert not torch.allclose(output_four, output_eight)


def test_common_page_attention_rejects_insufficient_pages() -> None:
    pool, codec, handles = make_common_pages()

    with pytest.raises(ValueError, match="do not cover sequence length"):
        cachegen_int8_common_page_decode_attention(
            query=torch.ones((4, 8)),
            pool=pool,
            codec=codec,
            page_handles=handles[:1],
            sequence_length=8,
        )


def test_common_page_attention_rejects_invalid_query_shape() -> None:
    pool, codec, handles = make_common_pages()

    with pytest.raises(ValueError, match="query must have shape"):
        cachegen_int8_common_page_decode_attention(
            query=torch.ones((1, 4, 8)),
            pool=pool,
            codec=codec,
            page_handles=handles,
            sequence_length=4,
        )


def test_common_page_attention_rejects_invalid_head_size() -> None:
    pool, codec, handles = make_common_pages()

    with pytest.raises(ValueError, match="query head size"):
        cachegen_int8_common_page_decode_attention(
            query=torch.ones((4, 4)),
            pool=pool,
            codec=codec,
            page_handles=handles,
            sequence_length=4,
        )
