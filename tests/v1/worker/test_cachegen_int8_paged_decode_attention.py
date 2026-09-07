# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.worker.experimental.cachegen_int8_paged_decode_attention import (
    CacheGenInt8PagedKV,
    cachegen_int8_paged_decode_attention_reference,
    dense_decode_attention_reference,
    quantize_cachegen_int8_pages,
)


TOKENS_PER_PAGE = 4
NUM_PAGES = 5
NUM_KV_HEADS = 2
NUM_QUERY_HEADS = 4
HEAD_SIZE = 8


def make_dense_kv(
    *,
    seq_len: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    keys = torch.randn(
        (seq_len, NUM_KV_HEADS, HEAD_SIZE),
        generator=generator,
        dtype=torch.float32,
    )
    values = torch.randn(
        keys.shape,
        generator=generator,
        dtype=keys.dtype,
        device=keys.device,
    )
    return keys, values


def make_paged_kv(
    keys: torch.Tensor,
    values: torch.Tensor,
    *,
    page_ids: list[int],
) -> tuple[CacheGenInt8PagedKV, torch.Tensor]:
    num_pages = max(page_ids) + 1
    paged_keys = torch.zeros(
        (num_pages, TOKENS_PER_PAGE, NUM_KV_HEADS, HEAD_SIZE),
        dtype=torch.float32,
    )
    paged_values = torch.zeros_like(paged_keys)

    for token_index in range(keys.shape[0]):
        logical_block = token_index // TOKENS_PER_PAGE
        page_offset = token_index % TOKENS_PER_PAGE
        page_id = page_ids[logical_block]
        paged_keys[page_id, page_offset] = keys[token_index]
        paged_values[page_id, page_offset] = values[token_index]

    cache = quantize_cachegen_int8_pages(paged_keys, paged_values)
    return cache, torch.tensor(page_ids, dtype=torch.int32)


@pytest.mark.parametrize("seq_len", [1, 4, 5, 11])
def test_paged_int8_decode_matches_dense_reference(
    seq_len: int,
) -> None:
    keys, values = make_dense_kv(seq_len=seq_len, seed=11)
    required_blocks = (seq_len + TOKENS_PER_PAGE - 1) // TOKENS_PER_PAGE
    page_ids = list(range(required_blocks))
    cache, block_table = make_paged_kv(keys, values, page_ids=page_ids)

    query = torch.randn(
        (NUM_QUERY_HEADS, HEAD_SIZE),
        generator=torch.Generator().manual_seed(12),
        dtype=torch.float32,
    )

    actual = cachegen_int8_paged_decode_attention_reference(
        query=query,
        cache=cache,
        block_table=block_table,
        seq_len=seq_len,
    )
    expected = dense_decode_attention_reference(
        query=query,
        keys=keys,
        values=values,
    )

    assert actual.shape == expected.shape
    assert torch.allclose(actual, expected, atol=0.035, rtol=0.035)


def test_paged_int8_decode_supports_noncontiguous_page_ids() -> None:
    seq_len = 9
    keys, values = make_dense_kv(seq_len=seq_len, seed=21)
    cache, block_table = make_paged_kv(
        keys,
        values,
        page_ids=[3, 0, 2],
    )
    query = torch.randn(
        (NUM_QUERY_HEADS, HEAD_SIZE),
        generator=torch.Generator().manual_seed(22),
        dtype=torch.float32,
    )

    actual = cachegen_int8_paged_decode_attention_reference(
        query=query,
        cache=cache,
        block_table=block_table,
        seq_len=seq_len,
    )
    expected = dense_decode_attention_reference(query, keys, values)

    assert torch.allclose(actual, expected, atol=0.035, rtol=0.035)


def test_paged_int8_decode_handles_zero_key_value_page() -> None:
    seq_len = 4
    keys = torch.zeros(
        (seq_len, NUM_KV_HEADS, HEAD_SIZE),
        dtype=torch.float32,
    )
    values = torch.zeros_like(keys)
    cache, block_table = make_paged_kv(keys, values, page_ids=[0])
    query = torch.ones(
        (NUM_QUERY_HEADS, HEAD_SIZE),
        dtype=torch.float32,
    )

    actual = cachegen_int8_paged_decode_attention_reference(
        query=query,
        cache=cache,
        block_table=block_table,
        seq_len=seq_len,
    )

    assert torch.equal(actual, torch.zeros_like(actual))
    assert torch.equal(cache.key_scales, torch.zeros_like(cache.key_scales))
    assert torch.equal(
        cache.value_scales,
        torch.zeros_like(cache.value_scales),
    )


def test_paged_int8_decode_uses_grouped_query_head_mapping() -> None:
    seq_len = 4
    keys = torch.zeros(
        (seq_len, NUM_KV_HEADS, HEAD_SIZE),
        dtype=torch.float32,
    )
    values = torch.zeros_like(keys)

    keys[:, 0, 0] = 1.0
    keys[:, 1, 0] = -1.0
    values[:, 0, 1] = 2.0
    values[:, 1, 1] = -3.0

    cache, block_table = make_paged_kv(keys, values, page_ids=[0])
    query = torch.zeros(
        (NUM_QUERY_HEADS, HEAD_SIZE),
        dtype=torch.float32,
    )
    query[:, 0] = 1.0

    actual = cachegen_int8_paged_decode_attention_reference(
        query=query,
        cache=cache,
        block_table=block_table,
        seq_len=seq_len,
    )

    assert torch.allclose(actual[:2, 1], torch.full((2,), 2.0))
    assert torch.allclose(actual[2:, 1], torch.full((2,), -3.0))


def test_cache_rejects_invalid_scale_shape() -> None:
    pages = torch.zeros(
        (2, TOKENS_PER_PAGE, NUM_KV_HEADS, HEAD_SIZE),
        dtype=torch.int8,
    )
    with pytest.raises(ValueError, match="key_scales must have shape"):
        CacheGenInt8PagedKV(
            key_pages=pages,
            value_pages=pages.clone(),
            key_scales=torch.ones((2, NUM_KV_HEADS + 1)),
            value_scales=torch.ones((2, NUM_KV_HEADS)),
        )


def test_decode_rejects_insufficient_block_table() -> None:
    keys, values = make_dense_kv(seq_len=5, seed=31)
    cache, _ = make_paged_kv(keys, values, page_ids=[0, 1])
    query = torch.randn((NUM_QUERY_HEADS, HEAD_SIZE))

    with pytest.raises(ValueError, match="block_table needs at least"):
        cachegen_int8_paged_decode_attention_reference(
            query=query,
            cache=cache,
            block_table=torch.tensor([0], dtype=torch.int32),
            seq_len=5,
        )


def test_decode_rejects_out_of_range_page_id() -> None:
    keys, values = make_dense_kv(seq_len=4, seed=41)
    cache, _ = make_paged_kv(keys, values, page_ids=[0])
    query = torch.randn((NUM_QUERY_HEADS, HEAD_SIZE))

    with pytest.raises(ValueError, match="active block_table page IDs"):
        cachegen_int8_paged_decode_attention_reference(
            query=query,
            cache=cache,
            block_table=torch.tensor([cache.num_pages], dtype=torch.int32),
            seq_len=4,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_paged_int8_decode_matches_dense_reference_on_cuda() -> None:
    device = torch.device("cuda")
    seq_len = 11
    generator = torch.Generator(device=device)
    generator.manual_seed(51)

    keys = torch.randn(
        (seq_len, NUM_KV_HEADS, HEAD_SIZE),
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    values = torch.randn(
        keys.shape,
        generator=generator,
        device=device,
        dtype=keys.dtype,
    )
    required_blocks = (seq_len + TOKENS_PER_PAGE - 1) // TOKENS_PER_PAGE
    page_ids = list(range(required_blocks))

    paged_keys = torch.zeros(
        (required_blocks, TOKENS_PER_PAGE, NUM_KV_HEADS, HEAD_SIZE),
        device=device,
        dtype=torch.float32,
    )
    paged_values = torch.zeros_like(paged_keys)
    for token_index in range(seq_len):
        page_id = token_index // TOKENS_PER_PAGE
        page_offset = token_index % TOKENS_PER_PAGE
        paged_keys[page_id, page_offset] = keys[token_index]
        paged_values[page_id, page_offset] = values[token_index]

    cache = quantize_cachegen_int8_pages(paged_keys, paged_values)
    block_table = torch.tensor(
        page_ids,
        device=device,
        dtype=torch.int32,
    )
    query = torch.randn(
        (NUM_QUERY_HEADS, HEAD_SIZE),
        generator=generator,
        device=device,
        dtype=torch.float32,
    )

    actual = cachegen_int8_paged_decode_attention_reference(
        query=query,
        cache=cache,
        block_table=block_table,
        seq_len=seq_len,
    )
    expected = dense_decode_attention_reference(query, keys, values)

    assert actual.is_cuda
    assert torch.allclose(actual, expected, atol=0.035, rtol=0.035)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("seq_len", [1, 4, 5, 11, 16])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_triton_paged_int8_decode_matches_reference(
    seq_len: int,
    dtype: torch.dtype,
) -> None:
    from vllm.v1.worker.experimental.cachegen_int8_paged_decode_attention import (
        cachegen_int8_paged_decode_attention_triton,
    )

    device = torch.device("cuda")
    generator = torch.Generator(device=device)
    generator.manual_seed(1000 + seq_len)

    keys = torch.randn(
        (seq_len, NUM_KV_HEADS, HEAD_SIZE),
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    values = torch.randn(
        keys.shape,
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    required_blocks = (seq_len + TOKENS_PER_PAGE - 1) // TOKENS_PER_PAGE
    page_ids = list(range(required_blocks))

    paged_keys = torch.zeros(
        (required_blocks, TOKENS_PER_PAGE, NUM_KV_HEADS, HEAD_SIZE),
        device=device,
        dtype=torch.float32,
    )
    paged_values = torch.zeros_like(paged_keys)
    for token_index in range(seq_len):
        page_id = token_index // TOKENS_PER_PAGE
        page_offset = token_index % TOKENS_PER_PAGE
        paged_keys[page_id, page_offset] = keys[token_index]
        paged_values[page_id, page_offset] = values[token_index]

    cache = quantize_cachegen_int8_pages(paged_keys, paged_values)
    block_table = torch.tensor(
        page_ids,
        device=device,
        dtype=torch.int32,
    )
    query = torch.randn(
        (NUM_QUERY_HEADS, HEAD_SIZE),
        generator=generator,
        device=device,
        dtype=dtype,
    )

    actual = cachegen_int8_paged_decode_attention_triton(
        query=query,
        cache=cache,
        block_table=block_table,
        seq_len=seq_len,
    )
    expected = cachegen_int8_paged_decode_attention_reference(
        query=query,
        cache=cache,
        block_table=block_table,
        seq_len=seq_len,
    )

    assert actual.is_cuda
    assert actual.dtype == query.dtype
    assert torch.allclose(actual, expected, atol=0.008, rtol=0.008)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_triton_paged_int8_decode_supports_noncontiguous_page_ids() -> None:
    from vllm.v1.worker.experimental.cachegen_int8_paged_decode_attention import (
        cachegen_int8_paged_decode_attention_triton,
    )

    device = torch.device("cuda")
    seq_len = 11
    generator = torch.Generator(device=device)
    generator.manual_seed(2001)

    keys = torch.randn(
        (seq_len, NUM_KV_HEADS, HEAD_SIZE),
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    values = torch.randn(
        keys.shape,
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    page_ids = [2, 0, 3]
    num_pages = max(page_ids) + 1
    paged_keys = torch.zeros(
        (num_pages, TOKENS_PER_PAGE, NUM_KV_HEADS, HEAD_SIZE),
        device=device,
        dtype=torch.float32,
    )
    paged_values = torch.zeros_like(paged_keys)

    for token_index in range(seq_len):
        logical_block = token_index // TOKENS_PER_PAGE
        page_offset = token_index % TOKENS_PER_PAGE
        page_id = page_ids[logical_block]
        paged_keys[page_id, page_offset] = keys[token_index]
        paged_values[page_id, page_offset] = values[token_index]

    cache = quantize_cachegen_int8_pages(paged_keys, paged_values)
    block_table = torch.tensor(
        page_ids,
        device=device,
        dtype=torch.int32,
    )
    query = torch.randn(
        (NUM_QUERY_HEADS, HEAD_SIZE),
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )

    actual = cachegen_int8_paged_decode_attention_triton(
        query=query,
        cache=cache,
        block_table=block_table,
        seq_len=seq_len,
    )
    expected = cachegen_int8_paged_decode_attention_reference(
        query=query,
        cache=cache,
        block_table=block_table,
        seq_len=seq_len,
    )

    assert torch.allclose(actual, expected, atol=0.008, rtol=0.008)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("seq_len", [1, 16, 17, 127])
def test_triton_paged_int8_decode_qwen25_05b_shape(
    seq_len: int,
) -> None:
    from vllm.v1.worker.experimental.cachegen_int8_paged_decode_attention import (
        cachegen_int8_paged_decode_attention_triton,
    )

    device = torch.device("cuda")
    tokens_per_page = 16
    num_query_heads = 14
    num_kv_heads = 2
    head_size = 64
    required_blocks = (seq_len + tokens_per_page - 1) // tokens_per_page

    generator = torch.Generator(device=device)
    generator.manual_seed(3000 + seq_len)

    keys = torch.randn(
        (seq_len, num_kv_heads, head_size),
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    values = torch.randn(
        keys.shape,
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    paged_keys = torch.zeros(
        (required_blocks, tokens_per_page, num_kv_heads, head_size),
        device=device,
        dtype=torch.float32,
    )
    paged_values = torch.zeros_like(paged_keys)

    for token_index in range(seq_len):
        page_id = token_index // tokens_per_page
        page_offset = token_index % tokens_per_page
        paged_keys[page_id, page_offset] = keys[token_index]
        paged_values[page_id, page_offset] = values[token_index]

    cache = quantize_cachegen_int8_pages(paged_keys, paged_values)
    block_table = torch.arange(
        required_blocks,
        device=device,
        dtype=torch.int32,
    )
    query = torch.randn(
        (num_query_heads, head_size),
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )

    fused = cachegen_int8_paged_decode_attention_triton(
        query=query,
        cache=cache,
        block_table=block_table,
        seq_len=seq_len,
    )
    int8_reference = cachegen_int8_paged_decode_attention_reference(
        query=query,
        cache=cache,
        block_table=block_table,
        seq_len=seq_len,
    )
    native_reference = dense_decode_attention_reference(
        query=query,
        keys=keys,
        values=values,
    )

    assert fused.shape == (num_query_heads, head_size)
    assert fused.dtype == torch.bfloat16
    assert torch.allclose(fused, int8_reference, atol=0.01, rtol=0.01)
    assert torch.allclose(fused, native_reference, atol=0.05, rtol=0.05)


def test_triton_decode_source_uses_fused_paged_reads() -> None:
    from pathlib import Path

    source_path = Path(
        "vllm/v1/worker/experimental/"
        "cachegen_int8_paged_decode_attention.py"
    )
    source = source_path.read_text()

    start = source.index(
        "def cachegen_int8_paged_decode_attention_triton("
    )
    end = len(source)
    triton_source = source[start:end]

    assert "_cachegen_int8_paged_decode_attention_kernel[" in triton_source
    assert "_dequantize_page(" not in triton_source
    assert "dense_decode_attention_reference(" not in triton_source
