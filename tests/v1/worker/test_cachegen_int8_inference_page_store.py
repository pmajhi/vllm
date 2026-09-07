# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.worker.experimental.cachegen_int8_inference_page_store import (
    CacheGenInt8InferencePageStore,
)
from vllm.v1.worker.experimental.cachegen_int8_paged_decode_attention import (
    cachegen_int8_paged_decode_attention_reference,
    dense_decode_attention_reference,
)


NUM_PAGES = 4
TOKENS_PER_PAGE = 4
NUM_KV_HEADS = 2
NUM_QUERY_HEADS = 4
HEAD_SIZE = 8


def make_store() -> CacheGenInt8InferencePageStore:
    return CacheGenInt8InferencePageStore.allocate(
        num_pages=NUM_PAGES,
        tokens_per_page=TOKENS_PER_PAGE,
        num_kv_heads=NUM_KV_HEADS,
        head_size=HEAD_SIZE,
        device=torch.device("cpu"),
    )


def test_store_allocates_compact_typed_layout() -> None:
    store = make_store()

    assert store.key_pages.shape == (
        NUM_PAGES,
        TOKENS_PER_PAGE,
        NUM_KV_HEADS,
        HEAD_SIZE,
    )
    assert store.value_pages.shape == store.key_pages.shape
    assert store.key_pages.dtype == torch.int8
    assert store.value_pages.dtype == torch.int8
    assert store.key_scales.dtype == torch.float16
    assert store.value_scales.dtype == torch.float16
    assert store.key_abs_max.dtype == torch.float16
    assert store.value_abs_max.dtype == torch.float16
    assert not hasattr(store, "key_staging")
    assert not hasattr(store, "value_staging")


def test_compact_store_is_smaller_than_bfloat16_kv_payload() -> None:
    store = make_store()
    bf16_kv_bytes = (
        2
        * NUM_PAGES
        * TOKENS_PER_PAGE
        * NUM_KV_HEADS
        * HEAD_SIZE
        * torch.tensor([], dtype=torch.bfloat16).element_size()
    )

    assert store.persistent_nbytes < bf16_kv_bytes
    assert store.persistent_nbytes / bf16_kv_bytes < 0.60


def test_mapped_token_append_updates_scale_and_preserves_prior_token() -> None:
    store = make_store()

    first_key = torch.full(
        (1, NUM_KV_HEADS, HEAD_SIZE),
        0.25,
        dtype=torch.float32,
    )
    first_value = torch.full_like(first_key, -0.50)
    store.write_mapped_tokens(
        keys=first_key,
        values=first_value,
        page_ids=torch.tensor([1], dtype=torch.int32),
        page_offsets=torch.tensor([0], dtype=torch.int32),
    )
    first_scale = store.key_scales[1].clone()

    second_key = torch.full_like(first_key, 8.0)
    second_value = torch.full_like(first_key, -12.0)
    store.write_mapped_tokens(
        keys=second_key,
        values=second_value,
        page_ids=torch.tensor([1], dtype=torch.int32),
        page_offsets=torch.tensor([1], dtype=torch.int32),
    )

    assert torch.all(store.key_scales[1] > first_scale)
    cache = store.as_paged_kv()
    decoded_first = (
        cache.key_pages[1, 0].float()
        * cache.key_scales[1][:, None]
    )
    decoded_second = (
        cache.key_pages[1, 1].float()
        * cache.key_scales[1][:, None]
    )
    assert torch.allclose(decoded_first, first_key[0], atol=0.08, rtol=0.08)
    assert torch.allclose(decoded_second, second_key[0], atol=0.04, rtol=0.04)


def test_mapped_token_writes_are_page_isolated() -> None:
    store = make_store()

    keys = torch.stack(
        [
            torch.full(
                (NUM_KV_HEADS, HEAD_SIZE),
                1.0,
                dtype=torch.float32,
            ),
            torch.full(
                (NUM_KV_HEADS, HEAD_SIZE),
                -3.0,
                dtype=torch.float32,
            ),
        ]
    )
    values = -keys

    store.write_mapped_tokens(
        keys=keys,
        values=values,
        page_ids=torch.tensor([0, 2], dtype=torch.int32),
        page_offsets=torch.tensor([1, 3], dtype=torch.int32),
    )

    assert store.valid_tokens[0, 1]
    assert store.valid_tokens[2, 3]
    assert not store.valid_tokens[0, 0]
    assert not store.valid_tokens[2, 0]
    assert torch.equal(
        store.key_pages[1],
        torch.zeros_like(store.key_pages[1]),
    )


def test_compact_store_matches_dense_decode_reference() -> None:
    store = make_store()
    seq_len = 7
    generator = torch.Generator().manual_seed(17)

    keys = torch.randn(
        (seq_len, NUM_KV_HEADS, HEAD_SIZE),
        generator=generator,
    )
    values = torch.randn(
        keys.shape,
        generator=generator,
    )
    page_ids = torch.tensor(
        [2, 2, 2, 2, 0, 0, 0],
        dtype=torch.int32,
    )
    page_offsets = torch.tensor(
        [0, 1, 2, 3, 0, 1, 2],
        dtype=torch.int32,
    )
    store.write_mapped_tokens(
        keys=keys,
        values=values,
        page_ids=page_ids,
        page_offsets=page_offsets,
    )

    query = torch.randn(
        (NUM_QUERY_HEADS, HEAD_SIZE),
        generator=generator,
    )
    actual = cachegen_int8_paged_decode_attention_reference(
        query=query,
        cache=store.as_paged_kv(),
        block_table=torch.tensor([2, 0], dtype=torch.int32),
        seq_len=seq_len,
    )
    expected = dense_decode_attention_reference(
        query=query,
        keys=keys,
        values=values,
    )

    assert torch.allclose(actual, expected, atol=0.08, rtol=0.08)


def test_store_rejects_out_of_range_page_id() -> None:
    store = make_store()
    keys = torch.zeros((1, NUM_KV_HEADS, HEAD_SIZE))

    with pytest.raises(ValueError, match="page_ids must lie"):
        store.write_mapped_tokens(
            keys=keys,
            values=keys,
            page_ids=torch.tensor([NUM_PAGES], dtype=torch.int32),
            page_offsets=torch.tensor([0], dtype=torch.int32),
        )


def test_store_rejects_out_of_range_page_offset() -> None:
    store = make_store()
    keys = torch.zeros((1, NUM_KV_HEADS, HEAD_SIZE))

    with pytest.raises(ValueError, match="page_offsets must lie"):
        store.write_mapped_tokens(
            keys=keys,
            values=keys,
            page_ids=torch.tensor([0], dtype=torch.int32),
            page_offsets=torch.tensor([TOKENS_PER_PAGE], dtype=torch.int32),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_compact_store_feeds_fused_qwen_decode_on_cuda() -> None:
    from vllm.v1.worker.experimental.cachegen_int8_paged_decode_attention import (
        cachegen_int8_paged_decode_attention_triton,
    )

    device = torch.device("cuda")
    tokens_per_page = 16
    num_query_heads = 14
    num_kv_heads = 2
    head_size = 64
    seq_len = 23

    generator = torch.Generator(device=device)
    generator.manual_seed(23)

    store = CacheGenInt8InferencePageStore.allocate(
        num_pages=3,
        tokens_per_page=tokens_per_page,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        device=device,
    )
    keys = torch.randn(
        (seq_len, num_kv_heads, head_size),
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    values = torch.randn(
        keys.shape,
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    page_ids = torch.tensor(
        [0] * 16 + [2] * 7,
        device=device,
        dtype=torch.int32,
    )
    page_offsets = torch.tensor(
        list(range(16)) + list(range(7)),
        device=device,
        dtype=torch.int32,
    )
    store.write_mapped_tokens(
        keys=keys,
        values=values,
        page_ids=page_ids,
        page_offsets=page_offsets,
    )

    query = torch.randn(
        (num_query_heads, head_size),
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    fused = cachegen_int8_paged_decode_attention_triton(
        query=query,
        cache=store.as_paged_kv(),
        block_table=torch.tensor(
            [0, 2],
            device=device,
            dtype=torch.int32,
        ),
        seq_len=seq_len,
    )
    expected = dense_decode_attention_reference(
        query=query,
        keys=keys,
        values=values,
    )

    assert torch.allclose(fused, expected, atol=0.08, rtol=0.08)


def test_read_page_dequantizes_full_typed_int8_page() -> None:
    store = CacheGenInt8InferencePageStore.allocate(
        num_pages=1,
        tokens_per_page=4,
        num_kv_heads=2,
        head_size=8,
        device=torch.device("cpu"),
    )
    keys = torch.linspace(-1.0, 1.0, steps=64).reshape(4, 2, 8)
    values = torch.linspace(1.0, -1.0, steps=64).reshape(4, 2, 8)

    store.write_mapped_tokens(
        keys=keys,
        values=values,
        page_ids=torch.tensor([0, 0, 0, 0], dtype=torch.int32),
        page_offsets=torch.tensor([0, 1, 2, 3], dtype=torch.int32),
    )

    decoded_keys, decoded_values = store.read_page(
        page_id=0,
        dtype=torch.float32,
    )

    assert decoded_keys.shape == keys.shape
    assert decoded_values.shape == values.shape
    assert torch.max(torch.abs(decoded_keys - keys)) <= 1.0 / 127.0
    assert torch.max(torch.abs(decoded_values - values)) <= 1.0 / 127.0


def test_read_page_rejects_invalid_page_id_and_dtype() -> None:
    store = CacheGenInt8InferencePageStore.allocate(
        num_pages=1,
        tokens_per_page=4,
        num_kv_heads=2,
        head_size=8,
        device=torch.device("cpu"),
    )

    with pytest.raises(ValueError, match="page_id must be"):
        store.read_page(page_id=1, dtype=torch.float32)

    with pytest.raises(ValueError, match="dtype must be floating point"):
        store.read_page(page_id=0, dtype=torch.int8)
