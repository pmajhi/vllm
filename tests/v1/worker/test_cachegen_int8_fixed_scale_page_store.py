# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.worker.experimental.cachegen_int8_fixed_scale_page_store import (
    CacheGenInt8FixedScalePageStore,
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


def make_store() -> CacheGenInt8FixedScalePageStore:
    device = torch.device("cpu")
    scales = torch.full(
        (NUM_KV_HEADS,),
        1.0 / 32.0,
        dtype=torch.float32,
        device=device,
    )
    return CacheGenInt8FixedScalePageStore.allocate(
        num_pages=NUM_PAGES,
        tokens_per_page=TOKENS_PER_PAGE,
        num_kv_heads=NUM_KV_HEADS,
        head_size=HEAD_SIZE,
        key_scales=scales,
        value_scales=scales,
        device=device,
    )


def test_store_is_compact_and_has_no_floating_kv_staging() -> None:
    store = make_store()
    bf16_kv_bytes = (
        2
        * NUM_PAGES
        * TOKENS_PER_PAGE
        * NUM_KV_HEADS
        * HEAD_SIZE
        * torch.tensor([], dtype=torch.bfloat16).element_size()
    )

    assert store.key_pages.dtype == torch.int8
    assert store.value_pages.dtype == torch.int8
    assert store.key_scales.dtype == torch.float16
    assert store.value_scales.dtype == torch.float16
    assert not hasattr(store, "key_staging")
    assert not hasattr(store, "value_staging")
    assert store.persistent_nbytes < bf16_kv_bytes
    assert store.persistent_nbytes / bf16_kv_bytes < 0.55


def test_store_quantizes_mapped_tokens_with_fixed_scales() -> None:
    store = make_store()
    keys = torch.full(
        (2, NUM_KV_HEADS, HEAD_SIZE),
        0.25,
        dtype=torch.float32,
    )
    values = torch.full_like(keys, -0.50)

    store.write_mapped_tokens(
        keys=keys,
        values=values,
        page_ids=torch.tensor([2, 0], dtype=torch.int32),
        page_offsets=torch.tensor([1, 3], dtype=torch.int32),
    )

    cache = store.as_paged_kv()
    decoded_key = (
        cache.key_pages[2, 1].float()
        * cache.key_scales[2][:, None]
    )
    decoded_value = (
        cache.value_pages[0, 3].float()
        * cache.value_scales[0][:, None]
    )

    assert torch.allclose(decoded_key, keys[0], atol=0.02, rtol=0.02)
    assert torch.allclose(decoded_value, values[1], atol=0.02, rtol=0.02)
    assert store.valid_tokens[2, 1]
    assert store.valid_tokens[0, 3]


def test_store_clamps_values_outside_calibrated_range() -> None:
    store = make_store()
    keys = torch.full(
        (1, NUM_KV_HEADS, HEAD_SIZE),
        100.0,
        dtype=torch.float32,
    )

    store.write_mapped_tokens(
        keys=keys,
        values=keys,
        page_ids=torch.tensor([0], dtype=torch.int32),
        page_offsets=torch.tensor([0], dtype=torch.int32),
    )

    assert torch.all(store.key_pages[0, 0] == 127)
    assert torch.all(store.value_pages[0, 0] == 127)


def test_store_feeds_paged_decode_reference() -> None:
    store = make_store()
    seq_len = 7
    generator = torch.Generator().manual_seed(19)

    keys = torch.empty(
        (seq_len, NUM_KV_HEADS, HEAD_SIZE),
    ).uniform_(-0.5, 0.5, generator=generator)
    values = torch.empty_like(keys).uniform_(-0.5, 0.5, generator=generator)
    page_ids = torch.tensor(
        [3, 3, 3, 3, 1, 1, 1],
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

    query = torch.empty(
        (NUM_QUERY_HEADS, HEAD_SIZE),
    ).uniform_(-0.5, 0.5, generator=generator)

    actual = cachegen_int8_paged_decode_attention_reference(
        query=query,
        cache=store.as_paged_kv(),
        block_table=torch.tensor([3, 1], dtype=torch.int32),
        seq_len=seq_len,
    )
    expected = dense_decode_attention_reference(
        query=query,
        keys=keys,
        values=values,
    )

    assert torch.allclose(actual, expected, atol=0.05, rtol=0.05)


def test_store_rejects_invalid_scale_shape() -> None:
    with pytest.raises(ValueError, match="key_scales must have shape"):
        CacheGenInt8FixedScalePageStore.allocate(
            num_pages=NUM_PAGES,
            tokens_per_page=TOKENS_PER_PAGE,
            num_kv_heads=NUM_KV_HEADS,
            head_size=HEAD_SIZE,
            key_scales=torch.ones(NUM_KV_HEADS + 1),
            value_scales=torch.ones(NUM_KV_HEADS),
            device=torch.device("cpu"),
        )


def test_store_rejects_nonpositive_scales() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        CacheGenInt8FixedScalePageStore.allocate(
            num_pages=NUM_PAGES,
            tokens_per_page=TOKENS_PER_PAGE,
            num_kv_heads=NUM_KV_HEADS,
            head_size=HEAD_SIZE,
            key_scales=torch.tensor([0.0, 1.0]),
            value_scales=torch.ones(NUM_KV_HEADS),
            device=torch.device("cpu"),
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
def test_fixed_scale_store_feeds_fused_qwen_decode_on_cuda() -> None:
    from vllm.v1.worker.experimental.cachegen_int8_paged_decode_attention import (
        cachegen_int8_paged_decode_attention_triton,
    )

    device = torch.device("cuda")
    tokens_per_page = 16
    num_pages = 3
    num_query_heads = 14
    num_kv_heads = 2
    head_size = 64
    seq_len = 23

    generator = torch.Generator(device=device)
    generator.manual_seed(29)
    scales = torch.full(
        (num_kv_heads,),
        1.0 / 64.0,
        dtype=torch.float32,
        device=device,
    )
    store = CacheGenInt8FixedScalePageStore.allocate(
        num_pages=num_pages,
        tokens_per_page=tokens_per_page,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        key_scales=scales,
        value_scales=scales,
        device=device,
    )

    keys = torch.empty(
        (seq_len, num_kv_heads, head_size),
        device=device,
        dtype=torch.bfloat16,
    ).uniform_(-0.5, 0.5, generator=generator)
    values = torch.empty_like(keys).uniform_(
        -0.5,
        0.5,
        generator=generator,
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

    query = torch.empty(
        (num_query_heads, head_size),
        device=device,
        dtype=torch.bfloat16,
    ).uniform_(-0.5, 0.5, generator=generator)

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

    assert torch.allclose(fused, expected, atol=0.05, rtol=0.05)


def test_fixed_scale_store_accepts_cpu_scale_tensors() -> None:
    scales = torch.full(
        (NUM_KV_HEADS,),
        1.0 / 32.0,
        dtype=torch.float32,
    )

    store = CacheGenInt8FixedScalePageStore.allocate(
        num_pages=NUM_PAGES,
        tokens_per_page=TOKENS_PER_PAGE,
        num_kv_heads=NUM_KV_HEADS,
        head_size=HEAD_SIZE,
        key_scales=scales,
        value_scales=scales,
        device=torch.device("cpu"),
    )

    assert store.device == torch.device("cpu")
