# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.worker.experimental.cachegen_int8_dynamic_selected_layer_cache import (
    CacheGenInt8DynamicSelectedLayerCache,
)


def make_cache() -> CacheGenInt8DynamicSelectedLayerCache:
    return CacheGenInt8DynamicSelectedLayerCache.allocate(
        max_pages=2,
        tokens_per_page=4,
        num_kv_heads=2,
        head_size=8,
        device=torch.device("cpu"),
    )


def test_maps_native_blocks_to_compact_pages() -> None:
    cache = make_cache()

    compact_ids = cache.map_active_block_table(
        torch.tensor([17, 4, 17], dtype=torch.int32)
    )

    assert compact_ids is not None
    assert compact_ids.tolist() == [0, 1, 0]


def test_writes_actual_values_to_dynamic_compact_pages() -> None:
    cache = make_cache()
    keys = torch.full((2, 2, 8), 0.25)
    values = torch.full_like(keys, -0.50)

    wrote = cache.write_current_tokens(
        keys=keys,
        values=values,
        native_page_ids=torch.tensor([17, 4], dtype=torch.int32),
        page_offsets=torch.tensor([0, 1], dtype=torch.int32),
    )

    assert wrote
    assert cache.page_store.valid_tokens[0, 0]
    assert cache.page_store.valid_tokens[1, 1]
    assert torch.all(cache.page_store.key_scales[0] > 0)
    assert torch.all(cache.page_store.value_scales[1] > 0)


def test_capacity_exhaustion_returns_false_without_partial_write() -> None:
    cache = make_cache()
    keys = torch.ones((3, 2, 8))

    wrote = cache.write_current_tokens(
        keys=keys,
        values=keys,
        native_page_ids=torch.tensor([1, 2, 3], dtype=torch.int32),
        page_offsets=torch.tensor([0, 1, 2], dtype=torch.int32),
    )

    assert not wrote
    assert cache.page_map.num_mapped_pages == 0
    assert not torch.any(cache.page_store.valid_tokens)


def test_reset_clears_mapping_pages_scales_and_validity() -> None:
    cache = make_cache()
    keys = torch.full((1, 2, 8), 0.25)
    values = torch.full_like(keys, -0.50)

    assert cache.write_current_tokens(
        keys=keys,
        values=values,
        native_page_ids=torch.tensor([17], dtype=torch.int32),
        page_offsets=torch.tensor([0], dtype=torch.int32),
    )
    assert cache.page_map.num_mapped_pages == 1
    assert torch.any(cache.page_store.valid_tokens)

    cache.reset()

    assert cache.page_map.num_mapped_pages == 0
    assert cache.page_map.remaining_pages == 2
    assert not torch.any(cache.page_store.valid_tokens)
    assert not torch.any(cache.page_store.key_pages)
    assert not torch.any(cache.page_store.value_pages)
    assert not torch.any(cache.page_store.key_scales)
    assert not torch.any(cache.page_store.value_scales)


def test_complete_page_is_mirrored_to_common_fixed_byte_pool() -> None:
    cache = make_cache()
    keys = torch.linspace(-1.0, 1.0, steps=64).reshape(4, 2, 8)
    values = torch.linspace(1.0, -1.0, steps=64).reshape(4, 2, 8)

    wrote = cache.write_current_tokens(
        keys=keys,
        values=values,
        native_page_ids=torch.tensor([17, 17, 17, 17], dtype=torch.int32),
        page_offsets=torch.tensor([0, 1, 2, 3], dtype=torch.int32),
    )

    assert wrote
    assert cache.common_page_pool.num_allocated_pages == 1
    handle = cache.common_page_handles[0]
    decoded_keys, decoded_values = cache.common_page_codec.read_page(
        pool=cache.common_page_pool,
        handle=handle,
        dtype=torch.float32,
    )

    typed_keys, typed_values = cache.page_store.read_page(
        page_id=0,
        dtype=torch.float32,
    )
    assert torch.allclose(decoded_keys, typed_keys, atol=1.0 / 127.0)
    assert torch.allclose(decoded_values, typed_values, atol=1.0 / 127.0)


def test_partial_page_is_mirrored_with_valid_token_count() -> None:
    cache = make_cache()
    keys = torch.ones((2, 2, 8))

    wrote = cache.write_current_tokens(
        keys=keys,
        values=keys,
        native_page_ids=torch.tensor([17, 17], dtype=torch.int32),
        page_offsets=torch.tensor([0, 1], dtype=torch.int32),
    )

    assert wrote
    assert cache.common_page_pool.num_allocated_pages == 1
    handle = cache.common_page_handles[0]
    assert cache.common_page_pool.get_valid_tokens(handle) == 2

    decoded_keys, decoded_values = cache.common_page_codec.read_page(
        pool=cache.common_page_pool,
        handle=handle,
        dtype=torch.float32,
    )
    assert torch.allclose(decoded_keys[:2], keys, atol=1.0 / 127.0)
    assert torch.allclose(decoded_values[:2], keys, atol=1.0 / 127.0)
    assert not torch.any(decoded_keys[2:])
    assert not torch.any(decoded_values[2:])


def test_reset_clears_common_fixed_byte_pool_and_handles() -> None:
    cache = make_cache()
    keys = torch.ones((4, 2, 8))

    assert cache.write_current_tokens(
        keys=keys,
        values=keys,
        native_page_ids=torch.tensor([17, 17, 17, 17], dtype=torch.int32),
        page_offsets=torch.tensor([0, 1, 2, 3], dtype=torch.int32),
    )
    assert cache.common_page_pool.num_allocated_pages == 1

    cache.reset()

    assert cache.common_page_pool.num_allocated_pages == 0
    assert cache.common_page_pool.num_free_pages == 2
    assert cache.common_page_handles == {}


def test_get_common_page_ids_follows_compact_block_table_order() -> None:
    cache = make_cache()
    keys = torch.ones((4, 2, 8))

    assert cache.write_current_tokens(
        keys=keys,
        values=keys,
        native_page_ids=torch.tensor([17, 17, 17, 17], dtype=torch.int32),
        page_offsets=torch.tensor([0, 1, 2, 3], dtype=torch.int32),
    )
    assert cache.write_current_tokens(
        keys=keys,
        values=keys,
        native_page_ids=torch.tensor([4, 4, 4, 4], dtype=torch.int32),
        page_offsets=torch.tensor([0, 1, 2, 3], dtype=torch.int32),
    )

    common_ids = cache.get_common_page_ids(
        compact_page_ids=torch.tensor([1, 0, 1], dtype=torch.int32),
    )

    assert common_ids.tolist() == [
        cache.common_page_handles[1].page_id,
        cache.common_page_handles[0].page_id,
        cache.common_page_handles[1].page_id,
    ]


def test_get_common_page_ids_rejects_unmirrored_page() -> None:
    cache = make_cache()

    with pytest.raises(ValueError, match="no common-slab mirror"):
        cache.get_common_page_ids(
            compact_page_ids=torch.tensor([0], dtype=torch.int32),
        )


def test_has_common_page_ids_requires_every_requested_mirror() -> None:
    cache = make_cache()
    assert not cache.has_common_page_ids(
        compact_page_ids=torch.tensor([0], dtype=torch.int32),
    )

    keys = torch.ones((4, 2, 8))
    assert cache.write_current_tokens(
        keys=keys,
        values=keys,
        native_page_ids=torch.tensor([17, 17, 17, 17], dtype=torch.int32),
        page_offsets=torch.tensor([0, 1, 2, 3], dtype=torch.int32),
    )

    assert cache.has_common_page_ids(
        compact_page_ids=torch.tensor([0, 0], dtype=torch.int32),
    )
    assert not cache.has_common_page_ids(
        compact_page_ids=torch.tensor([0, 1], dtype=torch.int32),
    )


def test_write_native_mapped_tokens_handles_multiple_prefill_pages() -> None:
    cache = make_cache()
    keys = torch.arange(4 * 2 * 8, dtype=torch.float32).reshape(4, 2, 8)

    wrote = cache.write_native_mapped_tokens(
        keys=keys,
        values=-keys,
        native_page_ids=torch.tensor([17, 17, 4, 4], dtype=torch.int32),
        page_offsets=torch.tensor([0, 1, 0, 1], dtype=torch.int32),
    )

    assert wrote
    assert cache.page_map.num_mapped_pages == 2
    assert cache.common_page_pool.num_allocated_pages == 2
    assert cache.common_page_pool.get_valid_tokens(
        cache.common_page_handles[0]
    ) == 2
    assert cache.common_page_pool.get_valid_tokens(
        cache.common_page_handles[1]
    ) == 2


def test_write_native_mapped_tokens_fails_cleanly_on_capacity() -> None:
    cache = make_cache()
    keys = torch.ones((3, 2, 8))

    wrote = cache.write_native_mapped_tokens(
        keys=keys,
        values=keys,
        native_page_ids=torch.tensor([1, 2, 3], dtype=torch.int32),
        page_offsets=torch.tensor([0, 0, 0], dtype=torch.int32),
    )

    assert not wrote
    assert cache.page_map.num_mapped_pages == 0
    assert cache.common_page_pool.num_allocated_pages == 0
