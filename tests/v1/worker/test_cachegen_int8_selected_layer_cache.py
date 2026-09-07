# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.worker.experimental.cachegen_int8_selected_layer_cache import (
    CacheGenInt8SelectedLayerCache,
)


def make_cache() -> CacheGenInt8SelectedLayerCache:
    device = torch.device("cpu")
    scales = torch.full((2,), 1.0 / 32.0, device=device)
    return CacheGenInt8SelectedLayerCache.allocate(
        max_pages=2,
        tokens_per_page=4,
        num_kv_heads=2,
        head_size=8,
        key_scales=scales,
        value_scales=scales,
        device=device,
    )


def test_maps_native_block_table_to_compact_page_ids() -> None:
    cache = make_cache()

    compact = cache.map_active_block_table(
        torch.tensor([17, 4, 17], dtype=torch.int32)
    )

    assert compact is not None
    assert compact.tolist() == [0, 1, 0]


def test_returns_none_without_mutating_map_on_capacity_exhaustion() -> None:
    cache = make_cache()
    assert cache.map_active_block_table(
        torch.tensor([17], dtype=torch.int32)
    ) is not None

    assert cache.map_active_block_table(
        torch.tensor([17, 4, 9], dtype=torch.int32)
    ) is None
    assert cache.page_map.num_mapped_pages == 1
    assert cache.page_map.compact_page_id(17) == 0
    assert cache.page_map.compact_page_id(4) is None


def test_writes_tokens_using_compact_native_block_mapping() -> None:
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


def test_rejects_nonrank_one_block_table() -> None:
    cache = make_cache()

    with pytest.raises(ValueError, match="must be rank 1"):
        cache.map_active_block_table(
            torch.tensor([[1, 2]], dtype=torch.int32)
        )
