# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.v1.worker.experimental.cachegen_int8_block_page_map import (
    CacheGenInt8BlockPageMap,
)


def test_map_assigns_compact_pages_in_first_seen_order() -> None:
    page_map = CacheGenInt8BlockPageMap(max_pages=4)

    assert page_map.map_block_ids([7, 3, 7]) == [0, 1, 0]
    assert page_map.num_mapped_pages == 2
    assert page_map.remaining_pages == 2
    assert page_map.compact_page_id(7) == 0
    assert page_map.compact_page_id(3) == 1


def test_map_reuses_existing_assignments() -> None:
    page_map = CacheGenInt8BlockPageMap(max_pages=3)

    assert page_map.map_block_ids([9, 4]) == [0, 1]
    assert page_map.map_block_ids([4, 9, 4]) == [1, 0, 1]
    assert page_map.num_mapped_pages == 2


def test_map_is_atomic_when_capacity_is_insufficient() -> None:
    page_map = CacheGenInt8BlockPageMap(max_pages=2)

    assert page_map.map_block_ids([5]) == [0]
    assert page_map.map_block_ids([5, 6, 7]) is None
    assert page_map.num_mapped_pages == 1
    assert page_map.compact_page_id(5) == 0
    assert page_map.compact_page_id(6) is None
    assert page_map.compact_page_id(7) is None

    assert page_map.map_block_ids([5, 6]) == [0, 1]


def test_map_rejects_negative_native_block_id() -> None:
    page_map = CacheGenInt8BlockPageMap(max_pages=2)

    with pytest.raises(ValueError, match="must be nonnegative"):
        page_map.map_block_ids([0, -1])


def test_map_clear_releases_all_pages() -> None:
    page_map = CacheGenInt8BlockPageMap(max_pages=2)
    assert page_map.map_block_ids([3, 4]) == [0, 1]

    page_map.clear()

    assert page_map.num_mapped_pages == 0
    assert page_map.remaining_pages == 2
    assert page_map.compact_page_id(3) is None
    assert page_map.map_block_ids([8, 3]) == [0, 1]


def test_map_rejects_nonpositive_capacity() -> None:
    with pytest.raises(ValueError, match="max_pages must be positive"):
        CacheGenInt8BlockPageMap(max_pages=0)
