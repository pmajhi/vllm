# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.v1.worker.experimental.hetero_fixed_byte_page_pool import (
    HeteroFixedBytePagePool,
)


def test_pool_uses_one_configurable_page_size() -> None:
    pool = HeteroFixedBytePagePool(
        num_pages=4,
        page_bytes=128 * 1024,
    )

    assert pool.num_pages == 4
    assert pool.page_bytes == 128 * 1024
    assert pool.total_bytes == 4 * 128 * 1024
    assert pool.free_bytes == pool.total_bytes
    assert pool.allocated_bytes == 0


def test_allocation_is_deterministic_and_accounted() -> None:
    pool = HeteroFixedBytePagePool(num_pages=4, page_bytes=128 * 1024)

    assert pool.allocate_many(3) == [0, 1, 2]
    assert pool.num_allocated_pages == 3
    assert pool.num_free_pages == 1
    assert pool.allocated_bytes == 3 * 128 * 1024
    assert pool.free_bytes == 128 * 1024


def test_freed_page_is_reused_across_codec_families() -> None:
    pool = HeteroFixedBytePagePool(num_pages=3, page_bytes=128 * 1024)

    cachegen_page_id = pool.allocate()
    kivi_page_id = pool.allocate()
    assert (cachegen_page_id, kivi_page_id) == (0, 1)

    pool.free(cachegen_page_id)

    turboquant_page_id = pool.allocate()
    assert turboquant_page_id == cachegen_page_id
    assert pool.is_allocated(kivi_page_id)
    assert pool.is_allocated(turboquant_page_id)


def test_free_many_reuses_ids_in_ascending_order() -> None:
    pool = HeteroFixedBytePagePool(num_pages=5, page_bytes=128 * 1024)
    assert pool.allocate_many(5) == [0, 1, 2, 3, 4]

    pool.free_many([3, 1])

    assert pool.allocate_many(2) == [1, 3]


def test_allocate_many_is_atomic_when_pool_is_exhausted() -> None:
    pool = HeteroFixedBytePagePool(num_pages=3, page_bytes=128 * 1024)
    assert pool.allocate_many(2) == [0, 1]

    with pytest.raises(MemoryError, match="out of pages"):
        pool.allocate_many(2)

    assert pool.num_allocated_pages == 2
    assert pool.num_free_pages == 1
    assert pool.allocate() == 2


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"num_pages": 0, "page_bytes": 128 * 1024}, "num_pages must be positive"),
        ({"num_pages": 1, "page_bytes": 0}, "page_bytes must be positive"),
    ],
)
def test_pool_rejects_invalid_dimensions(
    kwargs: dict[str, int],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        HeteroFixedBytePagePool(**kwargs)


@pytest.mark.parametrize("page_id", (-1, 3))
def test_pool_rejects_out_of_bounds_page_ids(page_id: int) -> None:
    pool = HeteroFixedBytePagePool(num_pages=3, page_bytes=128 * 1024)

    with pytest.raises(ValueError, match="outside"):
        pool.free(page_id)

    with pytest.raises(ValueError, match="outside"):
        pool.is_allocated(page_id)


def test_pool_rejects_duplicate_and_unallocated_frees() -> None:
    pool = HeteroFixedBytePagePool(num_pages=3, page_bytes=128 * 1024)
    allocated = pool.allocate()

    with pytest.raises(ValueError, match="not currently allocated"):
        pool.free(1)

    pool.free(allocated)

    with pytest.raises(ValueError, match="not currently allocated"):
        pool.free(allocated)

    allocated = pool.allocate_many(2)
    with pytest.raises(ValueError, match="must not contain duplicates"):
        pool.free_many([allocated[0], allocated[0]])
