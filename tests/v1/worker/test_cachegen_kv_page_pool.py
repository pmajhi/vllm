# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.worker.experimental.cachegen_kv_page_format import (
    CacheGenKVPageFormat,
)
from vllm.v1.worker.experimental.cachegen_kv_page_pool import (
    CacheGenKVPagePool,
)
from vllm.v1.worker.experimental.cachegen_quantizer_types import (
    CacheGenKVQuantizer,
)


def make_format(
    quantizer: CacheGenKVQuantizer,
    *,
    page_nbytes: int = 256,
    implemented: bool = True,
) -> CacheGenKVPageFormat:
    return CacheGenKVPageFormat(
        quantizer=quantizer,
        page_nbytes=page_nbytes,
        tokens_per_page=16,
        metadata_nbytes=32,
        key_payload_nbytes=112,
        value_payload_nbytes=112,
        supports_fused_decode=implemented,
        implemented=implemented,
    )


def make_pool(
    *,
    num_pages: int = 3,
) -> CacheGenKVPagePool:
    return CacheGenKVPagePool(
        num_pages=num_pages,
        page_nbytes=256,
        device=torch.device("cpu"),
        page_formats={
            CacheGenKVQuantizer.INT8_ADAPTIVE: make_format(
                CacheGenKVQuantizer.INT8_ADAPTIVE
            ),
            CacheGenKVQuantizer.KIVI: make_format(
                CacheGenKVQuantizer.KIVI,
                implemented=False,
            ),
            CacheGenKVQuantizer.TURBOQUANT: make_format(
                CacheGenKVQuantizer.TURBOQUANT,
                implemented=False,
            ),
        },
    )


def test_allocates_same_size_pages_for_different_quantizers() -> None:
    pool = make_pool()

    int8_page = pool.allocate(CacheGenKVQuantizer.INT8_ADAPTIVE)
    assert int8_page is not None
    assert int8_page.quantizer is CacheGenKVQuantizer.INT8_ADAPTIVE

    pool.release(int8_page.page_id)

    int8_page = pool.allocate(CacheGenKVQuantizer.INT8_ADAPTIVE)
    assert int8_page is not None
    assert pool.page_bytes_view(int8_page).shape == (256,)
    assert pool.page_bytes_view(int8_page).dtype is torch.uint8
    assert pool.num_allocated_pages == 1
    assert pool.num_free_pages == 2


def test_release_clears_payload_and_metadata() -> None:
    pool = make_pool()
    handle = pool.allocate(CacheGenKVQuantizer.INT8_ADAPTIVE)
    assert handle is not None

    page = pool.page_bytes_view(handle)
    page.fill_(23)

    released = pool.release(handle.page_id)

    assert released == handle
    assert not pool.page_in_use[handle.page_id]
    assert pool.page_quantizer_ids[handle.page_id].item() == -1
    assert not torch.any(pool.page_bytes[handle.page_id])
    assert pool.num_allocated_pages == 0
    assert pool.num_free_pages == 3


def test_pool_returns_none_when_capacity_is_exhausted() -> None:
    pool = make_pool(num_pages=1)

    assert pool.allocate(CacheGenKVQuantizer.INT8_ADAPTIVE) is not None
    assert pool.allocate(CacheGenKVQuantizer.INT8_ADAPTIVE) is None


@pytest.mark.parametrize(
    "quantizer",
    [
        CacheGenKVQuantizer.KIVI,
        CacheGenKVQuantizer.TURBOQUANT,
    ],
)
def test_unimplemented_format_cannot_allocate(
    quantizer: CacheGenKVQuantizer,
) -> None:
    with pytest.raises(ValueError, match="unimplemented"):
        make_pool().allocate(quantizer)


def test_reset_clears_all_pages() -> None:
    pool = make_pool()
    handle = pool.allocate(CacheGenKVQuantizer.INT8_ADAPTIVE)
    assert handle is not None
    pool.page_bytes_view(handle).fill_(7)

    pool.reset()

    assert pool.num_allocated_pages == 0
    assert pool.num_free_pages == 3
    assert not torch.any(pool.page_bytes)
    assert not torch.any(pool.page_in_use)
    assert torch.all(pool.page_quantizer_ids == -1)


def test_rejects_mismatched_page_sizes() -> None:
    mismatched_format = CacheGenKVPageFormat(
        quantizer=CacheGenKVQuantizer.INT8_ADAPTIVE,
        page_nbytes=128,
        tokens_per_page=16,
        metadata_nbytes=16,
        key_payload_nbytes=56,
        value_payload_nbytes=56,
        supports_fused_decode=True,
        implemented=True,
    )

    with pytest.raises(ValueError, match="fixed page size"):
        CacheGenKVPagePool(
            num_pages=2,
            page_nbytes=256,
            device=torch.device("cpu"),
            page_formats={
                CacheGenKVQuantizer.INT8_ADAPTIVE: mismatched_format,
            },
        )


def test_persistent_nbytes_includes_all_gpu_resident_tensors() -> None:
    pool = make_pool(num_pages=3)

    page_bytes_nbytes = 3 * 256
    quantizer_ids_nbytes = 3 * 4
    in_use_nbytes = 3
    valid_tokens_nbytes = 3 * 4

    assert pool.persistent_nbytes == (
        page_bytes_nbytes
        + quantizer_ids_nbytes
        + in_use_nbytes
        + valid_tokens_nbytes
    )


def test_pool_accepts_shared_format_registry() -> None:
    from vllm.v1.worker.experimental.cachegen_kv_page_format_registry import (
        CacheGenKVPageFormatRegistry,
    )

    registry = CacheGenKVPageFormatRegistry(
        page_nbytes=256,
        tokens_per_page=16,
    )
    pool = CacheGenKVPagePool(
        num_pages=2,
        page_nbytes=256,
        device=torch.device("cpu"),
        page_formats=registry,
    )

    handle = pool.allocate(CacheGenKVQuantizer.INT8_ADAPTIVE)

    assert handle is not None
    assert handle.format is registry.get(CacheGenKVQuantizer.INT8_ADAPTIVE)
    assert pool.page_bytes_view(handle).shape == (256,)


def test_valid_token_count_clears_on_release_and_reset() -> None:
    pool = make_pool()
    handle = pool.allocate(CacheGenKVQuantizer.INT8_ADAPTIVE)
    assert handle is not None

    pool.set_valid_tokens(handle=handle, valid_tokens=3)
    assert pool.get_valid_tokens(handle) == 3

    pool.release(handle.page_id)
    assert pool.page_valid_tokens[handle.page_id].item() == 0

    handle = pool.allocate(CacheGenKVQuantizer.INT8_ADAPTIVE)
    assert handle is not None
    pool.set_valid_tokens(handle=handle, valid_tokens=2)
    pool.reset()

    assert not torch.any(pool.page_valid_tokens)
