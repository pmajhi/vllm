# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.v1.worker.experimental.cachegen_kv_page_format_registry import (
    DEFAULT_CACHEGEN_KV_PAGE_NBYTES,
    DEFAULT_CACHEGEN_KV_TOKENS_PER_PAGE,
    CacheGenKVPageFormatRegistry,
)
from vllm.v1.worker.experimental.cachegen_quantizer_types import (
    CacheGenKVQuantizer,
)


def test_all_registered_formats_share_one_fixed_page_size() -> None:
    registry = CacheGenKVPageFormatRegistry()

    assert registry.page_nbytes == DEFAULT_CACHEGEN_KV_PAGE_NBYTES
    assert registry.tokens_per_page == DEFAULT_CACHEGEN_KV_TOKENS_PER_PAGE
    assert {
        page_format.page_nbytes
        for page_format in registry.formats().values()
    } == {DEFAULT_CACHEGEN_KV_PAGE_NBYTES}


def test_only_adaptive_int8_has_an_implemented_cachegen_page_format() -> None:
    registry = CacheGenKVPageFormatRegistry()

    assert registry.implemented_formats() == {
        CacheGenKVQuantizer.INT8_ADAPTIVE: registry.get(
            CacheGenKVQuantizer.INT8_ADAPTIVE
        ),
    }
    assert registry.get(
        CacheGenKVQuantizer.INT8_ADAPTIVE
    ).supports_fused_decode


@pytest.mark.parametrize(
    "quantizer",
    [
        CacheGenKVQuantizer.NATIVE,
        CacheGenKVQuantizer.FP8,
        CacheGenKVQuantizer.KIVI,
        CacheGenKVQuantizer.TURBOQUANT,
        CacheGenKVQuantizer.INT4_GROUPWISE,
    ],
)
def test_non_cachegen_page_descriptors_remain_unimplemented(
    quantizer: CacheGenKVQuantizer,
) -> None:
    page_format = CacheGenKVPageFormatRegistry().get(quantizer)

    assert not page_format.implemented
    assert not page_format.supports_fused_decode
    assert page_format.metadata_nbytes == 0
    assert page_format.key_payload_nbytes == 0
    assert page_format.value_payload_nbytes == 0


def test_int8_page_offsets_partition_the_page() -> None:
    page_format = CacheGenKVPageFormatRegistry().get(
        CacheGenKVQuantizer.INT8_ADAPTIVE
    )

    assert page_format.metadata_offset == 0
    assert page_format.key_payload_offset == page_format.metadata_nbytes
    assert page_format.value_payload_offset == (
        page_format.metadata_nbytes + page_format.key_payload_nbytes
    )
    assert page_format.unused_nbytes == 0


@pytest.mark.parametrize("page_nbytes", [128, 256, 4096, 8192])
def test_custom_page_geometry_is_applied_to_all_formats(
    page_nbytes: int,
) -> None:
    registry = CacheGenKVPageFormatRegistry(
        page_nbytes=page_nbytes,
        tokens_per_page=32,
    )

    assert {
        page_format.page_nbytes
        for page_format in registry.formats().values()
    } == {page_nbytes}
    assert {
        page_format.tokens_per_page
        for page_format in registry.formats().values()
    } == {32}

    int8_format = registry.get(CacheGenKVQuantizer.INT8_ADAPTIVE)
    assert int8_format.metadata_nbytes % 4 == 0
    assert int8_format.key_payload_nbytes > 0
    assert int8_format.value_payload_nbytes > 0
    assert int8_format.unused_nbytes == 0


@pytest.mark.parametrize(
    ("page_nbytes", "tokens_per_page", "message"),
    [
        (0, 16, "page_nbytes must be positive"),
        (4096, 0, "tokens_per_page must be positive"),
    ],
)
def test_invalid_geometry_is_rejected(
    page_nbytes: int,
    tokens_per_page: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        CacheGenKVPageFormatRegistry(
            page_nbytes=page_nbytes,
            tokens_per_page=tokens_per_page,
        )


def test_registry_from_geometry_fits_adaptive_int8_payload() -> None:
    from vllm.v1.worker.experimental.cachegen_kv_page_geometry import (
        CacheGenKVPageGeometry,
    )

    geometry = CacheGenKVPageGeometry(
        tokens_per_page=16,
        num_kv_heads=2,
        head_size=64,
    )
    registry = CacheGenKVPageFormatRegistry.from_geometry(geometry)
    page_format = registry.get(CacheGenKVQuantizer.INT8_ADAPTIVE)

    assert page_format.page_nbytes == 4352
    assert page_format.tokens_per_page == 16
    assert page_format.metadata_nbytes >= (
        geometry.adaptive_int8_metadata_nbytes
    )
    assert page_format.key_payload_nbytes >= geometry.kv_values_per_page
    assert page_format.value_payload_nbytes >= geometry.kv_values_per_page


def test_registry_from_geometry_fits_adaptive_int8_payload() -> None:
    from vllm.v1.worker.experimental.cachegen_kv_page_geometry import (
        CacheGenKVPageGeometry,
    )

    geometry = CacheGenKVPageGeometry(
        tokens_per_page=16,
        num_kv_heads=2,
        head_size=64,
    )
    registry = CacheGenKVPageFormatRegistry.from_geometry(geometry)
    page_format = registry.get(CacheGenKVQuantizer.INT8_ADAPTIVE)

    assert page_format.page_nbytes == 4352
    assert page_format.tokens_per_page == 16
    assert page_format.metadata_nbytes >= (
        geometry.adaptive_int8_metadata_nbytes
    )
    assert page_format.key_payload_nbytes >= geometry.kv_values_per_page
    assert page_format.value_payload_nbytes >= geometry.kv_values_per_page
