# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.v1.worker.experimental.turboquant_page_geometry import (
    SUPPORTED_TURBOQUANT_KEY_BITS,
    SUPPORTED_TURBOQUANT_VALUE_BITS,
    TURBOQUANT_PAGE_HEADER_BYTES,
    make_turboquant_page_geometry,
    turboquant_ring_buffer_bytes,
)


def test_turboquant_geometry_accounts_for_all_compressed_categories() -> None:
    geometry = make_turboquant_page_geometry(
        page_bytes=128 * 1024,
        num_kv_heads=8,
        head_size=128,
        key_bits=3,
        value_bits=2,
        value_group_size=32,
    )

    assert geometry.page_bytes == 128 * 1024
    assert geometry.tokens_per_page > 0
    assert geometry.value_groups_per_token == 4
    assert geometry.key_mse_index_bytes > 0
    assert geometry.key_qjl_sign_bytes > 0
    assert geometry.key_metadata_bytes > 0
    assert geometry.value_payload_bytes > 0
    assert geometry.value_metadata_bytes > 0
    assert geometry.used_bytes <= geometry.page_bytes
    assert geometry.tail_bytes >= 0
    assert geometry.used_bytes >= TURBOQUANT_PAGE_HEADER_BYTES
    assert geometry.name == "turboquant_k3_v2"


@pytest.mark.parametrize("key_bits", [2, 3, 4])
@pytest.mark.parametrize("value_bits", [2, 4])
def test_turboquant_geometry_supports_all_bit_combinations(
    key_bits: int,
    value_bits: int,
) -> None:
    geometry = make_turboquant_page_geometry(
        page_bytes=128 * 1024,
        num_kv_heads=8,
        head_size=128,
        key_bits=key_bits,
        value_bits=value_bits,
    )

    assert geometry.key_bits == key_bits
    assert geometry.value_bits == value_bits
    assert geometry.name == f"turboquant_k{key_bits}_v{value_bits}"
    assert geometry.used_bytes <= geometry.page_bytes


def test_turboquant_lower_bits_hold_at_least_as_many_tokens() -> None:
    low_bits = make_turboquant_page_geometry(
        page_bytes=128 * 1024,
        num_kv_heads=8,
        head_size=128,
        key_bits=2,
        value_bits=2,
    )
    high_bits = make_turboquant_page_geometry(
        page_bytes=128 * 1024,
        num_kv_heads=8,
        head_size=128,
        key_bits=4,
        value_bits=4,
    )

    assert low_bits.tokens_per_page >= high_bits.tokens_per_page


def test_turboquant_qjl_projection_count_changes_page_capacity() -> None:
    fewer_projections = make_turboquant_page_geometry(
        page_bytes=128 * 1024,
        num_kv_heads=8,
        head_size=128,
        key_bits=3,
        value_bits=2,
        qjl_projections=64,
    )
    more_projections = make_turboquant_page_geometry(
        page_bytes=128 * 1024,
        num_kv_heads=8,
        head_size=128,
        key_bits=3,
        value_bits=2,
        qjl_projections=256,
    )

    assert fewer_projections.tokens_per_page >= more_projections.tokens_per_page
    assert fewer_projections.key_qjl_sign_bytes < more_projections.key_qjl_sign_bytes


def test_turboquant_ring_buffer_is_separate_full_precision_kv_storage() -> None:
    assert turboquant_ring_buffer_bytes(
        num_kv_heads=8,
        head_size=128,
        ring_capacity=128,
    ) == 524_288


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"page_bytes": 0}, "page_bytes must be positive"),
        ({"num_kv_heads": 0}, "num_kv_heads must be positive"),
        ({"head_size": 0}, "head_size must be positive"),
        ({"key_bits": 8}, "key_bits must be one of"),
        ({"value_bits": 3}, "value_bits must be one of"),
        ({"value_group_size": 0}, "value_group_size must be positive"),
        ({"qjl_projections": 0}, "qjl_projections must be positive"),
        (
            {
                "page_bytes": 64,
                "num_kv_heads": 8,
                "head_size": 128,
            },
            "insufficient usable bytes",
        ),
    ],
)
def test_turboquant_geometry_rejects_invalid_configuration(
    kwargs: dict[str, int],
    message: str,
) -> None:
    values = {
        "page_bytes": 128 * 1024,
        "num_kv_heads": 8,
        "head_size": 128,
    }
    values.update(kwargs)

    with pytest.raises(ValueError, match=message):
        make_turboquant_page_geometry(**values)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"num_kv_heads": 0}, "num_kv_heads must be positive"),
        ({"head_size": 0}, "head_size must be positive"),
        ({"ring_capacity": -1}, "ring_capacity must be nonnegative"),
        ({"dtype_bytes": 0}, "dtype_bytes must be positive"),
    ],
)
def test_turboquant_ring_buffer_rejects_invalid_configuration(
    kwargs: dict[str, int],
    message: str,
) -> None:
    values = {
        "num_kv_heads": 8,
        "head_size": 128,
        "ring_capacity": 128,
    }
    values.update(kwargs)

    with pytest.raises(ValueError, match=message):
        turboquant_ring_buffer_bytes(**values)


def test_supported_turboquant_bits_are_explicit() -> None:
    assert SUPPORTED_TURBOQUANT_KEY_BITS == (2, 3, 4)
    assert SUPPORTED_TURBOQUANT_VALUE_BITS == (2, 4)
