# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.v1.worker.experimental.kivi_page_geometry import (
    KIVI_PAGE_HEADER_BYTES,
    SUPPORTED_KIVI_BITS,
    kivi_residual_bytes,
    make_kivi_page_geometry,
)


def test_kivi_geometry_uses_complete_key_groups_and_fixed_page_bytes() -> None:
    geometry = make_kivi_page_geometry(
        page_bytes=128 * 1024,
        num_kv_heads=8,
        head_size=128,
        key_bits=2,
        value_bits=2,
        key_group_size=32,
        value_group_size=32,
    )

    assert geometry.page_bytes == 128 * 1024
    assert geometry.quantized_tokens > 0
    assert geometry.quantized_tokens % 32 == 0
    assert geometry.token_capacity == geometry.quantized_tokens
    assert geometry.key_groups == geometry.quantized_tokens // 32
    assert geometry.value_groups_per_token == 4
    assert geometry.name == "kivi_k2_v2"
    assert geometry.used_bytes <= geometry.page_bytes
    assert geometry.tail_bytes >= 0
    assert geometry.used_bytes >= KIVI_PAGE_HEADER_BYTES


@pytest.mark.parametrize(
    ("key_bits", "value_bits"),
    [
        (2, 2),
        (2, 4),
        (2, 8),
        (4, 2),
        (4, 4),
        (4, 8),
        (8, 2),
        (8, 4),
        (8, 8),
    ],
)
def test_kivi_geometry_supports_all_key_value_bit_combinations(
    key_bits: int,
    value_bits: int,
) -> None:
    geometry = make_kivi_page_geometry(
        page_bytes=128 * 1024,
        num_kv_heads=8,
        head_size=128,
        key_bits=key_bits,
        value_bits=value_bits,
        key_group_size=32,
        value_group_size=32,
    )

    assert geometry.key_bits == key_bits
    assert geometry.value_bits == value_bits
    assert geometry.name == f"kivi_k{key_bits}_v{value_bits}"
    assert geometry.quantized_tokens % geometry.key_group_size == 0
    assert geometry.used_bytes <= geometry.page_bytes


def test_kivi_lower_bit_widths_hold_at_least_as_many_quantized_tokens() -> None:
    k2v2 = make_kivi_page_geometry(
        page_bytes=128 * 1024,
        num_kv_heads=8,
        head_size=128,
        key_bits=2,
        value_bits=2,
    )
    k4v4 = make_kivi_page_geometry(
        page_bytes=128 * 1024,
        num_kv_heads=8,
        head_size=128,
        key_bits=4,
        value_bits=4,
    )
    k8v8 = make_kivi_page_geometry(
        page_bytes=128 * 1024,
        num_kv_heads=8,
        head_size=128,
        key_bits=8,
        value_bits=8,
    )

    assert k2v2.quantized_tokens >= k4v4.quantized_tokens
    assert k4v4.quantized_tokens >= k8v8.quantized_tokens


def test_kivi_residual_is_separate_full_precision_kv_storage() -> None:
    assert kivi_residual_bytes(
        num_kv_heads=8,
        head_size=128,
        residual_tokens=32,
    ) == 131_072


def test_kivi_residual_byte_count_respects_dtype_bytes() -> None:
    fp16 = kivi_residual_bytes(
        num_kv_heads=2,
        head_size=8,
        residual_tokens=32,
        dtype_bytes=2,
    )
    fp32 = kivi_residual_bytes(
        num_kv_heads=2,
        head_size=8,
        residual_tokens=32,
        dtype_bytes=4,
    )

    assert fp16 == 2048
    assert fp32 == 4096


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"page_bytes": 0}, "page_bytes must be positive"),
        ({"num_kv_heads": 0}, "num_kv_heads must be positive"),
        ({"head_size": 0}, "head_size must be positive"),
        ({"key_bits": 3}, "key_bits must be one of"),
        ({"value_bits": 3}, "value_bits must be one of"),
        ({"key_group_size": 0}, "key_group_size must be positive"),
        ({"value_group_size": 0}, "value_group_size must be positive"),
        (
            {
                "page_bytes": 1024,
                "num_kv_heads": 8,
                "head_size": 128,
            },
            "insufficient usable bytes",
        ),
    ],
)
def test_kivi_geometry_rejects_invalid_configuration(
    kwargs: dict[str, int],
    message: str,
) -> None:
    values = {
        "page_bytes": 64 * 1024,
        "num_kv_heads": 8,
        "head_size": 128,
    }
    values.update(kwargs)

    with pytest.raises(ValueError, match=message):
        make_kivi_page_geometry(**values)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"num_kv_heads": 0}, "num_kv_heads must be positive"),
        ({"head_size": 0}, "head_size must be positive"),
        ({"residual_tokens": -1}, "residual_tokens must be nonnegative"),
        ({"dtype_bytes": 0}, "dtype_bytes must be positive"),
    ],
)
def test_kivi_residual_rejects_invalid_configuration(
    kwargs: dict[str, int],
    message: str,
) -> None:
    values = {
        "num_kv_heads": 8,
        "head_size": 128,
        "residual_tokens": 32,
    }
    values.update(kwargs)

    with pytest.raises(ValueError, match=message):
        kivi_residual_bytes(**values)


def test_supported_kivi_bits_are_exactly_two_four_eight() -> None:
    assert SUPPORTED_KIVI_BITS == (2, 4, 8)


def test_kivi_k8_v8_requires_at_least_128kib_for_this_head_geometry() -> None:
    with pytest.raises(ValueError, match="insufficient usable bytes"):
        make_kivi_page_geometry(
            page_bytes=64 * 1024,
            num_kv_heads=8,
            head_size=128,
            key_bits=8,
            value_bits=8,
            key_group_size=32,
            value_group_size=32,
        )

    geometry = make_kivi_page_geometry(
        page_bytes=128 * 1024,
        num_kv_heads=8,
        head_size=128,
        key_bits=8,
        value_bits=8,
        key_group_size=32,
        value_group_size=32,
    )

    assert geometry.quantized_tokens >= 32
    assert geometry.quantized_tokens % 32 == 0
    assert geometry.used_bytes <= geometry.page_bytes
