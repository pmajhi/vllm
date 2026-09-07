# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.v1.worker.experimental.cachegen_int8_page_geometry import (
    CACHEGEN_INT8_BINS,
    CACHEGEN_INT8_PAGE_HEADER_BYTES,
    make_cachegen_int8_page_geometry,
)


def test_cachegen_int8_geometry_counts_bins_and_max_metadata() -> None:
    geometry = make_cachegen_int8_page_geometry(
        page_bytes=128 * 1024,
        num_kv_heads=8,
        head_size=128,
    )

    assert geometry.page_bytes == 128 * 1024
    assert geometry.tokens_per_page > 0
    assert geometry.name == "cachegen_int8_quant_only"
    assert geometry.key_payload_bytes == (
        geometry.tokens_per_page * 8 * 128
    )
    assert geometry.value_payload_bytes == (
        geometry.tokens_per_page * 8 * 128
    )
    assert geometry.key_max_bytes == geometry.tokens_per_page * 8 * 2
    assert geometry.value_max_bytes == geometry.tokens_per_page * 8 * 2
    assert geometry.used_bytes <= geometry.page_bytes
    assert geometry.tail_bytes >= 0
    assert geometry.used_bytes >= CACHEGEN_INT8_PAGE_HEADER_BYTES


def test_cachegen_int8_geometry_has_two_kv_payloads_and_two_max_arrays() -> None:
    geometry = make_cachegen_int8_page_geometry(
        page_bytes=4096,
        num_kv_heads=1,
        head_size=4,
    )

    # Per token: K/V payload = 2 * 4 uint8 bytes; K/V max = 2 * FP16.
    assert geometry.bytes_per_token == 12
    assert geometry.tokens_per_page == (4096 - 64) // 12


def test_cachegen_int8_uses_256_unsigned_bins() -> None:
    assert CACHEGEN_INT8_BINS == 256


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"page_bytes": 0}, "page_bytes must be positive"),
        ({"num_kv_heads": 0}, "num_kv_heads must be positive"),
        ({"head_size": 0}, "head_size must be positive"),
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
def test_cachegen_int8_geometry_rejects_invalid_values(
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
        make_cachegen_int8_page_geometry(**values)
