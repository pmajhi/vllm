# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.v1.worker.experimental.cachegen_kv_page_geometry import (
    CacheGenKVPageGeometry,
)


def test_qwen_style_geometry_derives_4352_byte_common_page() -> None:
    geometry = CacheGenKVPageGeometry(
        tokens_per_page=16,
        num_kv_heads=2,
        head_size=64,
    )

    assert geometry.kv_values_per_page == 2048
    assert geometry.adaptive_int8_payload_nbytes == 4096
    assert geometry.adaptive_int8_metadata_nbytes == 16
    assert geometry.adaptive_int8_minimum_nbytes == 4112
    assert geometry.common_page_nbytes == 4352


def test_geometry_rounds_to_configured_alignment() -> None:
    geometry = CacheGenKVPageGeometry(
        tokens_per_page=4,
        num_kv_heads=2,
        head_size=8,
        alignment_nbytes=64,
    )

    assert geometry.adaptive_int8_minimum_nbytes == 144
    assert geometry.common_page_nbytes == 192


def test_exactly_aligned_geometry_is_not_overallocated() -> None:
    geometry = CacheGenKVPageGeometry(
        tokens_per_page=1,
        num_kv_heads=1,
        head_size=120,
        alignment_nbytes=256,
    )

    assert geometry.adaptive_int8_minimum_nbytes == 248
    assert geometry.common_page_nbytes == 256


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"tokens_per_page": 0}, "tokens_per_page must be positive"),
        ({"num_kv_heads": 0}, "num_kv_heads must be positive"),
        ({"head_size": 0}, "head_size must be positive"),
        ({"alignment_nbytes": 0}, "alignment_nbytes must be positive"),
        (
            {"alignment_nbytes": 96},
            "alignment_nbytes must be a power of two",
        ),
    ],
)
def test_invalid_geometry_is_rejected(
    kwargs: dict[str, int],
    message: str,
) -> None:
    defaults = {
        "tokens_per_page": 16,
        "num_kv_heads": 2,
        "head_size": 64,
    }
    defaults.update(kwargs)

    with pytest.raises(ValueError, match=message):
        CacheGenKVPageGeometry(**defaults)
