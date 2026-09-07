# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.kv_cache_interface import FixedByteQuantizedAttentionSpec


def make_spec(**overrides: object) -> FixedByteQuantizedAttentionSpec:
    values: dict[str, object] = {
        "block_size": 16,
        "page_bytes": 64 * 1024,
        "num_kv_heads": 8,
        "head_size": 128,
        "dtype": torch.bfloat16,
        "layout_version": 1,
    }
    values.update(overrides)
    return FixedByteQuantizedAttentionSpec(**values)  # type: ignore[arg-type]


def test_fixed_byte_spec_reports_exact_page_bytes() -> None:
    spec = make_spec()

    assert spec.block_size == 16
    assert spec.page_size_bytes == 64 * 1024
    assert spec.num_kv_heads == 8
    assert spec.head_size == 128


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("block_size", 0, "block_size must be positive"),
        ("page_bytes", 0, "page_bytes must be positive"),
        ("num_kv_heads", 0, "num_kv_heads must be positive"),
        ("head_size", 0, "head_size must be positive"),
        ("layout_version", 0, "layout_version must be positive"),
    ],
)
def test_fixed_byte_spec_rejects_invalid_values(
    field: str,
    value: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        make_spec(**{field: value})


def test_fixed_byte_spec_merges_identical_specs() -> None:
    first = make_spec()
    second = make_spec()

    merged = FixedByteQuantizedAttentionSpec.merge([first, second])

    assert merged == first
    assert merged is not first


def test_fixed_byte_spec_rejects_mismatched_page_geometry() -> None:
    with pytest.raises(ValueError, match="identical page geometry"):
        FixedByteQuantizedAttentionSpec.merge([
            make_spec(page_bytes=64 * 1024),
            make_spec(page_bytes=32 * 1024),
        ])
