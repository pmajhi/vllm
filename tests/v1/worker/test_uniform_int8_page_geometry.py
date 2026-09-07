# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.worker.experimental.uniform_int8_codec import UniformInt8KVCodec


def test_uniform_int8_page_geometry_counts_payload_and_scales() -> None:
    geometry = UniformInt8KVCodec().page_geometry(
        physical_page_bytes=64 * 1024,
        num_kv_heads=8,
        head_size=128,
        dtype=torch.bfloat16,
    )

    assert geometry.physical_page_bytes == 64 * 1024
    assert geometry.page_header_bytes == 0
    assert geometry.tokens_per_page == 31

    # K and V each store [num_kv_heads, head_size] as int8.
    assert geometry.payload_bytes == 31 * (2 * 8 * 128)

    # K and V each carry one float32 scale for every KV head.
    assert geometry.metadata_bytes == 31 * (2 * 8 * 4)

    assert (
        geometry.payload_bytes
        + geometry.metadata_bytes
        + geometry.internal_padding_bytes
        == geometry.physical_page_bytes
    )
    assert geometry.internal_padding_bytes == 64


@pytest.mark.parametrize(
    ("physical_page_bytes", "num_kv_heads", "head_size", "expected_capacity"),
    [
        (4096, 1, 4, 256),
        (4096, 2, 8, 85),
        (16 * 1024, 8, 128, 7),
        (64 * 1024, 8, 128, 31),
        (64 * 1024, 32, 128, 7),
    ],
)
def test_uniform_int8_page_geometry_capacity(
    physical_page_bytes: int,
    num_kv_heads: int,
    head_size: int,
    expected_capacity: int,
) -> None:
    geometry = UniformInt8KVCodec().page_geometry(
        physical_page_bytes=physical_page_bytes,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=torch.float16,
    )

    assert geometry.tokens_per_page == expected_capacity
    assert geometry.payload_bytes > 0
    assert geometry.metadata_bytes > 0
    assert geometry.internal_padding_bytes >= 0


@pytest.mark.parametrize(
    ("physical_page_bytes", "num_kv_heads", "head_size", "message"),
    [
        (0, 8, 128, "physical_page_bytes must be positive"),
        (64 * 1024, 0, 128, "num_kv_heads must be positive"),
        (64 * 1024, 8, 0, "head_size must be positive"),
        (1, 8, 128, "insufficient bytes"),
    ],
)
def test_uniform_int8_page_geometry_rejects_invalid_values(
    physical_page_bytes: int,
    num_kv_heads: int,
    head_size: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        UniformInt8KVCodec().page_geometry(
            physical_page_bytes=physical_page_bytes,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            dtype=torch.float16,
        )
