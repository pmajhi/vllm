# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.worker.experimental.uniform_int8_byte_page import (
    PAGE_HEADER_BYTES,
    UniformInt8BytePagePool,
    make_uniform_int8_byte_page_layout,
)


def test_layout_has_nonoverlapping_regions() -> None:
    layout = make_uniform_int8_byte_page_layout(
        page_bytes=4096,
        num_kv_heads=2,
        head_size=8,
    )

    assert layout.key_payload_offset == PAGE_HEADER_BYTES
    assert layout.key_payload_offset < layout.value_payload_offset
    assert layout.value_payload_offset < layout.key_scales_offset
    assert layout.key_scales_offset < layout.value_scales_offset
    assert layout.used_bytes <= layout.page_bytes
    assert layout.tail_bytes >= 0
    assert layout.tokens_per_page == 84


def test_byte_page_round_trip_preserves_int8_reference_accuracy() -> None:
    torch.manual_seed(0)
    pool = UniformInt8BytePagePool(
        num_pages=2,
        page_bytes=4096,
        num_kv_heads=2,
        head_size=8,
        device=torch.device("cpu"),
    )
    key = torch.randn(2, 8, dtype=torch.float32)
    value = torch.randn(2, 8, dtype=torch.float32)

    pool.write(page_id=1, page_offset=3, key=key, value=value)
    decoded_key, decoded_value = pool.read(
        page_id=1,
        page_offset=3,
        dtype=torch.float32,
    )

    assert torch.allclose(decoded_key, key, atol=0.03, rtol=0.03)
    assert torch.allclose(decoded_value, value, atol=0.03, rtol=0.03)
    assert torch.count_nonzero(pool.pages[0]) == 0


def test_byte_page_addressing_separates_offsets_and_pages() -> None:
    pool = UniformInt8BytePagePool(
        num_pages=2,
        page_bytes=4096,
        num_kv_heads=1,
        head_size=4,
        device=torch.device("cpu"),
    )
    first = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    second = torch.tensor([[-4.0, -3.0, -2.0, -1.0]])

    pool.write(page_id=0, page_offset=0, key=first, value=first)
    pool.write(page_id=0, page_offset=1, key=second, value=second)
    pool.write(page_id=1, page_offset=0, key=second, value=second)

    key_00, _ = pool.read(page_id=0, page_offset=0, dtype=torch.float32)
    key_01, _ = pool.read(page_id=0, page_offset=1, dtype=torch.float32)
    key_10, _ = pool.read(page_id=1, page_offset=0, dtype=torch.float32)

    assert torch.allclose(key_00, first, atol=0.03, rtol=0.03)
    assert torch.allclose(key_01, second, atol=0.03, rtol=0.03)
    assert torch.allclose(key_10, second, atol=0.03, rtol=0.03)


@pytest.mark.parametrize(
    ("page_bytes", "num_kv_heads", "head_size", "message"),
    [
        (0, 1, 4, "page_bytes must be positive"),
        (64, 0, 4, "num_kv_heads must be positive"),
        (64, 1, 0, "head_size must be positive"),
        (64, 8, 128, "insufficient usable bytes"),
    ],
)
def test_layout_rejects_invalid_values(
    page_bytes: int,
    num_kv_heads: int,
    head_size: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        make_uniform_int8_byte_page_layout(
            page_bytes=page_bytes,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
        )


def test_pool_rejects_invalid_addresses_and_shapes() -> None:
    pool = UniformInt8BytePagePool(
        num_pages=1,
        page_bytes=4096,
        num_kv_heads=1,
        head_size=4,
        device=torch.device("cpu"),
    )
    valid = torch.zeros(1, 4)

    with pytest.raises(IndexError, match="page_id"):
        pool.write(page_id=1, page_offset=0, key=valid, value=valid)
    with pytest.raises(IndexError, match="page_offset"):
        pool.write(
            page_id=0,
            page_offset=pool.layout.tokens_per_page,
            key=valid,
            value=valid,
        )
    with pytest.raises(ValueError, match="key must have shape"):
        pool.write(
            page_id=0,
            page_offset=0,
            key=torch.zeros(4),
            value=valid,
        )
