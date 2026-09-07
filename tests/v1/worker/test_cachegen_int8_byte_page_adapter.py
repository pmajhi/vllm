# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.worker.experimental.cachegen_int8_byte_page_adapter import (
    CacheGenInt8FixedBytePageAdapter,
)


PAGE_BYTES = 128 * 1024
NUM_KV_HEADS = 8
HEAD_SIZE = 128


def make_adapter(num_pages: int = 3) -> CacheGenInt8FixedBytePageAdapter:
    return CacheGenInt8FixedBytePageAdapter(
        page_pool=torch.zeros(
            (num_pages, PAGE_BYTES),
            dtype=torch.uint8,
        ),
        num_kv_heads=NUM_KV_HEADS,
        head_size=HEAD_SIZE,
    )


def test_layout_fits_one_fixed_byte_page() -> None:
    adapter = make_adapter()

    assert adapter.layout.used_bytes <= PAGE_BYTES
    assert adapter.layout.slack_bytes == PAGE_BYTES - adapter.layout.used_bytes
    assert adapter.tokens_per_page > 0


def test_round_trip_one_token() -> None:
    torch.manual_seed(0)
    adapter = make_adapter()
    key = torch.randn((NUM_KV_HEADS, HEAD_SIZE), dtype=torch.float32)
    value = torch.randn((NUM_KV_HEADS, HEAD_SIZE), dtype=torch.float32)

    adapter.write_token(
        page_id=0,
        page_offset=0,
        key=key,
        value=value,
    )
    decoded_key, decoded_value = adapter.read_token(
        page_id=0,
        page_offset=0,
    )

    assert torch.allclose(decoded_key, key, atol=0.04, rtol=0.02)
    assert torch.allclose(decoded_value, value, atol=0.04, rtol=0.02)


def test_zero_vectors_round_trip_exactly() -> None:
    adapter = make_adapter()
    zero = torch.zeros((NUM_KV_HEADS, HEAD_SIZE), dtype=torch.float32)

    adapter.write_token(
        page_id=0,
        page_offset=0,
        key=zero,
        value=zero,
    )
    decoded_key, decoded_value = adapter.read_token(
        page_id=0,
        page_offset=0,
    )

    assert torch.equal(decoded_key, zero)
    assert torch.equal(decoded_value, zero)


def test_multiple_pages_and_offsets_are_independent() -> None:
    adapter = make_adapter()
    offset = adapter.tokens_per_page - 1
    first_key = torch.full(
        (NUM_KV_HEADS, HEAD_SIZE),
        0.25,
        dtype=torch.float32,
    )
    first_value = torch.full(
        (NUM_KV_HEADS, HEAD_SIZE),
        -0.5,
        dtype=torch.float32,
    )
    second_key = torch.full(
        (NUM_KV_HEADS, HEAD_SIZE),
        1.5,
        dtype=torch.float32,
    )
    second_value = torch.full(
        (NUM_KV_HEADS, HEAD_SIZE),
        -1.25,
        dtype=torch.float32,
    )

    adapter.write_token(
        page_id=0,
        page_offset=offset,
        key=first_key,
        value=first_value,
    )
    adapter.write_token(
        page_id=1,
        page_offset=0,
        key=second_key,
        value=second_value,
    )

    decoded_first_key, decoded_first_value = adapter.read_token(
        page_id=0,
        page_offset=offset,
    )
    decoded_second_key, decoded_second_value = adapter.read_token(
        page_id=1,
        page_offset=0,
    )

    assert torch.allclose(decoded_first_key, first_key, atol=0.01, rtol=0)
    assert torch.allclose(decoded_first_value, first_value, atol=0.01, rtol=0)
    assert torch.allclose(decoded_second_key, second_key, atol=0.01, rtol=0)
    assert torch.allclose(decoded_second_value, second_value, atol=0.01, rtol=0)


def test_mapping_driven_writes_round_trip() -> None:
    torch.manual_seed(1)
    adapter = make_adapter()
    page_ids = torch.tensor([2, 0, 2], dtype=torch.int32)
    page_offsets = torch.tensor(
        [0, adapter.tokens_per_page - 1, 1],
        dtype=torch.int32,
    )
    keys = torch.randn((3, NUM_KV_HEADS, HEAD_SIZE), dtype=torch.float32)
    values = torch.randn_like(keys)

    for index in range(3):
        adapter.write_token(
            page_id=int(page_ids[index]),
            page_offset=int(page_offsets[index]),
            key=keys[index],
            value=values[index],
        )

    for index in range(3):
        decoded_key, decoded_value = adapter.read_token(
            page_id=int(page_ids[index]),
            page_offset=int(page_offsets[index]),
        )
        assert torch.allclose(
            decoded_key,
            keys[index],
            atol=0.04,
            rtol=0.02,
        )
        assert torch.allclose(
            decoded_value,
            values[index],
            atol=0.04,
            rtol=0.02,
        )


@pytest.mark.parametrize("page_id", (-1, 3))
def test_rejects_invalid_page_id(page_id: int) -> None:
    adapter = make_adapter()
    tensor = torch.zeros((NUM_KV_HEADS, HEAD_SIZE), dtype=torch.float32)

    with pytest.raises(ValueError, match="page_id must be"):
        adapter.write_token(
            page_id=page_id,
            page_offset=0,
            key=tensor,
            value=tensor,
        )


def test_rejects_invalid_page_offset() -> None:
    adapter = make_adapter()
    tensor = torch.zeros((NUM_KV_HEADS, HEAD_SIZE), dtype=torch.float32)

    with pytest.raises(ValueError, match="page_offset must be"):
        adapter.write_token(
            page_id=0,
            page_offset=adapter.tokens_per_page,
            key=tensor,
            value=tensor,
        )


def test_rejects_invalid_token_shape_and_dtype() -> None:
    adapter = make_adapter()
    value = torch.zeros((NUM_KV_HEADS, HEAD_SIZE), dtype=torch.float32)

    with pytest.raises(ValueError, match="key must have shape"):
        adapter.write_token(
            page_id=0,
            page_offset=0,
            key=torch.zeros((NUM_KV_HEADS, HEAD_SIZE - 1)),
            value=value,
        )

    with pytest.raises(ValueError, match="key must have floating-point dtype"):
        adapter.write_token(
            page_id=0,
            page_offset=0,
            key=torch.zeros(
                (NUM_KV_HEADS, HEAD_SIZE),
                dtype=torch.int8,
            ),
            value=value,
        )


def test_rejects_non_uint8_pool() -> None:
    with pytest.raises(ValueError, match="dtype torch.uint8"):
        CacheGenInt8FixedBytePageAdapter(
            page_pool=torch.zeros((1, PAGE_BYTES), dtype=torch.int8),
            num_kv_heads=NUM_KV_HEADS,
            head_size=HEAD_SIZE,
        )


def test_mapped_token_writer_round_trips_block_table_style_mapping() -> None:
    from vllm.v1.worker.experimental.cachegen_int8_byte_page_adapter import (
        write_cachegen_int8_mapped_tokens,
    )

    torch.manual_seed(2)
    adapter = make_adapter()
    keys = torch.randn((4, NUM_KV_HEADS, HEAD_SIZE), dtype=torch.float32)
    values = torch.randn_like(keys)

    page_ids = torch.tensor([1, 1, 0, 2], dtype=torch.int32)
    page_offsets = torch.tensor(
        [0, adapter.tokens_per_page - 1, 3, 1],
        dtype=torch.int32,
    )

    write_cachegen_int8_mapped_tokens(
        adapter=adapter,
        keys=keys,
        values=values,
        quantized_page_ids=page_ids,
        quantized_page_offsets=page_offsets,
    )

    for index in range(keys.shape[0]):
        decoded_key, decoded_value = adapter.read_token(
            page_id=int(page_ids[index]),
            page_offset=int(page_offsets[index]),
        )
        assert torch.allclose(
            decoded_key,
            keys[index],
            atol=0.04,
            rtol=0.02,
        )
        assert torch.allclose(
            decoded_value,
            values[index],
            atol=0.04,
            rtol=0.02,
        )


def test_mapped_token_writer_rejects_invalid_mapping_shapes() -> None:
    from vllm.v1.worker.experimental.cachegen_int8_byte_page_adapter import (
        write_cachegen_int8_mapped_tokens,
    )

    adapter = make_adapter()
    keys = torch.zeros((2, NUM_KV_HEADS, HEAD_SIZE), dtype=torch.float32)
    values = torch.zeros_like(keys)

    with pytest.raises(ValueError, match="quantized_page_ids must have shape"):
        write_cachegen_int8_mapped_tokens(
            adapter=adapter,
            keys=keys,
            values=values,
            quantized_page_ids=torch.tensor([0], dtype=torch.int32),
            quantized_page_offsets=torch.tensor([0, 1], dtype=torch.int32),
        )

    with pytest.raises(
        ValueError,
        match="quantized_page_offsets must have integer dtype",
    ):
        write_cachegen_int8_mapped_tokens(
            adapter=adapter,
            keys=keys,
            values=values,
            quantized_page_ids=torch.tensor([0, 1], dtype=torch.int32),
            quantized_page_offsets=torch.tensor([0.0, 1.0]),
        )


def test_mapped_token_writer_rejects_incompatible_kv_shape() -> None:
    from vllm.v1.worker.experimental.cachegen_int8_byte_page_adapter import (
        write_cachegen_int8_mapped_tokens,
    )

    adapter = make_adapter()
    keys = torch.zeros((2, NUM_KV_HEADS, HEAD_SIZE - 1), dtype=torch.float32)
    values = torch.zeros_like(keys)
    page_ids = torch.tensor([0, 1], dtype=torch.int32)
    page_offsets = torch.tensor([0, 1], dtype=torch.int32)

    with pytest.raises(ValueError, match="keys must have shape"):
        write_cachegen_int8_mapped_tokens(
            adapter=adapter,
            keys=keys,
            values=values,
            quantized_page_ids=page_ids,
            quantized_page_offsets=page_offsets,
        )
