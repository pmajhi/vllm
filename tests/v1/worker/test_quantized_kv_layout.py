# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest
import torch

from vllm.v1.quantized_kv_layout import (
    get_quantized_kv_page_layout,
    tokens_per_page_for_quantizer,
    fixed_byte_page_geometry,
)
from vllm.v1.worker.block_table import BlockTable, MultiGroupBlockTable
from vllm.v1.worker.gpu_input_batch import InputBatch
from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu_input_batch import CachedRequestState

@pytest.mark.parametrize(
    ("quantizer_id", "physical_page_bytes", "expected_tokens_per_page"),
    [
        (0, 512, 16),
        (1, 512, 16),
        (0, 1024, 32),
        (1, 1024, 48),
    ],
)

def test_tokens_per_page_for_supported_quantizer(
    quantizer_id: int,
    physical_page_bytes: int,
    expected_tokens_per_page: int,
) -> None:
    assert tokens_per_page_for_quantizer(
        quantizer_id,
        physical_page_bytes,
    ) == expected_tokens_per_page


def test_get_quantized_kv_page_layout_returns_registered_id() -> None:
    layout = get_quantized_kv_page_layout(1)

    assert layout.quantizer_id == 1
    assert layout.bytes_per_token == 16
    assert layout.quantizer_metadata_bytes == 256


@pytest.mark.parametrize("quantizer_id", [-1, 2, 99])
def test_unknown_quantizer_id_is_rejected(quantizer_id: int) -> None:
    with pytest.raises(ValueError, match="Unsupported quantizer_id"):
        get_quantized_kv_page_layout(quantizer_id)


@pytest.mark.parametrize("physical_page_bytes", [0, -1])
def test_nonpositive_physical_page_bytes_are_rejected(
    physical_page_bytes: int,
) -> None:
    with pytest.raises(ValueError, match="physical_page_bytes"):
        tokens_per_page_for_quantizer(
            0,
            physical_page_bytes,
        )


def _make_block_table() -> BlockTable:
    return BlockTable(
        block_size=16,
        max_num_reqs=2,
        max_num_blocks_per_req=4,
        max_num_batched_tokens=8,
        pin_memory=False,
        device=torch.device("cpu"),
    )


def test_quantized_mapping_uses_per_request_page_capacity() -> None:
    block_table = _make_block_table()

    # Request 0 logical pages map to physical pages 10, 11, 12, 13.
    # Request 1 logical pages map to physical pages 20, 21, 22, 23.
    block_table.add_row([10, 11, 12, 13], row_idx=0)
    block_table.add_row([20, 21, 22, 23], row_idx=1)

    request_indices = np.array([0, 0, 1, 1], dtype=np.int32)
    positions = np.array([15, 16, 31, 32], dtype=np.int32)
    tokens_per_page_by_request = np.array([16, 32], dtype=np.int32)

    page_ids, page_offsets = block_table.compute_quantized_mapping(
        request_indices,
        positions,
        tokens_per_page_by_request,
    )

    np.testing.assert_array_equal(
        page_ids,
        np.array([10, 11, 20, 21], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        page_offsets,
        np.array([15, 0, 31, 0], dtype=np.int32),
    )


def test_quantized_mapping_rejects_invalid_page_capacity() -> None:
    block_table = _make_block_table()

    with pytest.raises(ValueError, match="positive tokens_per_page"):
        block_table.compute_quantized_mapping(
            np.array([0], dtype=np.int32),
            np.array([0], dtype=np.int32),
            np.array([0, 16], dtype=np.int32),
        )


def test_multi_group_quantized_mapping_returns_one_result_per_group() -> None:
    block_tables = MultiGroupBlockTable(
        max_num_reqs=2,
        max_model_len=64,
        max_num_batched_tokens=4,
        pin_memory=False,
        device=torch.device("cpu"),
        block_sizes=[16, 16],
    )

    block_tables.add_row(([10, 11, 12, 13], [30, 31, 32, 33]), row_idx=0)
    block_tables.add_row(([20, 21, 22, 23], [40, 41, 42, 43]), row_idx=1)

    request_indices = np.array([0, 1], dtype=np.int32)
    positions = np.array([16, 32], dtype=np.int32)
    tokens_per_page_by_request = np.array([16, 32], dtype=np.int32)

    page_ids_by_group, offsets_by_group = (
        block_tables.compute_quantized_mapping(
            request_indices,
            positions,
            tokens_per_page_by_request,
        )
    )

    assert len(page_ids_by_group) == 2
    assert len(offsets_by_group) == 2

    np.testing.assert_array_equal(
        page_ids_by_group[0],
        np.array([11, 21], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        page_ids_by_group[1],
        np.array([31, 41], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        offsets_by_group[0],
        np.array([0, 0], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        offsets_by_group[1],
        np.array([0, 0], dtype=np.int32),
    )


def test_multi_group_quantized_mapping_handles_no_cache_groups() -> None:
    block_tables = MultiGroupBlockTable(
        max_num_reqs=2,
        max_model_len=64,
        max_num_batched_tokens=4,
        pin_memory=False,
        device=torch.device("cpu"),
        block_sizes=[],
    )

    page_ids_by_group, offsets_by_group = (
        block_tables.compute_quantized_mapping(
            np.array([0], dtype=np.int32),
            np.array([0], dtype=np.int32),
            np.array([16, 16], dtype=np.int32),
        )
    )

    assert page_ids_by_group == []
    assert offsets_by_group == []


def test_experimental_quantized_page_metadata_commit() -> None:
    input_batch = InputBatch(
        max_num_reqs=2,
        max_model_len=64,
        max_num_batched_tokens=8,
        block_sizes=[16],
        is_pooling_model=False,
        pin_memory=False,
        device=torch.device("cpu"),
        vocab_size=32000,
    )

    input_batch.quantizer_id_cpu[:2] = [0, 1]
    input_batch.tokens_per_page_cpu[:2] = [16, 32]

    input_batch.commit_quantized_page_metadata(num_reqs=2)

    assert input_batch.quantizer_id_gpu[:2].tolist() == [0, 1]
    assert input_batch.tokens_per_page_gpu[:2].tolist() == [16, 32]

def test_fixed_byte_pages_can_have_different_token_capacities() -> None:
    fp8_geometry = fixed_byte_page_geometry(
        physical_page_bytes=4096,
        page_header_bytes=64,
        quantizer_metadata_bytes=0,
        bytes_per_token=32,
    )
    kivi_value_geometry = fixed_byte_page_geometry(
        physical_page_bytes=4096,
        page_header_bytes=64,
        quantizer_metadata_bytes=256,
        bytes_per_token=16,
    )

    assert fp8_geometry.physical_page_bytes == 4096
    assert fp8_geometry.tokens_per_page == 126
    assert fp8_geometry.internal_padding_bytes == 0

    assert kivi_value_geometry.physical_page_bytes == 4096
    assert kivi_value_geometry.tokens_per_page == 236
    assert kivi_value_geometry.internal_padding_bytes == 0

    assert (
        fp8_geometry.tokens_per_page
        != kivi_value_geometry.tokens_per_page
    )

@pytest.mark.parametrize(
    (
        "physical_page_bytes",
        "page_header_bytes",
        "quantizer_metadata_bytes",
        "bytes_per_token",
        "error_message",
    ),
    [
        (0, 0, 0, 1, "physical_page_bytes"),
        (4096, -1, 0, 1, "page_header_bytes"),
        (4096, 0, -1, 1, "quantizer_metadata_bytes"),
        (4096, 0, 0, 0, "bytes_per_token"),
        (64, 64, 1, 1, "cover header and quantizer metadata"),
        (64, 64, 0, 1, "insufficient usable bytes"),
    ],
)
def test_fixed_byte_page_geometry_rejects_invalid_inputs(
    physical_page_bytes: int,
    page_header_bytes: int,
    quantizer_metadata_bytes: int,
    bytes_per_token: int,
    error_message: str,
) -> None:
    with pytest.raises(ValueError, match=error_message):
        fixed_byte_page_geometry(
            physical_page_bytes=physical_page_bytes,
            page_header_bytes=page_header_bytes,
            quantizer_metadata_bytes=quantizer_metadata_bytes,
            bytes_per_token=bytes_per_token,
        )

def test_input_batch_derives_byte_based_capacity_from_quantizer_id() -> None:
    input_batch = InputBatch(
        max_num_reqs=2,
        max_model_len=64,
        max_num_batched_tokens=8,
        block_sizes=[16],
        is_pooling_model=False,
        pin_memory=False,
        device=torch.device("cpu"),
        vocab_size=32000,
    )
    request = CachedRequestState(
        req_id="quantizer-1-request",
        prompt_token_ids=[1, 2, 3],
        mm_kwargs=[],
        mm_positions=[],
        sampling_params=SamplingParams(temperature=0.0),
        pooling_params=None,
        generator=None,
        block_ids=([],),
        num_computed_tokens=0,
        output_token_ids=[],
        quantizer_id=1,
    )

    request_index = input_batch.add_request(request)

    assert request_index == 0
    assert input_batch.quantizer_id_cpu[request_index] == 1

    # Baseline: 16 tokens * 32 synthetic bytes/token = 512 physical bytes.
    # Quantizer 1: (512 - 256 metadata bytes) / 16 bytes/token = 16 tokens.
    assert input_batch.tokens_per_page_cpu[request_index] == 16

def test_quantized_mapping_interleaves_requests_at_page_boundaries() -> None:
    block_table = BlockTable(
        block_size=16,
        max_num_reqs=3,
        max_num_blocks_per_req=4,
        max_num_batched_tokens=16,
        pin_memory=False,
        device=torch.device("cpu"),
    )

    # Logical-page -> physical-page mappings.
    block_table.add_row([7, 2, 11, 13], row_idx=0)
    block_table.add_row([5, 19, 23, 29], row_idx=1)
    block_table.add_row([31, 37, 41, 43], row_idx=2)

    # Request 1 has twice the token capacity of requests 0 and 2.
    tokens_per_page_by_request = np.array([16, 32, 16], dtype=np.int32)

    # Requests are intentionally interleaved.
    request_indices = np.array([0, 1, 0, 1, 2, 1, 2], dtype=np.int32)
    positions = np.array([15, 15, 16, 32, 31, 63, 32], dtype=np.int32)

    page_ids, page_offsets = block_table.compute_quantized_mapping(
        request_indices,
        positions,
        tokens_per_page_by_request,
    )

    np.testing.assert_array_equal(
        page_ids,
        np.array([7, 5, 2, 19, 37, 19, 41], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        page_offsets,
        np.array([15, 15, 0, 0, 15, 31, 0], dtype=np.int32),
    )


def test_quantized_mapping_rejects_logical_page_beyond_capacity() -> None:
    block_table = BlockTable(
        block_size=16,
        max_num_reqs=1,
        max_num_blocks_per_req=2,
        max_num_batched_tokens=4,
        pin_memory=False,
        device=torch.device("cpu"),
    )
    block_table.add_row([10, 11], row_idx=0)

    # With 16 tokens/page and capacity for logical pages 0 and 1, position 32
    # would require logical page 2 and must not index beyond the table row.
    with pytest.raises(ValueError, match="exceeds BlockTable capacity"):
        block_table.compute_quantized_mapping(
            np.array([0], dtype=np.int32),
            np.array([32], dtype=np.int32),
            np.array([16], dtype=np.int32),
        )

def test_quantized_mapping_commit_copies_page_ids_and_offsets() -> None:
    block_table = _make_block_table()

    page_ids = np.array([10, 11, 20, 21], dtype=np.int32)
    page_offsets = np.array([15, 0, 31, 0], dtype=np.int32)

    block_table.set_quantized_mapping(page_ids, page_offsets)
    block_table.commit_quantized_mapping(num_tokens=4)

    assert block_table.quantized_page_ids[:4].tolist() == [10, 11, 20, 21]
    assert block_table.quantized_page_offsets[:4].tolist() == [15, 0, 31, 0]

def test_compute_and_commit_quantized_mapping() -> None:
    block_table = _make_block_table()
    block_table.add_row([10, 11, 12, 13], row_idx=0)
    block_table.add_row([20, 21, 22, 23], row_idx=1)

    block_table.compute_and_set_quantized_mapping(
        req_indices=np.array([0, 0, 1, 1], dtype=np.int32),
        positions=np.array([15, 16, 31, 32], dtype=np.int32),
        tokens_per_page_by_request=np.array([16, 32], dtype=np.int32),
    )
    block_table.commit_quantized_mapping(num_tokens=4)

    assert block_table.quantized_page_ids[:4].tolist() == [10, 11, 20, 21]
    assert block_table.quantized_page_offsets[:4].tolist() == [15, 0, 31, 0]