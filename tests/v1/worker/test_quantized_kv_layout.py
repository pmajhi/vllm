# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest
import torch

from vllm.v1.worker.block_table import BlockTable, MultiGroupBlockTable
from vllm.v1.worker.quantized_kv_layout import (
    get_quantized_kv_page_layout,
    tokens_per_page_for_quantizer,
)


@pytest.mark.parametrize(
    ("quantizer_id", "baseline_tokens_per_page", "expected_tokens_per_page"),
    [
        (0, 16, 16),
        (1, 16, 32),
        (0, 32, 32),
        (1, 32, 64),
    ],
)
def test_tokens_per_page_for_supported_quantizer(
    quantizer_id: int,
    baseline_tokens_per_page: int,
    expected_tokens_per_page: int,
) -> None:
    assert tokens_per_page_for_quantizer(
        quantizer_id,
        baseline_tokens_per_page,
    ) == expected_tokens_per_page


def test_get_quantized_kv_page_layout_returns_registered_id() -> None:
    layout = get_quantized_kv_page_layout(1)

    assert layout.quantizer_id == 1
    assert layout.tokens_per_page_multiplier == 2


@pytest.mark.parametrize("quantizer_id", [-1, 2, 99])
def test_unknown_quantizer_id_is_rejected(quantizer_id: int) -> None:
    with pytest.raises(ValueError, match="Unsupported quantizer_id"):
        get_quantized_kv_page_layout(quantizer_id)


@pytest.mark.parametrize("baseline_tokens_per_page", [0, -1])
def test_nonpositive_baseline_capacity_is_rejected(
    baseline_tokens_per_page: int,
) -> None:
    with pytest.raises(ValueError, match="baseline_tokens_per_page"):
        tokens_per_page_for_quantizer(
            0,
            baseline_tokens_per_page,
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
