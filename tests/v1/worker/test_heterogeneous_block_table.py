# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest
import torch

from vllm.v1.worker.block_table import BlockTable, MultiGroupBlockTable


def make_block_table() -> BlockTable:
    table = BlockTable(
        block_size=16,
        max_num_reqs=3,
        max_num_blocks_per_req=8,
        max_num_batched_tokens=16,
        pin_memory=False,
        device=torch.device("cpu"),
    )
    table.add_row([10, 11, 12, 13], row_idx=0)
    table.add_row([20, 21, 22, 23], row_idx=1)
    table.add_row([30, 31, 32, 33], row_idx=2)
    return table


def test_quantized_mapping_uses_per_request_token_capacity() -> None:
    table = make_block_table()
    request_indices = np.array([0, 0, 1, 1, 2, 2], dtype=np.int32)
    positions = np.array([15, 16, 31, 32, 23, 24], dtype=np.int32)
    tokens_per_page = np.array([16, 32, 24], dtype=np.int32)

    page_ids, page_offsets = table.compute_quantized_mapping(
        request_indices,
        positions,
        tokens_per_page,
    )

    np.testing.assert_array_equal(page_ids, [10, 11, 20, 21, 30, 31])
    np.testing.assert_array_equal(page_offsets, [15, 0, 31, 0, 23, 0])


def test_quantized_mapping_can_be_committed_to_device_tensors() -> None:
    table = make_block_table()
    request_indices = np.array([0, 1, 2], dtype=np.int32)
    positions = np.array([16, 32, 48], dtype=np.int32)
    tokens_per_page = np.array([16, 32, 24], dtype=np.int32)

    table.compute_and_set_quantized_mapping(
        request_indices,
        positions,
        tokens_per_page,
    )
    table.commit_quantized_mapping(num_tokens=3)

    assert table.quantized_page_ids[:3].tolist() == [11, 21, 32]
    assert table.quantized_page_offsets[:3].tolist() == [0, 0, 0]


def test_quantized_mapping_rejects_nonpositive_capacity() -> None:
    table = make_block_table()

    with pytest.raises(ValueError, match="positive tokens_per_page"):
        table.compute_quantized_mapping(
            np.array([1], dtype=np.int32),
            np.array([0], dtype=np.int32),
            np.array([16, 0, 24], dtype=np.int32),
        )


def test_quantized_mapping_rejects_page_index_outside_row_capacity() -> None:
    table = make_block_table()

    with pytest.raises(ValueError, match="exceeds BlockTable capacity"):
        table.compute_quantized_mapping(
            np.array([0], dtype=np.int32),
            np.array([128], dtype=np.int32),
            np.array([16, 32, 24], dtype=np.int32),
        )


def test_quantized_mapping_rejects_wrong_capacity_vector_shape() -> None:
    table = make_block_table()

    with pytest.raises(ValueError, match="must have shape"):
        table.compute_quantized_mapping(
            np.array([0], dtype=np.int32),
            np.array([0], dtype=np.int32),
            np.array([16, 32], dtype=np.int32),
        )


def test_multigroup_quantized_mapping_matches_each_group() -> None:
    tables = MultiGroupBlockTable(
        max_num_reqs=2,
        max_model_len=128,
        max_num_batched_tokens=8,
        pin_memory=False,
        device=torch.device("cpu"),
        block_sizes=[16, 16],
    )
    tables.add_row(([7, 8], [70, 80]), row_idx=0)
    tables.add_row(([9, 10], [90, 100]), row_idx=1)

    page_ids, page_offsets = tables.compute_quantized_mapping(
        req_indices=np.array([0, 1], dtype=np.int32),
        positions=np.array([16, 32], dtype=np.int32),
        tokens_per_page_by_request=np.array([16, 32], dtype=np.int32),
    )

    assert len(page_ids) == 2
    assert len(page_offsets) == 2
    np.testing.assert_array_equal(page_ids[0], [8, 10])
    np.testing.assert_array_equal(page_ids[1], [80, 100])
    np.testing.assert_array_equal(page_offsets[0], [0, 0])
    np.testing.assert_array_equal(page_offsets[1], [0, 0])
