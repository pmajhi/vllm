# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.v1.worker.experimental.hetero_kv_page import (
    HETERO_KV_PAGE_BYTES,
    KVCodecId,
    PageGeometry,
    logical_page_and_offset,
)


def test_page_geometry_accounts_for_each_byte() -> None:
    geometry = PageGeometry(
        codec_id=KVCodecId.UNIFORM_INT8,
        page_bytes=HETERO_KV_PAGE_BYTES,
        header_bytes=64,
        metadata_bytes=2048,
        payload_bytes=60_000,
        alignment_bytes=128,
        token_capacity=128,
        bytes_per_token_effective=(
            60_000 + 2048 + 128 + 64
        ) / 128,
    )

    assert geometry.used_bytes == 62_240
    assert geometry.tail_bytes == HETERO_KV_PAGE_BYTES - 62_240
    assert 0 < geometry.utilization <= 1


def test_page_geometry_rejects_overflow() -> None:
    with pytest.raises(ValueError, match="exceeds page capacity"):
        PageGeometry(
            codec_id=KVCodecId.KIVI,
            page_bytes=1024,
            header_bytes=64,
            metadata_bytes=128,
            payload_bytes=900,
            alignment_bytes=64,
            token_capacity=1,
            bytes_per_token_effective=1156.0,
        )


@pytest.mark.parametrize(
    ("position", "token_capacity", "expected"),
    [
        (0, 16, (0, 0)),
        (15, 16, (0, 15)),
        (16, 16, (1, 0)),
        (17, 16, (1, 1)),
        (31, 16, (1, 15)),
        (32, 16, (2, 0)),
        (0, 32, (0, 0)),
        (31, 32, (0, 31)),
        (32, 32, (1, 0)),
        (65, 32, (2, 1)),
    ],
)
def test_logical_page_and_offset(
    position: int,
    token_capacity: int,
    expected: tuple[int, int],
) -> None:
    assert logical_page_and_offset(position, token_capacity) == expected


@pytest.mark.parametrize(
    ("position", "token_capacity", "message"),
    [
        (-1, 16, "position must be nonnegative"),
        (0, 0, "token_capacity must be positive"),
        (0, -1, "token_capacity must be positive"),
    ],
)
def test_logical_page_and_offset_rejects_invalid_values(
    position: int,
    token_capacity: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        logical_page_and_offset(position, token_capacity)

from vllm.v1.worker.experimental.hetero_kv_page import (
    heterogeneous_page_mapping,
)


def test_heterogeneous_page_mapping_uses_request_specific_capacities() -> None:
    block_table = [
        [10, 11, 12],
        [20, 21, 22],
        [30, 31, 32],
    ]
    tokens_per_page = [16, 32, 24]

    page_ids, page_offsets = heterogeneous_page_mapping(
        block_table=block_table,
        request_indices=[0, 0, 1, 1, 2, 2],
        positions=[15, 16, 31, 32, 23, 24],
        tokens_per_page=tokens_per_page,
    )

    assert page_ids == [10, 11, 20, 21, 30, 31]
    assert page_offsets == [15, 0, 31, 0, 23, 0]


def test_heterogeneous_page_mapping_handles_interleaved_requests() -> None:
    page_ids, page_offsets = heterogeneous_page_mapping(
        block_table=[
            [7, 8, 9],
            [3, 4, 5],
        ],
        request_indices=[1, 0, 1, 0, 0],
        positions=[0, 17, 16, 31, 32],
        tokens_per_page=[16, 16],
    )

    assert page_ids == [3, 8, 4, 8, 9]
    assert page_offsets == [0, 1, 0, 15, 0]


@pytest.mark.parametrize(
    ("block_table", "request_indices", "positions", "tokens_per_page",
     "message"),
    [
        (
            [[0]],
            [0],
            [0],
            [],
            "tokens_per_page length must match",
        ),
        (
            [[0]],
            [0],
            [],
            [16],
            "request_indices and positions must have equal length",
        ),
        (
            [[0]],
            [1],
            [0],
            [16],
            "request index is outside",
        ),
        (
            [[0]],
            [0],
            [16],
            [16],
            "does not contain the logical page",
        ),
        (
            [[-1]],
            [0],
            [0],
            [16],
            "physical page ID must be nonnegative",
        ),
    ],
)
def test_heterogeneous_page_mapping_rejects_invalid_input(
    block_table: list[list[int]],
    request_indices: list[int],
    positions: list[int],
    tokens_per_page: list[int],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        heterogeneous_page_mapping(
            block_table=block_table,
            request_indices=request_indices,
            positions=positions,
            tokens_per_page=tokens_per_page,
        )
