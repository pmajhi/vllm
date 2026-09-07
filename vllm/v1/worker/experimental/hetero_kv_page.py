# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Common byte-page definitions for experimental heterogeneous KV caches."""

from dataclasses import dataclass
from enum import IntEnum


HETERO_KV_PAGE_BYTES = 64 * 1024
HETERO_KV_LAYOUT_VERSION = 1


class KVCodecId(IntEnum):
    """Stable identifiers carried with a request for its entire lifetime."""

    UNIFORM_INT8 = 0
    KIVI = 1
    TURBOQUANT = 2


@dataclass(frozen=True)
class PageGeometry:
    """Byte accounting and token capacity for one codec in one physical page."""

    codec_id: KVCodecId
    page_bytes: int
    header_bytes: int
    metadata_bytes: int
    payload_bytes: int
    alignment_bytes: int
    token_capacity: int
    bytes_per_token_effective: float
    layout_version: int = HETERO_KV_LAYOUT_VERSION

    def __post_init__(self) -> None:
        if self.page_bytes <= 0:
            raise ValueError("page_bytes must be positive")
        if self.header_bytes < 0:
            raise ValueError("header_bytes must be nonnegative")
        if self.metadata_bytes < 0:
            raise ValueError("metadata_bytes must be nonnegative")
        if self.payload_bytes < 0:
            raise ValueError("payload_bytes must be nonnegative")
        if self.alignment_bytes < 0:
            raise ValueError("alignment_bytes must be nonnegative")
        if self.token_capacity <= 0:
            raise ValueError("token_capacity must be positive")
        if self.bytes_per_token_effective <= 0:
            raise ValueError("bytes_per_token_effective must be positive")
        if self.used_bytes > self.page_bytes:
            raise ValueError(
                "page layout exceeds page capacity: "
                f"{self.used_bytes} > {self.page_bytes}"
            )

    @property
    def used_bytes(self) -> int:
        return (
            self.header_bytes
            + self.metadata_bytes
            + self.payload_bytes
            + self.alignment_bytes
        )

    @property
    def tail_bytes(self) -> int:
        return self.page_bytes - self.used_bytes

    @property
    def utilization(self) -> float:
        return self.used_bytes / self.page_bytes


def logical_page_and_offset(
    position: int,
    token_capacity: int,
) -> tuple[int, int]:
    """Map one nonnegative logical token position into a codec page location."""
    if position < 0:
        raise ValueError("position must be nonnegative")
    if token_capacity <= 0:
        raise ValueError("token_capacity must be positive")
    return divmod(position, token_capacity)


def heterogeneous_page_mapping(
    *,
    block_table: list[list[int]],
    request_indices: list[int],
    positions: list[int],
    tokens_per_page: list[int],
) -> tuple[list[int], list[int]]:
    """Map scheduled tokens to physical fixed-byte pages and local offsets.

    Args:
        block_table: Per-request rows mapping logical page index to physical
            page ID. Every row must contain at least the pages addressed by
            its scheduled token positions.
        request_indices: Request row for each scheduled token.
        positions: Logical token position for each scheduled token.
        tokens_per_page: Codec-specific page capacity for each request row.

    Returns:
        A pair of lists: physical page IDs and page-local token offsets.
    """
    num_requests = len(block_table)
    if len(tokens_per_page) != num_requests:
        raise ValueError(
            "tokens_per_page length must match the number of block-table rows"
        )
    if len(request_indices) != len(positions):
        raise ValueError("request_indices and positions must have equal length")

    page_ids: list[int] = []
    page_offsets: list[int] = []

    for request_index, position in zip(request_indices, positions):
        if request_index < 0 or request_index >= num_requests:
            raise ValueError("request index is outside the block table")

        token_capacity = tokens_per_page[request_index]
        logical_page, page_offset = logical_page_and_offset(
            position,
            token_capacity,
        )
        row = block_table[request_index]
        if logical_page >= len(row):
            raise ValueError(
                "block-table row does not contain the logical page for "
                f"request {request_index} position {position}"
            )

        physical_page_id = row[logical_page]
        if physical_page_id < 0:
            raise ValueError("physical page ID must be nonnegative")

        page_ids.append(physical_page_id)
        page_offsets.append(page_offset)

    return page_ids, page_offsets
