# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Geometry-derived fixed-byte page sizing for heterogeneous KV formats."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CacheGenKVPageGeometry:
    """One cache group's shared physical-page geometry."""

    tokens_per_page: int
    num_kv_heads: int
    head_size: int
    alignment_nbytes: int = 256

    def __post_init__(self) -> None:
        if self.tokens_per_page <= 0:
            raise ValueError("tokens_per_page must be positive")
        if self.num_kv_heads <= 0:
            raise ValueError("num_kv_heads must be positive")
        if self.head_size <= 0:
            raise ValueError("head_size must be positive")
        if self.alignment_nbytes <= 0:
            raise ValueError("alignment_nbytes must be positive")
        if self.alignment_nbytes & (self.alignment_nbytes - 1):
            raise ValueError("alignment_nbytes must be a power of two")

    @property
    def kv_values_per_page(self) -> int:
        """Return K or V scalar values stored in one token page."""
        return self.tokens_per_page * self.num_kv_heads * self.head_size

    @property
    def adaptive_int8_payload_nbytes(self) -> int:
        """Return bytes for both INT8 K and INT8 V payloads."""
        return 2 * self.kv_values_per_page

    @property
    def adaptive_int8_metadata_nbytes(self) -> int:
        """Return bytes for FP32 K/V dynamic scales per KV head."""
        return 2 * self.num_kv_heads * 4

    @property
    def adaptive_int8_minimum_nbytes(self) -> int:
        """Return the unaligned minimum INT8 page size."""
        return (
            self.adaptive_int8_metadata_nbytes
            + self.adaptive_int8_payload_nbytes
        )

    @property
    def common_page_nbytes(self) -> int:
        """Return aligned common physical HBM-page size."""
        alignment = self.alignment_nbytes
        minimum = self.adaptive_int8_minimum_nbytes
        return ((minimum + alignment - 1) // alignment) * alignment
