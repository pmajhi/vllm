# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fixed-byte compressed-page geometry for a KIVI-style KV cache.

KIVI quantizes keys over token groups per channel and values per token over
head-dimension groups. Its recent full-precision residual window is sequence
state, not page-local state: it is flushed to compressed storage in whole key
groups. This module therefore accounts for compressed pages and residual bytes
separately.

Key and value bit widths are independently configurable as 2, 4, or 8 bits.
This is a geometry planner only; it does not yet encode pages or run attention.
"""

from dataclasses import dataclass
from math import ceil

from vllm.v1.worker.experimental.hetero_kv_page import (
    HETERO_KV_PAGE_BYTES,
)


KIVI_PAGE_HEADER_BYTES = 64
FP16_BYTES = 2
SUPPORTED_KIVI_BITS = (2, 4, 8)


@dataclass(frozen=True)
class KiviPageGeometry:
    """Byte accounting for one compressed fixed-byte KIVI page."""

    page_bytes: int
    num_kv_heads: int
    head_size: int
    key_bits: int
    value_bits: int
    key_group_size: int
    value_group_size: int
    quantized_tokens: int
    key_payload_bytes: int
    value_payload_bytes: int
    key_metadata_bytes: int
    value_metadata_bytes: int
    used_bytes: int

    @property
    def token_capacity(self) -> int:
        """Compressed-token capacity; residual tokens live outside this page."""
        return self.quantized_tokens

    @property
    def tail_bytes(self) -> int:
        return self.page_bytes - self.used_bytes

    @property
    def key_groups(self) -> int:
        return self.quantized_tokens // self.key_group_size

    @property
    def value_groups_per_token(self) -> int:
        return ceil(self.head_size / self.value_group_size)

    @property
    def name(self) -> str:
        return f"kivi_k{self.key_bits}_v{self.value_bits}"


def _packed_bytes(elements: int, bits: int) -> int:
    return (elements * bits + 7) // 8


def kivi_residual_bytes(
    *,
    num_kv_heads: int,
    head_size: int,
    residual_tokens: int,
    dtype_bytes: int = FP16_BYTES,
) -> int:
    """Return separate full-precision K/V residual storage in bytes."""
    if num_kv_heads <= 0:
        raise ValueError("num_kv_heads must be positive")
    if head_size <= 0:
        raise ValueError("head_size must be positive")
    if residual_tokens < 0:
        raise ValueError("residual_tokens must be nonnegative")
    if dtype_bytes <= 0:
        raise ValueError("dtype_bytes must be positive")

    return (
        residual_tokens
        * 2
        * num_kv_heads
        * head_size
        * dtype_bytes
    )


def make_kivi_page_geometry(
    *,
    page_bytes: int = HETERO_KV_PAGE_BYTES,
    num_kv_heads: int,
    head_size: int,
    key_bits: int = 2,
    value_bits: int = 2,
    key_group_size: int = 32,
    value_group_size: int = 32,
) -> KiviPageGeometry:
    """Plan one compressed KIVI page with complete page-local key groups."""
    if page_bytes <= 0:
        raise ValueError("page_bytes must be positive")
    if num_kv_heads <= 0:
        raise ValueError("num_kv_heads must be positive")
    if head_size <= 0:
        raise ValueError("head_size must be positive")
    if key_bits not in SUPPORTED_KIVI_BITS:
        raise ValueError(f"key_bits must be one of {SUPPORTED_KIVI_BITS}")
    if value_bits not in SUPPORTED_KIVI_BITS:
        raise ValueError(f"value_bits must be one of {SUPPORTED_KIVI_BITS}")
    if key_group_size <= 0:
        raise ValueError("key_group_size must be positive")
    if value_group_size <= 0:
        raise ValueError("value_group_size must be positive")

    value_groups_per_token = ceil(head_size / value_group_size)
    usable_bytes = page_bytes - KIVI_PAGE_HEADER_BYTES

    # Candidate capacity ignores K-group metadata. The loop then decreases in
    # complete K groups until codes and all scale/minimum tensors fit.
    compressed_bytes_per_token = (
        _packed_bytes(num_kv_heads * head_size, key_bits)
        + _packed_bytes(num_kv_heads * head_size, value_bits)
        + 2 * num_kv_heads * value_groups_per_token * FP16_BYTES
    )
    candidate_tokens = usable_bytes // compressed_bytes_per_token
    quantized_tokens = (
        candidate_tokens // key_group_size
    ) * key_group_size

    while quantized_tokens > 0:
        key_groups = quantized_tokens // key_group_size

        key_payload_bytes = _packed_bytes(
            quantized_tokens * num_kv_heads * head_size,
            key_bits,
        )
        value_payload_bytes = _packed_bytes(
            quantized_tokens * num_kv_heads * head_size,
            value_bits,
        )

        # Key scale + minimum per [KV head, head channel, token group].
        key_metadata_bytes = (
            2
            * key_groups
            * num_kv_heads
            * head_size
            * FP16_BYTES
        )

        # Value scale + minimum per [token, KV head, dim group].
        value_metadata_bytes = (
            2
            * quantized_tokens
            * num_kv_heads
            * value_groups_per_token
            * FP16_BYTES
        )

        used_bytes = (
            KIVI_PAGE_HEADER_BYTES
            + key_payload_bytes
            + value_payload_bytes
            + key_metadata_bytes
            + value_metadata_bytes
        )
        if used_bytes <= page_bytes:
            return KiviPageGeometry(
                page_bytes=page_bytes,
                num_kv_heads=num_kv_heads,
                head_size=head_size,
                key_bits=key_bits,
                value_bits=value_bits,
                key_group_size=key_group_size,
                value_group_size=value_group_size,
                quantized_tokens=quantized_tokens,
                key_payload_bytes=key_payload_bytes,
                value_payload_bytes=value_payload_bytes,
                key_metadata_bytes=key_metadata_bytes,
                value_metadata_bytes=value_metadata_bytes,
                used_bytes=used_bytes,
            )

        quantized_tokens -= key_group_size

    raise ValueError(
        "page_bytes has insufficient usable bytes for one complete KIVI key "
        "group"
    )
