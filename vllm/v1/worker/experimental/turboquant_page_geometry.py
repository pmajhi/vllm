# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fixed-byte compressed-page geometry planner for TurboQuant-style KV.

TurboQuant-style key storage contains packed scalar-quantizer indices, packed
QJL sign bits, and FP16 norm metadata. Value storage contains packed low-bit
codes plus FP16 scale and zero-point per head-dimension group.

The recent full-precision K/V ring buffer and layer-global rotation/codebook
state are intentionally accounted for outside shared compressed pages.
"""

from dataclasses import dataclass
from math import ceil

from vllm.v1.worker.experimental.hetero_kv_page import (
    HETERO_KV_PAGE_BYTES,
)


TURBOQUANT_PAGE_HEADER_BYTES = 64
FP16_BYTES = 2
SUPPORTED_TURBOQUANT_KEY_BITS = (2, 3, 4)
SUPPORTED_TURBOQUANT_VALUE_BITS = (2, 4)


@dataclass(frozen=True)
class TurboQuantPageGeometry:
    """Byte accounting for one compressed fixed-byte TurboQuant page."""

    page_bytes: int
    num_kv_heads: int
    head_size: int
    key_bits: int
    value_bits: int
    value_group_size: int
    qjl_projections: int
    tokens_per_page: int
    key_mse_index_bytes: int
    key_qjl_sign_bytes: int
    key_metadata_bytes: int
    value_payload_bytes: int
    value_metadata_bytes: int
    used_bytes: int

    @property
    def tail_bytes(self) -> int:
        return self.page_bytes - self.used_bytes

    @property
    def value_groups_per_token(self) -> int:
        return ceil(self.head_size / self.value_group_size)

    @property
    def name(self) -> str:
        return f"turboquant_k{self.key_bits}_v{self.value_bits}"


def _packed_bytes(elements: int, bits: int) -> int:
    return (elements * bits + 7) // 8


def turboquant_ring_buffer_bytes(
    *,
    num_kv_heads: int,
    head_size: int,
    ring_capacity: int,
    dtype_bytes: int = FP16_BYTES,
) -> int:
    """Return separate full-precision K/V ring-buffer storage in bytes."""
    if num_kv_heads <= 0:
        raise ValueError("num_kv_heads must be positive")
    if head_size <= 0:
        raise ValueError("head_size must be positive")
    if ring_capacity < 0:
        raise ValueError("ring_capacity must be nonnegative")
    if dtype_bytes <= 0:
        raise ValueError("dtype_bytes must be positive")

    return ring_capacity * 2 * num_kv_heads * head_size * dtype_bytes


def make_turboquant_page_geometry(
    *,
    page_bytes: int = HETERO_KV_PAGE_BYTES,
    num_kv_heads: int,
    head_size: int,
    key_bits: int = 3,
    value_bits: int = 2,
    value_group_size: int = 32,
    qjl_projections: int | None = None,
) -> TurboQuantPageGeometry:
    """Plan a compressed TurboQuant page using documented storage categories.

    ``qjl_projections`` is the number of packed one-bit QJL signs retained per
    key vector. If omitted, one sign per head-dimension coordinate is assumed.
    """
    if page_bytes <= 0:
        raise ValueError("page_bytes must be positive")
    if num_kv_heads <= 0:
        raise ValueError("num_kv_heads must be positive")
    if head_size <= 0:
        raise ValueError("head_size must be positive")
    if key_bits not in SUPPORTED_TURBOQUANT_KEY_BITS:
        raise ValueError(
            "key_bits must be one of "
            f"{SUPPORTED_TURBOQUANT_KEY_BITS}"
        )
    if value_bits not in SUPPORTED_TURBOQUANT_VALUE_BITS:
        raise ValueError(
            "value_bits must be one of "
            f"{SUPPORTED_TURBOQUANT_VALUE_BITS}"
        )
    if value_group_size <= 0:
        raise ValueError("value_group_size must be positive")
    if qjl_projections is None:
        qjl_projections = head_size
    if qjl_projections <= 0:
        raise ValueError("qjl_projections must be positive")

    value_groups = ceil(head_size / value_group_size)
    usable_bytes = page_bytes - TURBOQUANT_PAGE_HEADER_BYTES

    # Per key token/head:
    # - one key MSE code per coordinate, key_bits packed;
    # - qjl_projections one-bit signs;
    # - residual_norm and norm, both FP16.
    key_bytes_per_token = num_kv_heads * (
        _packed_bytes(head_size, key_bits)
        + _packed_bytes(qjl_projections, 1)
        + 2 * FP16_BYTES
    )

    # Per value token/head:
    # - head_size low-bit values;
    # - one FP16 scale and FP16 zero per dim group.
    value_bytes_per_token = num_kv_heads * (
        _packed_bytes(head_size, value_bits)
        + 2 * value_groups * FP16_BYTES
    )

    bytes_per_token = key_bytes_per_token + value_bytes_per_token
    tokens_per_page = usable_bytes // bytes_per_token
    if tokens_per_page <= 0:
        raise ValueError(
            "page_bytes has insufficient usable bytes for one TurboQuant token"
        )

    key_mse_index_bytes = (
        tokens_per_page
        * num_kv_heads
        * _packed_bytes(head_size, key_bits)
    )
    key_qjl_sign_bytes = (
        tokens_per_page
        * num_kv_heads
        * _packed_bytes(qjl_projections, 1)
    )
    key_metadata_bytes = (
        tokens_per_page
        * num_kv_heads
        * 2
        * FP16_BYTES
    )
    value_payload_bytes = (
        tokens_per_page
        * num_kv_heads
        * _packed_bytes(head_size, value_bits)
    )
    value_metadata_bytes = (
        tokens_per_page
        * num_kv_heads
        * value_groups
        * 2
        * FP16_BYTES
    )
    used_bytes = (
        TURBOQUANT_PAGE_HEADER_BYTES
        + key_mse_index_bytes
        + key_qjl_sign_bytes
        + key_metadata_bytes
        + value_payload_bytes
        + value_metadata_bytes
    )

    return TurboQuantPageGeometry(
        page_bytes=page_bytes,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        key_bits=key_bits,
        value_bits=value_bits,
        value_group_size=value_group_size,
        qjl_projections=qjl_projections,
        tokens_per_page=tokens_per_page,
        key_mse_index_bytes=key_mse_index_bytes,
        key_qjl_sign_bytes=key_qjl_sign_bytes,
        key_metadata_bytes=key_metadata_bytes,
        value_payload_bytes=value_payload_bytes,
        value_metadata_bytes=value_metadata_bytes,
        used_bytes=used_bytes,
    )
