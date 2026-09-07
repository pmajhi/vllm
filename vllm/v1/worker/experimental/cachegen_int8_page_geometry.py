# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fixed-byte geometry for CacheGen-style INT8 KV quantization.

This models CacheGen's quantization/dequantization stage only: per-vector
unsigned quantization bins plus maximum-value metadata. Arithmetic coding,
CDFs, and transport-oriented byte streams are intentionally excluded because
this layout targets random-access GPU-resident paged attention.
"""

from dataclasses import dataclass

from vllm.v1.worker.experimental.hetero_kv_page import (
    HETERO_KV_PAGE_BYTES,
)


CACHEGEN_INT8_PAGE_HEADER_BYTES = 64
UINT8_BYTES = 1
FP16_BYTES = 2
CACHEGEN_INT8_BINS = 256


@dataclass(frozen=True)
class CacheGenInt8PageGeometry:
    """Byte accounting for CacheGen-style quantization without entropy coding."""

    page_bytes: int
    num_kv_heads: int
    head_size: int
    tokens_per_page: int
    key_payload_bytes: int
    value_payload_bytes: int
    key_max_bytes: int
    value_max_bytes: int
    used_bytes: int

    @property
    def tail_bytes(self) -> int:
        return self.page_bytes - self.used_bytes

    @property
    def name(self) -> str:
        return "cachegen_int8_quant_only"

    @property
    def bytes_per_token(self) -> int:
        return (
            self.key_payload_bytes
            + self.value_payload_bytes
            + self.key_max_bytes
            + self.value_max_bytes
        ) // self.tokens_per_page


def make_cachegen_int8_page_geometry(
    *,
    page_bytes: int = HETERO_KV_PAGE_BYTES,
    num_kv_heads: int,
    head_size: int,
) -> CacheGenInt8PageGeometry:
    """Plan one fixed-byte CacheGen-style quantization-only page."""
    if page_bytes <= 0:
        raise ValueError("page_bytes must be positive")
    if num_kv_heads <= 0:
        raise ValueError("num_kv_heads must be positive")
    if head_size <= 0:
        raise ValueError("head_size must be positive")

    payload_bytes_per_token = num_kv_heads * head_size * UINT8_BYTES
    max_bytes_per_token = num_kv_heads * FP16_BYTES
    bytes_per_token = 2 * (
        payload_bytes_per_token + max_bytes_per_token
    )

    usable_bytes = page_bytes - CACHEGEN_INT8_PAGE_HEADER_BYTES
    tokens_per_page = usable_bytes // bytes_per_token
    if tokens_per_page <= 0:
        raise ValueError(
            "page_bytes has insufficient usable bytes for one CacheGen INT8 "
            "K/V token"
        )

    key_payload_bytes = tokens_per_page * payload_bytes_per_token
    value_payload_bytes = tokens_per_page * payload_bytes_per_token
    key_max_bytes = tokens_per_page * max_bytes_per_token
    value_max_bytes = tokens_per_page * max_bytes_per_token
    used_bytes = (
        CACHEGEN_INT8_PAGE_HEADER_BYTES
        + key_payload_bytes
        + value_payload_bytes
        + key_max_bytes
        + value_max_bytes
    )

    return CacheGenInt8PageGeometry(
        page_bytes=page_bytes,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        tokens_per_page=tokens_per_page,
        key_payload_bytes=key_payload_bytes,
        value_payload_bytes=value_payload_bytes,
        key_max_bytes=key_max_bytes,
        value_max_bytes=value_max_bytes,
        used_bytes=used_bytes,
    )
