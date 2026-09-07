# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Per-request fixed-byte KV-page layout policy.

This module defines the page-capacity contract for each quantizer ID. It does
not change scheduler allocation or attention-kernel behavior; callers must use
the same policy before activating a non-default layout end to end.

Non-default layouts cannot share the baseline attention path: current V1
backends select one cache shape and one physical block size for an execution
batch. A non-default layout therefore requires either isolated execution with
a matching backend or a backend that explicitly supports mixed page layouts.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class QuantizedKVPageLayout:
    """Synthetic fixed-byte page specification for one quantizer format.

    The byte values are test-policy placeholders, not claims about a real
    KIVI, FP8, TurboQuant, or CacheGen representation.
    """

    quantizer_id: int
    bytes_per_token: int
    quantizer_metadata_bytes: int


_PAGE_LAYOUTS: dict[int, QuantizedKVPageLayout] = {
    0: QuantizedKVPageLayout(
        quantizer_id=0,
        bytes_per_token=32,
        quantizer_metadata_bytes=0,
    ),
    1: QuantizedKVPageLayout(
        quantizer_id=1,
        bytes_per_token=16,
        quantizer_metadata_bytes=256,
    ),
}

@dataclass(frozen=True)
class FixedBytePageGeometry:
    """Byte-derived geometry for one fixed-size physical KV page."""

    physical_page_bytes: int
    page_header_bytes: int
    quantizer_metadata_bytes: int
    bytes_per_token: int
    tokens_per_page: int
    internal_padding_bytes: int

def get_quantized_kv_page_layout(
    quantizer_id: int,
) -> QuantizedKVPageLayout:
    """Return the registered fixed-byte page layout for a quantizer ID."""
    try:
        return _PAGE_LAYOUTS[quantizer_id]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported quantizer_id: {quantizer_id}. "
            f"Supported IDs: {sorted(_PAGE_LAYOUTS)}"
        ) from exc


def tokens_per_page_for_quantizer(
    quantizer_id: int,
    physical_page_bytes: int,
) -> int:
    """Derive capacity from an explicit fixed physical-page byte budget."""
    layout = get_quantized_kv_page_layout(quantizer_id)
    geometry = fixed_byte_page_geometry(
        physical_page_bytes=physical_page_bytes,
        page_header_bytes=0,
        quantizer_metadata_bytes=layout.quantizer_metadata_bytes,
        bytes_per_token=layout.bytes_per_token,
    )
    return geometry.tokens_per_page

def fixed_byte_page_geometry(
    *,
    physical_page_bytes: int,
    page_header_bytes: int,
    quantizer_metadata_bytes: int,
    bytes_per_token: int,
) -> FixedBytePageGeometry:
    """Compute token capacity and padding for one fixed-byte page."""
    if physical_page_bytes <= 0:
        raise ValueError("physical_page_bytes must be positive")
    if page_header_bytes < 0:
        raise ValueError("page_header_bytes must be nonnegative")
    if quantizer_metadata_bytes < 0:
        raise ValueError("quantizer_metadata_bytes must be nonnegative")
    if bytes_per_token <= 0:
        raise ValueError("bytes_per_token must be positive")

    usable_payload_bytes = (
        physical_page_bytes
        - page_header_bytes
        - quantizer_metadata_bytes
    )

    if usable_payload_bytes < 0:
        raise ValueError(
            "physical_page_bytes must cover header and quantizer metadata"
        )

    tokens_per_page = usable_payload_bytes // bytes_per_token

    if tokens_per_page <= 0:
        raise ValueError(
            "physical page has insufficient usable bytes for one token"
        )

    internal_padding_bytes = (
        usable_payload_bytes
        - tokens_per_page * bytes_per_token
    )

    return FixedBytePageGeometry(
        physical_page_bytes=physical_page_bytes,
        page_header_bytes=page_header_bytes,
        quantizer_metadata_bytes=quantizer_metadata_bytes,
        bytes_per_token=bytes_per_token,
        tokens_per_page=tokens_per_page,
        internal_padding_bytes=internal_padding_bytes,
    )
