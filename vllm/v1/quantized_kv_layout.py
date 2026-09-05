# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Per-request fixed-byte KV-page layout policy.

This module defines the page-capacity contract for each quantizer ID. It does
not change scheduler allocation or attention-kernel behavior; callers must use
the same policy before activating a non-default layout end to end.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class QuantizedKVPageLayout:
    """Fixed-byte page geometry for one quantizer format."""

    quantizer_id: int
    tokens_per_page_multiplier: int


_PAGE_LAYOUTS: dict[int, QuantizedKVPageLayout] = {
    0: QuantizedKVPageLayout(
        quantizer_id=0,
        tokens_per_page_multiplier=1,
    ),
    1: QuantizedKVPageLayout(
        quantizer_id=1,
        tokens_per_page_multiplier=2,
    ),
}


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
    baseline_tokens_per_page: int,
) -> int:
    """Derive one quantizer's page capacity from the baseline block size."""
    if baseline_tokens_per_page <= 0:
        raise ValueError("baseline_tokens_per_page must be positive")

    layout = get_quantized_kv_page_layout(quantizer_id)
    tokens_per_page = (
        baseline_tokens_per_page * layout.tokens_per_page_multiplier
    )

    if tokens_per_page <= 0:
        raise ValueError("tokens_per_page must be positive")

    return tokens_per_page
