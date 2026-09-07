# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration helpers for experimental heterogeneous fixed-byte KV pages."""

import os

from vllm.v1.worker.experimental.hetero_kv_codec_registry import (
    DEFAULT_HETERO_KV_PAGE_BYTES,
)

HETERO_KV_PAGE_BYTES_ENV_VAR = "VLLM_EXPERIMENTAL_HETERO_KV_PAGE_BYTES"


def get_hetero_kv_page_bytes() -> int:
    """Return the configured experimental heterogeneous KV page size in bytes.

    The default is 128 KiB. Parsing is intentionally centralized so every
    experimental planning path uses the same physical-page contract.
    """
    configured_value = os.getenv(HETERO_KV_PAGE_BYTES_ENV_VAR)
    if configured_value is None:
        return DEFAULT_HETERO_KV_PAGE_BYTES

    try:
        page_bytes = int(configured_value)
    except ValueError as exc:
        raise ValueError(
            f"{HETERO_KV_PAGE_BYTES_ENV_VAR} must be a positive integer, "
            f"got {configured_value!r}"
        ) from exc

    if page_bytes <= 0:
        raise ValueError(
            f"{HETERO_KV_PAGE_BYTES_ENV_VAR} must be positive, "
            f"got {page_bytes}"
        )
    return page_bytes
