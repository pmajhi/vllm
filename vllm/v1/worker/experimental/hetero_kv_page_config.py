# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration helpers for experimental heterogeneous fixed-byte KV pages."""

import os

from vllm.v1.worker.experimental.hetero_kv_codec_registry import (
    DEFAULT_HETERO_KV_PAGE_BYTES,
)

HETERO_KV_PAGE_BYTES_ENV_VAR = "VLLM_EXPERIMENTAL_HETERO_KV_PAGE_BYTES"
HETERO_KV_PAGE_PLANNING_ENV_VAR = (
    "VLLM_EXPERIMENTAL_HETERO_KV_PAGE_PLANNING"
)


def is_hetero_kv_page_planning_enabled() -> bool:
    """Return whether experimental heterogeneous KV-page planning is enabled.

    The feature defaults to disabled. Accepted truthy values are ``1``,
    ``true``, and ``yes``; accepted falsy values are ``0``, ``false``, and
    ``no``. Invalid values fail early rather than silently enabling an
    experimental runtime path.
    """
    configured_value = os.getenv(HETERO_KV_PAGE_PLANNING_ENV_VAR)
    if configured_value is None:
        return False

    normalized = configured_value.strip().lower()
    if normalized in ("1", "true", "yes"):
        return True
    if normalized in ("0", "false", "no"):
        return False
    raise ValueError(
        f"{HETERO_KV_PAGE_PLANNING_ENV_VAR} must be one of "
        "'1', '0', 'true', 'false', 'yes', or 'no', "
        f"got {configured_value!r}"
    )


def get_hetero_kv_page_bytes() -> int:
    """Return the configured experimental heterogeneous KV page size in bytes.

    The default is 128 KiB. Parsing is centralized so every experimental
    planning path uses the same physical-page contract.
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
