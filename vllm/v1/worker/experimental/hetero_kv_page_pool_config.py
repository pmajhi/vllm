# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration for the experimental GPU heterogeneous KV byte-page pool."""

import os

from vllm.v1.worker.experimental.hetero_kv_page_config import (
    is_hetero_kv_page_planning_enabled,
)

HETERO_KV_PAGE_POOL_ENV_VAR = "VLLM_EXPERIMENTAL_HETERO_KV_PAGE_POOL"
HETERO_KV_PAGE_POOL_PAGES_ENV_VAR = (
    "VLLM_EXPERIMENTAL_HETERO_KV_PAGE_POOL_PAGES"
)


def is_hetero_kv_page_pool_enabled() -> bool:
    """Return whether the experimental GPU byte-page pool is enabled.

    Page-pool storage depends on heterogeneous page planning. Requiring the
    planning flag prevents a GPU allocation that has no model-specific layout
    interpretation.
    """
    configured_value = os.getenv(HETERO_KV_PAGE_POOL_ENV_VAR)
    if configured_value is None:
        return False

    normalized = configured_value.strip().lower()
    if normalized in ("1", "true", "yes"):
        if not is_hetero_kv_page_planning_enabled():
            raise ValueError(
                f"{HETERO_KV_PAGE_POOL_ENV_VAR} requires "
                "VLLM_EXPERIMENTAL_HETERO_KV_PAGE_PLANNING=1"
            )
        return True
    if normalized in ("0", "false", "no"):
        return False
    raise ValueError(
        f"{HETERO_KV_PAGE_POOL_ENV_VAR} must be one of "
        "'1', '0', 'true', 'false', 'yes', or 'no', "
        f"got {configured_value!r}"
    )


def get_hetero_kv_page_pool_pages() -> int:
    """Return the explicitly configured number of experimental GPU pages."""
    configured_value = os.getenv(HETERO_KV_PAGE_POOL_PAGES_ENV_VAR)
    if configured_value is None:
        raise ValueError(
            f"{HETERO_KV_PAGE_POOL_PAGES_ENV_VAR} must be set when "
            f"{HETERO_KV_PAGE_POOL_ENV_VAR}=1"
        )

    try:
        num_pages = int(configured_value)
    except ValueError as exc:
        raise ValueError(
            f"{HETERO_KV_PAGE_POOL_PAGES_ENV_VAR} must be a positive integer, "
            f"got {configured_value!r}"
        ) from exc

    if num_pages <= 0:
        raise ValueError(
            f"{HETERO_KV_PAGE_POOL_PAGES_ENV_VAR} must be positive, "
            f"got {num_pages}"
        )
    return num_pages
