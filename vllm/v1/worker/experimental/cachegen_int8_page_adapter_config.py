# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration for the experimental CacheGen-style INT8 page adapter."""

import os

from vllm.v1.worker.experimental.hetero_kv_page_pool_config import (
    is_hetero_kv_page_pool_enabled,
)

CACHEGEN_INT8_PAGE_ADAPTER_ENV_VAR = (
    "VLLM_EXPERIMENTAL_CACHEGEN_INT8_PAGE_ADAPTER"
)


def is_cachegen_int8_page_adapter_enabled() -> bool:
    """Return whether runner-owned CacheGen-style page storage is enabled."""
    configured_value = os.getenv(CACHEGEN_INT8_PAGE_ADAPTER_ENV_VAR)
    if configured_value is None:
        return False

    normalized = configured_value.strip().lower()
    if normalized in ("1", "true", "yes"):
        if not is_hetero_kv_page_pool_enabled():
            raise ValueError(
                f"{CACHEGEN_INT8_PAGE_ADAPTER_ENV_VAR} requires "
                "VLLM_EXPERIMENTAL_HETERO_KV_PAGE_POOL=1"
            )
        return True
    if normalized in ("0", "false", "no"):
        return False
    raise ValueError(
        f"{CACHEGEN_INT8_PAGE_ADAPTER_ENV_VAR} must be one of "
        "'1', '0', 'true', 'false', 'yes', or 'no', "
        f"got {configured_value!r}"
    )
