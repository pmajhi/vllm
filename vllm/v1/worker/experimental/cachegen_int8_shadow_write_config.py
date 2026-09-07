# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration for reference CacheGen-style KV shadow writes."""

import os

from vllm.v1.worker.experimental.cachegen_int8_page_adapter_config import (
    is_cachegen_int8_page_adapter_enabled,
)

CACHEGEN_INT8_SHADOW_WRITE_ENV_VAR = (
    "VLLM_EXPERIMENTAL_CACHEGEN_INT8_SHADOW_WRITE"
)


def is_cachegen_int8_shadow_write_enabled() -> bool:
    """Return whether reference CacheGen-style shadow writes are enabled.

    Shadow writes are a diagnostic correctness path only. They require the
    runner-owned byte-page adapter and are intentionally disabled by default.
    """
    configured_value = os.getenv(CACHEGEN_INT8_SHADOW_WRITE_ENV_VAR)
    if configured_value is None:
        return False

    normalized = configured_value.strip().lower()
    if normalized in ("1", "true", "yes"):
        if not is_cachegen_int8_page_adapter_enabled():
            raise ValueError(
                f"{CACHEGEN_INT8_SHADOW_WRITE_ENV_VAR} requires "
                "VLLM_EXPERIMENTAL_CACHEGEN_INT8_PAGE_ADAPTER=1"
            )
        return True
    if normalized in ("0", "false", "no"):
        return False
    raise ValueError(
        f"{CACHEGEN_INT8_SHADOW_WRITE_ENV_VAR} must be one of "
        "'1', '0', 'true', 'false', 'yes', or 'no', "
        f"got {configured_value!r}"
    )
