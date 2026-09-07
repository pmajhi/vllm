# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration for all-layer CacheGen-style KV shadow storage."""

import os

from vllm.v1.worker.experimental.cachegen_int8_shadow_write_config import (
    is_cachegen_int8_shadow_write_enabled,
)

CACHEGEN_INT8_SHADOW_ALL_LAYERS_ENV_VAR = (
    "VLLM_EXPERIMENTAL_CACHEGEN_INT8_SHADOW_ALL_LAYERS"
)


def is_cachegen_int8_shadow_all_layers_enabled() -> bool:
    """Return whether CacheGen shadow storage is enabled for every layer."""
    configured_value = os.getenv(CACHEGEN_INT8_SHADOW_ALL_LAYERS_ENV_VAR)
    if configured_value is None:
        return False

    normalized = configured_value.strip().lower()
    if normalized in ("1", "true", "yes"):
        if not is_cachegen_int8_shadow_write_enabled():
            raise ValueError(
                f"{CACHEGEN_INT8_SHADOW_ALL_LAYERS_ENV_VAR} requires "
                "VLLM_EXPERIMENTAL_CACHEGEN_INT8_SHADOW_WRITE=1"
            )
        return True
    if normalized in ("0", "false", "no"):
        return False

    raise ValueError(
        f"{CACHEGEN_INT8_SHADOW_ALL_LAYERS_ENV_VAR} must be one of "
        "'1', '0', 'true', 'false', 'yes', or 'no'; "
        f"got {configured_value!r}"
    )
