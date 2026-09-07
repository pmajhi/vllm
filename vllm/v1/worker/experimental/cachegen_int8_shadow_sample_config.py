# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration for one-sample CacheGen shadow round-trip diagnostics."""

import os

from vllm.v1.worker.experimental.cachegen_int8_shadow_diagnostics_config import (
    is_cachegen_int8_shadow_diagnostics_enabled,
)

CACHEGEN_INT8_SHADOW_SAMPLE_ENV_VAR = (
    "VLLM_EXPERIMENTAL_CACHEGEN_INT8_SHADOW_SAMPLE"
)


def is_cachegen_int8_shadow_sample_enabled() -> bool:
    """Return whether one CacheGen shadow round-trip sample is enabled."""
    configured_value = os.getenv(CACHEGEN_INT8_SHADOW_SAMPLE_ENV_VAR)
    if configured_value is None:
        return False

    normalized = configured_value.strip().lower()
    if normalized in ("1", "true", "yes"):
        if not is_cachegen_int8_shadow_diagnostics_enabled():
            raise ValueError(
                f"{CACHEGEN_INT8_SHADOW_SAMPLE_ENV_VAR} requires "
                "VLLM_EXPERIMENTAL_CACHEGEN_INT8_SHADOW_DIAGNOSTICS=1"
            )
        return True
    if normalized in ("0", "false", "no"):
        return False

    raise ValueError(
        f"{CACHEGEN_INT8_SHADOW_SAMPLE_ENV_VAR} must be one of "
        "'1', '0', 'true', 'false', 'yes', or 'no'; "
        f"got {configured_value!r}"
    )
