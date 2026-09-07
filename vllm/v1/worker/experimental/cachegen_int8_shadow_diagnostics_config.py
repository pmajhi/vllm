# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration for CacheGen-style KV shadow-write diagnostics."""

import os

CACHEGEN_INT8_SHADOW_DIAGNOSTICS_ENV_VAR = (
    "VLLM_EXPERIMENTAL_CACHEGEN_INT8_SHADOW_DIAGNOSTICS"
)


def is_cachegen_int8_shadow_diagnostics_enabled() -> bool:
    """Return whether CacheGen shadow-write diagnostic counters are enabled."""
    configured_value = os.getenv(CACHEGEN_INT8_SHADOW_DIAGNOSTICS_ENV_VAR)
    if configured_value is None:
        return False

    normalized = configured_value.strip().lower()
    if normalized in ("1", "true", "yes"):
        return True
    if normalized in ("0", "false", "no"):
        return False

    raise ValueError(
        f"{CACHEGEN_INT8_SHADOW_DIAGNOSTICS_ENV_VAR} must be one of "
        "'1', '0', 'true', 'false', 'yes', or 'no'; "
        f"got {configured_value!r}"
    )
