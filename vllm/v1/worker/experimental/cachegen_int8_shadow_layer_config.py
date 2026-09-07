# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Layer selection for reference CacheGen-style KV shadow writes."""

import os

from vllm.v1.worker.experimental.cachegen_int8_shadow_write_config import (
    is_cachegen_int8_shadow_write_enabled,
)

CACHEGEN_INT8_SHADOW_LAYER_ENV_VAR = (
    "VLLM_EXPERIMENTAL_CACHEGEN_INT8_SHADOW_LAYER"
)


def get_cachegen_int8_shadow_layer() -> str | None:
    """Return the selected layer name for diagnostic CacheGen shadow writes.

    A layer is required only when shadow writing itself is enabled. Keeping the
    shadow path to one layer avoids aliasing all model layers into one
    experimental byte-page pool before per-layer byte-pool allocation exists.
    """
    layer_name = os.getenv(CACHEGEN_INT8_SHADOW_LAYER_ENV_VAR)
    if not is_cachegen_int8_shadow_write_enabled():
        return None

    if layer_name is None or not layer_name.strip():
        raise ValueError(
            f"{CACHEGEN_INT8_SHADOW_LAYER_ENV_VAR} must be set when "
            "CacheGen-style KV shadow writing is enabled"
        )
    return layer_name.strip()
