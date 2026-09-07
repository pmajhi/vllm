# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Configuration for CacheGen INT8 fixed-scale calibration."""

from __future__ import annotations

import os


CACHEGEN_INT8_CALIBRATION_LAYER_ENV_VAR = (
    "VLLM_EXPERIMENTAL_CACHEGEN_INT8_CALIBRATION_LAYER"
)


def get_cachegen_int8_calibration_layer() -> str | None:
    """Return the layer whose K/V ranges should be calibrated."""
    value = os.getenv(CACHEGEN_INT8_CALIBRATION_LAYER_ENV_VAR)
    if value is None or not value.strip():
        return None
    return value.strip()
