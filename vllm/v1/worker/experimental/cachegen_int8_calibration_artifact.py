# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Loading and validation for fixed-scale CacheGen INT8 calibration artifacts."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path


CACHEGEN_INT8_CALIBRATION_ARTIFACT_ENV_VAR = (
    "VLLM_EXPERIMENTAL_CACHEGEN_INT8_CALIBRATION_ARTIFACT"
)
EXPECTED_QUANTIZER = "symmetric_int8_fixed_per_kv_head"
EXPECTED_QMAX = 127


@dataclass(frozen=True)
class CacheGenInt8CalibrationArtifact:
    """Validated fixed per-KV-head K/V scales for one model layer."""

    model: str
    layer_name: str
    key_scales_per_head: tuple[float, ...]
    value_scales_per_head: tuple[float, ...]
    tokens_seen: int
    prompt_count: int

    @property
    def num_kv_heads(self) -> int:
        return len(self.key_scales_per_head)


def get_cachegen_int8_calibration_artifact_path() -> Path | None:
    """Return configured artifact path, or None when unset."""
    value = os.getenv(CACHEGEN_INT8_CALIBRATION_ARTIFACT_ENV_VAR)
    if value is None or not value.strip():
        return None
    return Path(value.strip())


def load_cachegen_int8_calibration_artifact(
    *,
    expected_layer_name: str,
    expected_num_kv_heads: int,
) -> CacheGenInt8CalibrationArtifact:
    """Load one validated calibration artifact for selected-layer INT8 cache."""
    path = get_cachegen_int8_calibration_artifact_path()
    if path is None:
        raise ValueError(
            f"{CACHEGEN_INT8_CALIBRATION_ARTIFACT_ENV_VAR} must be set when "
            "experimental CacheGen INT8 attention is enabled"
        )
    if not path.is_file():
        raise ValueError(f"Calibration artifact does not exist: {path}")

    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Calibration artifact is not valid JSON: {path}"
        ) from exc

    if not isinstance(payload, dict):
        raise ValueError("Calibration artifact must contain a JSON object")

    if payload.get("quantizer") != EXPECTED_QUANTIZER:
        raise ValueError(
            f"Calibration artifact quantizer must be {EXPECTED_QUANTIZER!r}; "
            f"got {payload.get('quantizer')!r}"
        )
    if payload.get("qmax") != EXPECTED_QMAX:
        raise ValueError(
            f"Calibration artifact qmax must be {EXPECTED_QMAX}; "
            f"got {payload.get('qmax')!r}"
        )
    if payload.get("layer_name") != expected_layer_name:
        raise ValueError(
            "Calibration artifact layer mismatch: expected "
            f"{expected_layer_name!r}, got {payload.get('layer_name')!r}"
        )

    key_scales = payload.get("key_scales_per_head")
    value_scales = payload.get("value_scales_per_head")
    if not isinstance(key_scales, list) or not isinstance(value_scales, list):
        raise ValueError(
            "Calibration artifact must contain key_scales_per_head and "
            "value_scales_per_head arrays"
        )
    if (
        len(key_scales) != expected_num_kv_heads
        or len(value_scales) != expected_num_kv_heads
    ):
        raise ValueError(
            "Calibration artifact scale count must equal selected layer "
            f"num_kv_heads={expected_num_kv_heads}; got "
            f"{len(key_scales)} key and {len(value_scales)} value scales"
        )

    try:
        key_scale_values = tuple(float(value) for value in key_scales)
        value_scale_values = tuple(float(value) for value in value_scales)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Calibration artifact scales must be numeric values"
        ) from exc

    all_scales = key_scale_values + value_scale_values
    if any(not math.isfinite(value) or value <= 0 for value in all_scales):
        raise ValueError(
            "Calibration artifact scales must be finite positive values"
        )

    tokens_seen = payload.get("tokens_seen")
    prompt_count = payload.get("prompt_count")
    if not isinstance(tokens_seen, int) or tokens_seen <= 0:
        raise ValueError(
            "Calibration artifact tokens_seen must be a positive integer"
        )
    if not isinstance(prompt_count, int) or prompt_count <= 0:
        raise ValueError(
            "Calibration artifact prompt_count must be a positive integer"
        )

    model = payload.get("model")
    if not isinstance(model, str) or not model:
        raise ValueError(
            "Calibration artifact model must be a nonempty string"
        )

    return CacheGenInt8CalibrationArtifact(
        model=model,
        layer_name=expected_layer_name,
        key_scales_per_head=key_scale_values,
        value_scales_per_head=value_scale_values,
        tokens_seen=tokens_seen,
        prompt_count=prompt_count,
    )
