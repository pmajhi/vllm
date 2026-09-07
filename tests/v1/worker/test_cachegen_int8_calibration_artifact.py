# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from pathlib import Path

import pytest

from vllm.v1.worker.experimental.cachegen_int8_calibration_artifact import (
    CACHEGEN_INT8_CALIBRATION_ARTIFACT_ENV_VAR,
    EXPECTED_QMAX,
    EXPECTED_QUANTIZER,
    get_cachegen_int8_calibration_artifact_path,
    load_cachegen_int8_calibration_artifact,
)


LAYER_NAME = "model.layers.0.self_attn.attn"


def make_payload() -> dict[str, object]:
    return {
        "model": "/models/Qwen2.5-0.5B-Instruct",
        "layer_name": LAYER_NAME,
        "quantizer": EXPECTED_QUANTIZER,
        "qmax": EXPECTED_QMAX,
        "key_scales_per_head": [1.0, 2.0],
        "value_scales_per_head": [0.01, 0.02],
        "tokens_seen": 207,
        "prompt_count": 8,
    }


def write_artifact(tmp_path: Path, payload: dict[str, object]) -> Path:
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(payload))
    return path


def test_path_defaults_to_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(CACHEGEN_INT8_CALIBRATION_ARTIFACT_ENV_VAR, raising=False)

    assert get_cachegen_int8_calibration_artifact_path() is None


def test_loads_valid_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = write_artifact(tmp_path, make_payload())
    monkeypatch.setenv(CACHEGEN_INT8_CALIBRATION_ARTIFACT_ENV_VAR, str(path))

    artifact = load_cachegen_int8_calibration_artifact(
        expected_layer_name=LAYER_NAME,
        expected_num_kv_heads=2,
    )

    assert artifact.layer_name == LAYER_NAME
    assert artifact.key_scales_per_head == (1.0, 2.0)
    assert artifact.value_scales_per_head == (0.01, 0.02)
    assert artifact.tokens_seen == 207
    assert artifact.prompt_count == 8


def test_rejects_missing_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(CACHEGEN_INT8_CALIBRATION_ARTIFACT_ENV_VAR, raising=False)

    with pytest.raises(ValueError, match="must be set"):
        load_cachegen_int8_calibration_artifact(
            expected_layer_name=LAYER_NAME,
            expected_num_kv_heads=2,
        )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("quantizer", "wrong", "quantizer"),
        ("qmax", 7, "qmax"),
        ("layer_name", "model.layers.1.self_attn.attn", "layer mismatch"),
        ("key_scales_per_head", [1.0], "scale count"),
        ("value_scales_per_head", [0.01], "scale count"),
        ("key_scales_per_head", [0.0, 1.0], "finite positive"),
        ("value_scales_per_head", [0.01, -1.0], "finite positive"),
        ("tokens_seen", 0, "tokens_seen"),
        ("prompt_count", 0, "prompt_count"),
    ],
)
def test_rejects_invalid_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    field: str,
    value: object,
    match: str,
) -> None:
    payload = make_payload()
    payload[field] = value
    path = write_artifact(tmp_path, payload)
    monkeypatch.setenv(CACHEGEN_INT8_CALIBRATION_ARTIFACT_ENV_VAR, str(path))

    with pytest.raises(ValueError, match=match):
        load_cachegen_int8_calibration_artifact(
            expected_layer_name=LAYER_NAME,
            expected_num_kv_heads=2,
        )
