# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.v1.worker.experimental.cachegen_int8_calibration_config import (
    CACHEGEN_INT8_CALIBRATION_LAYER_ENV_VAR,
    get_cachegen_int8_calibration_layer,
)


def test_calibration_defaults_to_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(CACHEGEN_INT8_CALIBRATION_LAYER_ENV_VAR, raising=False)

    assert get_cachegen_int8_calibration_layer() is None


@pytest.mark.parametrize("value", ("", "   "))
def test_calibration_ignores_empty_layer_name(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    monkeypatch.setenv(CACHEGEN_INT8_CALIBRATION_LAYER_ENV_VAR, value)

    assert get_cachegen_int8_calibration_layer() is None


def test_calibration_returns_layer_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        CACHEGEN_INT8_CALIBRATION_LAYER_ENV_VAR,
        "model.layers.0.self_attn.attn",
    )

    assert get_cachegen_int8_calibration_layer() == (
        "model.layers.0.self_attn.attn"
    )


def test_flash_attention_collects_selected_layer_calibration_ranges() -> None:
    from pathlib import Path

    source = Path("vllm/v1/attention/backends/flash_attn.py").read_text()

    assert "get_cachegen_int8_calibration_layer" in source
    assert 'calibration_layer == layer.layer_name' in source
    assert "_cachegen_int8_calibration_stats" in source
    assert "calibration_num_tokens = attn_metadata.num_actual_tokens" in source
    assert "key_abs_max = key[:calibration_num_tokens].float().abs().amax(" in source
    assert "value_abs_max = value[" in source
    assert ":calibration_num_tokens" in source
