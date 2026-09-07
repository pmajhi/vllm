# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.v1.worker.experimental.cachegen_int8_attention_config import (
    get_cachegen_int8_attention_layer,
    get_cachegen_int8_attention_max_pages,
    get_cachegen_int8_attention_mode,
    is_cachegen_int8_attention_enabled,
)


def test_attention_defaults_disabled_and_validate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("VLLM_EXPERIMENTAL_CACHEGEN_INT8_ATTENTION",
                       raising=False)
    monkeypatch.delenv("VLLM_EXPERIMENTAL_CACHEGEN_INT8_ATTENTION_LAYER",
                       raising=False)
    monkeypatch.delenv("VLLM_EXPERIMENTAL_CACHEGEN_INT8_ATTENTION_MODE",
                       raising=False)

    assert not is_cachegen_int8_attention_enabled()
    assert get_cachegen_int8_attention_layer() is None
    assert get_cachegen_int8_attention_max_pages() is None
    assert get_cachegen_int8_attention_mode() == "validate"


def test_attention_enabled_uses_default_page_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_EXPERIMENTAL_CACHEGEN_INT8_ATTENTION", "1")

    assert is_cachegen_int8_attention_enabled()
    assert get_cachegen_int8_attention_max_pages() == 256


@pytest.mark.parametrize("mode", ["validate", "int8", "VALIDATE", " INT8 "])
def test_attention_mode_accepts_valid_values(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    monkeypatch.setenv("VLLM_EXPERIMENTAL_CACHEGEN_INT8_ATTENTION_MODE", mode)

    assert get_cachegen_int8_attention_mode() == mode.strip().lower()


def test_attention_mode_rejects_unknown_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "VLLM_EXPERIMENTAL_CACHEGEN_INT8_ATTENTION_MODE",
        "fp8",
    )

    with pytest.raises(ValueError, match="must be one of"):
        get_cachegen_int8_attention_mode()
