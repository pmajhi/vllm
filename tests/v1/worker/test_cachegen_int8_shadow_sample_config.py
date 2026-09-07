# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.v1.worker.experimental.cachegen_int8_shadow_diagnostics_config import (
    CACHEGEN_INT8_SHADOW_DIAGNOSTICS_ENV_VAR,
)
from vllm.v1.worker.experimental.cachegen_int8_shadow_sample_config import (
    CACHEGEN_INT8_SHADOW_SAMPLE_ENV_VAR,
    is_cachegen_int8_shadow_sample_enabled,
)


def clear_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(
        CACHEGEN_INT8_SHADOW_DIAGNOSTICS_ENV_VAR,
        raising=False,
    )
    monkeypatch.delenv(
        CACHEGEN_INT8_SHADOW_SAMPLE_ENV_VAR,
        raising=False,
    )


def test_sample_defaults_to_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_environment(monkeypatch)

    assert not is_cachegen_int8_shadow_sample_enabled()


@pytest.mark.parametrize("value", ("0", "false", "FALSE", "no"))
def test_sample_accepts_disabled_values(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    clear_environment(monkeypatch)
    monkeypatch.setenv(CACHEGEN_INT8_SHADOW_SAMPLE_ENV_VAR, value)

    assert not is_cachegen_int8_shadow_sample_enabled()


@pytest.mark.parametrize("value", ("1", "true", "TRUE", "yes"))
def test_sample_requires_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    clear_environment(monkeypatch)
    monkeypatch.setenv(CACHEGEN_INT8_SHADOW_SAMPLE_ENV_VAR, value)

    with pytest.raises(ValueError, match="requires"):
        is_cachegen_int8_shadow_sample_enabled()


@pytest.mark.parametrize("value", ("1", "true", "TRUE", "yes"))
def test_sample_accepts_enabled_values_with_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    clear_environment(monkeypatch)
    monkeypatch.setenv(CACHEGEN_INT8_SHADOW_DIAGNOSTICS_ENV_VAR, "1")
    monkeypatch.setenv(CACHEGEN_INT8_SHADOW_SAMPLE_ENV_VAR, value)

    assert is_cachegen_int8_shadow_sample_enabled()


def test_sample_rejects_invalid_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_environment(monkeypatch)
    monkeypatch.setenv(CACHEGEN_INT8_SHADOW_SAMPLE_ENV_VAR, "sometimes")

    with pytest.raises(ValueError, match=CACHEGEN_INT8_SHADOW_SAMPLE_ENV_VAR):
        is_cachegen_int8_shadow_sample_enabled()
