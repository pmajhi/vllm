# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.v1.worker.experimental.cachegen_int8_shadow_diagnostics_config import (
    CACHEGEN_INT8_SHADOW_DIAGNOSTICS_ENV_VAR,
    is_cachegen_int8_shadow_diagnostics_enabled,
)


def test_diagnostics_default_to_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(
        CACHEGEN_INT8_SHADOW_DIAGNOSTICS_ENV_VAR,
        raising=False,
    )

    assert not is_cachegen_int8_shadow_diagnostics_enabled()


@pytest.mark.parametrize("value", ("1", "true", "TRUE", "yes"))
def test_diagnostics_accept_enabled_values(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    monkeypatch.setenv(CACHEGEN_INT8_SHADOW_DIAGNOSTICS_ENV_VAR, value)

    assert is_cachegen_int8_shadow_diagnostics_enabled()


@pytest.mark.parametrize("value", ("0", "false", "FALSE", "no"))
def test_diagnostics_accept_disabled_values(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    monkeypatch.setenv(CACHEGEN_INT8_SHADOW_DIAGNOSTICS_ENV_VAR, value)

    assert not is_cachegen_int8_shadow_diagnostics_enabled()


def test_diagnostics_reject_invalid_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        CACHEGEN_INT8_SHADOW_DIAGNOSTICS_ENV_VAR,
        "sometimes",
    )

    with pytest.raises(
        ValueError,
        match=CACHEGEN_INT8_SHADOW_DIAGNOSTICS_ENV_VAR,
    ):
        is_cachegen_int8_shadow_diagnostics_enabled()
