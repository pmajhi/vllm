# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.v1.worker.experimental.cachegen_int8_shadow_all_layers_config import (
    CACHEGEN_INT8_SHADOW_ALL_LAYERS_ENV_VAR,
    is_cachegen_int8_shadow_all_layers_enabled,
)


def clear_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "VLLM_EXPERIMENTAL_HETERO_KV_PAGE_PLANNING",
        "VLLM_EXPERIMENTAL_HETERO_KV_PAGE_POOL",
        "VLLM_EXPERIMENTAL_HETERO_KV_PAGE_POOL_PAGES",
        "VLLM_EXPERIMENTAL_CACHEGEN_INT8_PAGE_ADAPTER",
        "VLLM_EXPERIMENTAL_CACHEGEN_INT8_SHADOW_WRITE",
        CACHEGEN_INT8_SHADOW_ALL_LAYERS_ENV_VAR,
    ):
        monkeypatch.delenv(name, raising=False)


def enable_shadow_write(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_EXPERIMENTAL_HETERO_KV_PAGE_PLANNING", "1")
    monkeypatch.setenv("VLLM_EXPERIMENTAL_HETERO_KV_PAGE_POOL", "1")
    monkeypatch.setenv("VLLM_EXPERIMENTAL_HETERO_KV_PAGE_POOL_PAGES", "2")
    monkeypatch.setenv("VLLM_EXPERIMENTAL_CACHEGEN_INT8_PAGE_ADAPTER", "1")
    monkeypatch.setenv("VLLM_EXPERIMENTAL_CACHEGEN_INT8_SHADOW_WRITE", "1")


def test_all_layers_defaults_to_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_environment(monkeypatch)

    assert not is_cachegen_int8_shadow_all_layers_enabled()


@pytest.mark.parametrize("value", ("0", "false", "FALSE", "no"))
def test_all_layers_accepts_disabled_values(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    clear_environment(monkeypatch)
    monkeypatch.setenv(CACHEGEN_INT8_SHADOW_ALL_LAYERS_ENV_VAR, value)

    assert not is_cachegen_int8_shadow_all_layers_enabled()


@pytest.mark.parametrize("value", ("1", "true", "TRUE", "yes"))
def test_all_layers_requires_shadow_write(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    clear_environment(monkeypatch)
    monkeypatch.setenv(CACHEGEN_INT8_SHADOW_ALL_LAYERS_ENV_VAR, value)

    with pytest.raises(ValueError, match="requires"):
        is_cachegen_int8_shadow_all_layers_enabled()


@pytest.mark.parametrize("value", ("1", "true", "TRUE", "yes"))
def test_all_layers_accepts_enabled_values_with_shadow_write(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    clear_environment(monkeypatch)
    enable_shadow_write(monkeypatch)
    monkeypatch.setenv(CACHEGEN_INT8_SHADOW_ALL_LAYERS_ENV_VAR, value)

    assert is_cachegen_int8_shadow_all_layers_enabled()


def test_all_layers_rejects_invalid_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_environment(monkeypatch)
    monkeypatch.setenv(
        CACHEGEN_INT8_SHADOW_ALL_LAYERS_ENV_VAR,
        "sometimes",
    )

    with pytest.raises(
        ValueError,
        match=CACHEGEN_INT8_SHADOW_ALL_LAYERS_ENV_VAR,
    ):
        is_cachegen_int8_shadow_all_layers_enabled()
