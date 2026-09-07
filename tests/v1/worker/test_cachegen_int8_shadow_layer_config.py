# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.v1.worker.experimental.cachegen_int8_shadow_layer_config import (
    CACHEGEN_INT8_SHADOW_LAYER_ENV_VAR,
    get_cachegen_int8_shadow_layer,
)


def clear_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "VLLM_EXPERIMENTAL_HETERO_KV_PAGE_PLANNING",
        "VLLM_EXPERIMENTAL_HETERO_KV_PAGE_POOL",
        "VLLM_EXPERIMENTAL_HETERO_KV_PAGE_POOL_PAGES",
        "VLLM_EXPERIMENTAL_CACHEGEN_INT8_PAGE_ADAPTER",
        "VLLM_EXPERIMENTAL_CACHEGEN_INT8_SHADOW_WRITE",
        CACHEGEN_INT8_SHADOW_LAYER_ENV_VAR,
    ):
        monkeypatch.delenv(name, raising=False)


def enable_shadow_write(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_EXPERIMENTAL_HETERO_KV_PAGE_PLANNING", "1")
    monkeypatch.setenv("VLLM_EXPERIMENTAL_HETERO_KV_PAGE_POOL", "1")
    monkeypatch.setenv("VLLM_EXPERIMENTAL_HETERO_KV_PAGE_POOL_PAGES", "2")
    monkeypatch.setenv("VLLM_EXPERIMENTAL_CACHEGEN_INT8_PAGE_ADAPTER", "1")
    monkeypatch.setenv("VLLM_EXPERIMENTAL_CACHEGEN_INT8_SHADOW_WRITE", "1")


def test_layer_is_none_when_shadow_write_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_environment(monkeypatch)

    assert get_cachegen_int8_shadow_layer() is None


def test_layer_is_required_when_shadow_write_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_environment(monkeypatch)
    enable_shadow_write(monkeypatch)

    with pytest.raises(ValueError, match="must be set"):
        get_cachegen_int8_shadow_layer()


def test_layer_name_is_trimmed(monkeypatch: pytest.MonkeyPatch) -> None:
    clear_environment(monkeypatch)
    enable_shadow_write(monkeypatch)
    monkeypatch.setenv(
        CACHEGEN_INT8_SHADOW_LAYER_ENV_VAR,
        " model.decoder.layers.0.self_attn ",
    )

    assert get_cachegen_int8_shadow_layer() == (
        "model.decoder.layers.0.self_attn"
    )
