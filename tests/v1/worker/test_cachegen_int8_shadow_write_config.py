# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.v1.worker.experimental.cachegen_int8_page_adapter_config import (
    CACHEGEN_INT8_PAGE_ADAPTER_ENV_VAR,
)
from vllm.v1.worker.experimental.cachegen_int8_shadow_write_config import (
    CACHEGEN_INT8_SHADOW_WRITE_ENV_VAR,
    is_cachegen_int8_shadow_write_enabled,
)
from vllm.v1.worker.experimental.hetero_kv_page_config import (
    HETERO_KV_PAGE_PLANNING_ENV_VAR,
)
from vllm.v1.worker.experimental.hetero_kv_page_pool_config import (
    HETERO_KV_PAGE_POOL_ENV_VAR,
    HETERO_KV_PAGE_POOL_PAGES_ENV_VAR,
)


def clear_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(CACHEGEN_INT8_SHADOW_WRITE_ENV_VAR, raising=False)
    monkeypatch.delenv(CACHEGEN_INT8_PAGE_ADAPTER_ENV_VAR, raising=False)
    monkeypatch.delenv(HETERO_KV_PAGE_PLANNING_ENV_VAR, raising=False)
    monkeypatch.delenv(HETERO_KV_PAGE_POOL_ENV_VAR, raising=False)
    monkeypatch.delenv(HETERO_KV_PAGE_POOL_PAGES_ENV_VAR, raising=False)


def enable_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(HETERO_KV_PAGE_PLANNING_ENV_VAR, "1")
    monkeypatch.setenv(HETERO_KV_PAGE_POOL_ENV_VAR, "1")
    monkeypatch.setenv(HETERO_KV_PAGE_POOL_PAGES_ENV_VAR, "1")
    monkeypatch.setenv(CACHEGEN_INT8_PAGE_ADAPTER_ENV_VAR, "1")


def test_shadow_write_defaults_to_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_environment(monkeypatch)

    assert not is_cachegen_int8_shadow_write_enabled()


def test_shadow_write_requires_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_environment(monkeypatch)
    monkeypatch.setenv(CACHEGEN_INT8_SHADOW_WRITE_ENV_VAR, "1")

    with pytest.raises(ValueError, match="requires"):
        is_cachegen_int8_shadow_write_enabled()


@pytest.mark.parametrize("value", ("1", "true", "yes"))
def test_shadow_write_accepts_truthy_values(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    clear_environment(monkeypatch)
    enable_dependencies(monkeypatch)
    monkeypatch.setenv(CACHEGEN_INT8_SHADOW_WRITE_ENV_VAR, value)

    assert is_cachegen_int8_shadow_write_enabled()


@pytest.mark.parametrize("value", ("0", "false", "no"))
def test_shadow_write_accepts_falsy_values(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    clear_environment(monkeypatch)
    monkeypatch.setenv(CACHEGEN_INT8_SHADOW_WRITE_ENV_VAR, value)

    assert not is_cachegen_int8_shadow_write_enabled()


@pytest.mark.parametrize("value", ("", "2", "enabled"))
def test_shadow_write_rejects_invalid_values(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    clear_environment(monkeypatch)
    monkeypatch.setenv(CACHEGEN_INT8_SHADOW_WRITE_ENV_VAR, value)

    with pytest.raises(ValueError, match=CACHEGEN_INT8_SHADOW_WRITE_ENV_VAR):
        is_cachegen_int8_shadow_write_enabled()
