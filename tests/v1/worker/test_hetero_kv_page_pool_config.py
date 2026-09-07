# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.v1.worker.experimental.hetero_kv_page_config import (
    HETERO_KV_PAGE_PLANNING_ENV_VAR,
)
from vllm.v1.worker.experimental.hetero_kv_page_pool_config import (
    HETERO_KV_PAGE_POOL_ENV_VAR,
    HETERO_KV_PAGE_POOL_PAGES_ENV_VAR,
    get_hetero_kv_page_pool_pages,
    is_hetero_kv_page_pool_enabled,
)


def clear_pool_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(HETERO_KV_PAGE_PLANNING_ENV_VAR, raising=False)
    monkeypatch.delenv(HETERO_KV_PAGE_POOL_ENV_VAR, raising=False)
    monkeypatch.delenv(HETERO_KV_PAGE_POOL_PAGES_ENV_VAR, raising=False)


def test_pool_defaults_to_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    clear_pool_environment(monkeypatch)

    assert not is_hetero_kv_page_pool_enabled()


def test_pool_requires_page_planning(monkeypatch: pytest.MonkeyPatch) -> None:
    clear_pool_environment(monkeypatch)
    monkeypatch.setenv(HETERO_KV_PAGE_POOL_ENV_VAR, "1")

    with pytest.raises(ValueError, match="requires"):
        is_hetero_kv_page_pool_enabled()


@pytest.mark.parametrize("value", ("1", "true", "yes"))
def test_pool_accepts_truthy_values_when_planning_enabled(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    clear_pool_environment(monkeypatch)
    monkeypatch.setenv(HETERO_KV_PAGE_PLANNING_ENV_VAR, "1")
    monkeypatch.setenv(HETERO_KV_PAGE_POOL_ENV_VAR, value)

    assert is_hetero_kv_page_pool_enabled()


@pytest.mark.parametrize("value", ("0", "false", "no"))
def test_pool_accepts_falsy_values(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    clear_pool_environment(monkeypatch)
    monkeypatch.setenv(HETERO_KV_PAGE_POOL_ENV_VAR, value)

    assert not is_hetero_kv_page_pool_enabled()


@pytest.mark.parametrize("value", ("", "2", "enabled"))
def test_pool_rejects_invalid_values(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    clear_pool_environment(monkeypatch)
    monkeypatch.setenv(HETERO_KV_PAGE_POOL_ENV_VAR, value)

    with pytest.raises(ValueError, match=HETERO_KV_PAGE_POOL_ENV_VAR):
        is_hetero_kv_page_pool_enabled()


def test_pool_page_count_is_required(monkeypatch: pytest.MonkeyPatch) -> None:
    clear_pool_environment(monkeypatch)

    with pytest.raises(ValueError, match="must be set"):
        get_hetero_kv_page_pool_pages()


def test_pool_page_count_is_parsed(monkeypatch: pytest.MonkeyPatch) -> None:
    clear_pool_environment(monkeypatch)
    monkeypatch.setenv(HETERO_KV_PAGE_POOL_PAGES_ENV_VAR, "256")

    assert get_hetero_kv_page_pool_pages() == 256


@pytest.mark.parametrize("value", ("0", "-1", "invalid"))
def test_pool_page_count_rejects_invalid_values(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    clear_pool_environment(monkeypatch)
    monkeypatch.setenv(HETERO_KV_PAGE_POOL_PAGES_ENV_VAR, value)

    with pytest.raises(ValueError, match=HETERO_KV_PAGE_POOL_PAGES_ENV_VAR):
        get_hetero_kv_page_pool_pages()
