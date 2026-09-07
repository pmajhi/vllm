# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.v1.worker.experimental.hetero_kv_codec_registry import (
    DEFAULT_HETERO_KV_PAGE_BYTES,
)
from vllm.v1.worker.experimental.hetero_kv_page_config import (
    HETERO_KV_PAGE_BYTES_ENV_VAR,
    HETERO_KV_PAGE_PLANNING_ENV_VAR,
    get_hetero_kv_page_bytes,
    is_hetero_kv_page_planning_enabled,
)


def test_page_planning_defaults_to_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(HETERO_KV_PAGE_PLANNING_ENV_VAR, raising=False)

    assert not is_hetero_kv_page_planning_enabled()


@pytest.mark.parametrize("value", ("1", "true", "TRUE", "yes", " Yes "))
def test_page_planning_accepts_truthy_values(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    monkeypatch.setenv(HETERO_KV_PAGE_PLANNING_ENV_VAR, value)

    assert is_hetero_kv_page_planning_enabled()


@pytest.mark.parametrize("value", ("0", "false", "FALSE", "no", " No "))
def test_page_planning_accepts_falsy_values(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    monkeypatch.setenv(HETERO_KV_PAGE_PLANNING_ENV_VAR, value)

    assert not is_hetero_kv_page_planning_enabled()


@pytest.mark.parametrize("value", ("", "maybe", "2", "enabled"))
def test_page_planning_rejects_invalid_values(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    monkeypatch.setenv(HETERO_KV_PAGE_PLANNING_ENV_VAR, value)

    with pytest.raises(ValueError, match=HETERO_KV_PAGE_PLANNING_ENV_VAR):
        is_hetero_kv_page_planning_enabled()


def test_page_bytes_default_to_128kib(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(HETERO_KV_PAGE_BYTES_ENV_VAR, raising=False)

    assert get_hetero_kv_page_bytes() == DEFAULT_HETERO_KV_PAGE_BYTES
    assert get_hetero_kv_page_bytes() == 128 * 1024


def test_page_bytes_can_be_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(HETERO_KV_PAGE_BYTES_ENV_VAR, str(64 * 1024))

    assert get_hetero_kv_page_bytes() == 64 * 1024


@pytest.mark.parametrize("value", ("0", "-1", "not-an-integer", "128KiB"))
def test_page_bytes_reject_invalid_values(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    monkeypatch.setenv(HETERO_KV_PAGE_BYTES_ENV_VAR, value)

    with pytest.raises(ValueError, match=HETERO_KV_PAGE_BYTES_ENV_VAR):
        get_hetero_kv_page_bytes()
