# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.v1.worker.experimental.hetero_kv_codec_registry import (
    DEFAULT_HETERO_KV_PAGE_BYTES,
)
from vllm.v1.worker.experimental.hetero_kv_page_config import (
    HETERO_KV_PAGE_BYTES_ENV_VAR,
    get_hetero_kv_page_bytes,
)


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
