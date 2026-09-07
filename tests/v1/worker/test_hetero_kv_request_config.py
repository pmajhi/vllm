# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.v1.worker.experimental.hetero_kv_codec_registry import (
    HeteroKVCodecId,
)
from vllm.v1.worker.experimental.hetero_kv_request_config import (
    HETERO_KV_DEFAULT_QUANTIZER_ID_ENV_VAR,
    get_hetero_kv_default_quantizer_id,
)


PLANNING_ENV_VAR = "VLLM_EXPERIMENTAL_HETERO_KV_PAGE_PLANNING"


def clear_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(PLANNING_ENV_VAR, raising=False)
    monkeypatch.delenv(
        HETERO_KV_DEFAULT_QUANTIZER_ID_ENV_VAR,
        raising=False,
    )


def test_defaults_to_baseline_when_planning_is_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_environment(monkeypatch)

    assert get_hetero_kv_default_quantizer_id() == int(
        HeteroKVCodecId.BASELINE
    )


def test_disabled_planning_ignores_stale_quantizer_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_environment(monkeypatch)
    nonbaseline_codec = next(
        codec
        for codec in HeteroKVCodecId
        if codec is not HeteroKVCodecId.BASELINE
    )
    monkeypatch.setenv(
        HETERO_KV_DEFAULT_QUANTIZER_ID_ENV_VAR,
        str(int(nonbaseline_codec)),
    )

    assert get_hetero_kv_default_quantizer_id() == int(
        HeteroKVCodecId.BASELINE
    )


def test_enabled_planning_defaults_to_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_environment(monkeypatch)
    monkeypatch.setenv(PLANNING_ENV_VAR, "1")

    assert get_hetero_kv_default_quantizer_id() == int(
        HeteroKVCodecId.BASELINE
    )


def test_enabled_planning_accepts_registered_codec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_environment(monkeypatch)
    monkeypatch.setenv(PLANNING_ENV_VAR, "true")
    nonbaseline_codec = next(
        codec
        for codec in HeteroKVCodecId
        if codec is not HeteroKVCodecId.BASELINE
    )
    monkeypatch.setenv(
        HETERO_KV_DEFAULT_QUANTIZER_ID_ENV_VAR,
        str(int(nonbaseline_codec)),
    )

    assert get_hetero_kv_default_quantizer_id() == int(nonbaseline_codec)


def test_enabled_planning_rejects_noninteger_codec_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_environment(monkeypatch)
    monkeypatch.setenv(PLANNING_ENV_VAR, "1")
    monkeypatch.setenv(
        HETERO_KV_DEFAULT_QUANTIZER_ID_ENV_VAR,
        "not-a-number",
    )

    with pytest.raises(
        ValueError,
        match=HETERO_KV_DEFAULT_QUANTIZER_ID_ENV_VAR,
    ):
        get_hetero_kv_default_quantizer_id()


def test_enabled_planning_rejects_unknown_codec_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_environment(monkeypatch)
    monkeypatch.setenv(PLANNING_ENV_VAR, "1")
    monkeypatch.setenv(
        HETERO_KV_DEFAULT_QUANTIZER_ID_ENV_VAR,
        "999",
    )

    with pytest.raises(
        ValueError,
        match=HETERO_KV_DEFAULT_QUANTIZER_ID_ENV_VAR,
    ):
        get_hetero_kv_default_quantizer_id()
