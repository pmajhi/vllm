# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest

from vllm.v1.worker.gpu_model_runner import (
    _require_baseline_quantizer_ids,
)


def clear_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "VLLM_EXPERIMENTAL_HETERO_KV_PAGE_PLANNING",
        "VLLM_EXPERIMENTAL_HETERO_KV_PAGE_POOL",
        "VLLM_EXPERIMENTAL_HETERO_KV_PAGE_POOL_PAGES",
        "VLLM_EXPERIMENTAL_CACHEGEN_INT8_PAGE_ADAPTER",
        "VLLM_EXPERIMENTAL_CACHEGEN_INT8_SHADOW_WRITE",
    ):
        monkeypatch.delenv(name, raising=False)


def enable_shadow_write(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_EXPERIMENTAL_HETERO_KV_PAGE_PLANNING", "1")
    monkeypatch.setenv("VLLM_EXPERIMENTAL_HETERO_KV_PAGE_POOL", "1")
    monkeypatch.setenv("VLLM_EXPERIMENTAL_HETERO_KV_PAGE_POOL_PAGES", "1")
    monkeypatch.setenv("VLLM_EXPERIMENTAL_CACHEGEN_INT8_PAGE_ADAPTER", "1")
    monkeypatch.setenv("VLLM_EXPERIMENTAL_CACHEGEN_INT8_SHADOW_WRITE", "1")


def test_baseline_ids_are_always_allowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_environment(monkeypatch)

    _require_baseline_quantizer_ids(np.array([0, 0], dtype=np.int32))


def test_cachegen_shadow_ids_require_explicit_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_environment(monkeypatch)

    with pytest.raises(NotImplementedError, match="requires a quantized"):
        _require_baseline_quantizer_ids(np.array([2], dtype=np.int32))

    enable_shadow_write(monkeypatch)
    _require_baseline_quantizer_ids(np.array([2, 2], dtype=np.int32))


@pytest.mark.parametrize(
    "quantizer_ids",
    (
        [1],
        [3],
        [0, 2],
        [2, 3],
    ),
)
def test_all_other_nonbaseline_or_mixed_ids_remain_rejected(
    monkeypatch: pytest.MonkeyPatch,
    quantizer_ids: list[int],
) -> None:
    clear_environment(monkeypatch)
    enable_shadow_write(monkeypatch)

    with pytest.raises(NotImplementedError, match="requires a quantized"):
        _require_baseline_quantizer_ids(
            np.array(quantizer_ids, dtype=np.int32),
        )
