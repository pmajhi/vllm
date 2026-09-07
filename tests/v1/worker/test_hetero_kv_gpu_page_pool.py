# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.worker.gpu_model_runner import GPUModelRunner


def make_runner() -> GPUModelRunner:
    runner = object.__new__(GPUModelRunner)
    runner.device = torch.device("cpu")
    runner.hetero_kv_page_pool = None
    return runner


def clear_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VLLM_EXPERIMENTAL_HETERO_KV_PAGE_PLANNING", raising=False)
    monkeypatch.delenv("VLLM_EXPERIMENTAL_HETERO_KV_PAGE_POOL", raising=False)
    monkeypatch.delenv(
        "VLLM_EXPERIMENTAL_HETERO_KV_PAGE_POOL_PAGES",
        raising=False,
    )
    monkeypatch.delenv("VLLM_EXPERIMENTAL_HETERO_KV_PAGE_BYTES", raising=False)


def test_gpu_page_pool_is_absent_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_environment(monkeypatch)
    runner = make_runner()

    runner._initialize_hetero_kv_page_pool()

    assert runner.hetero_kv_page_pool is None


def test_gpu_page_pool_has_expected_byte_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_environment(monkeypatch)
    monkeypatch.setenv("VLLM_EXPERIMENTAL_HETERO_KV_PAGE_PLANNING", "1")
    monkeypatch.setenv("VLLM_EXPERIMENTAL_HETERO_KV_PAGE_POOL", "1")
    monkeypatch.setenv("VLLM_EXPERIMENTAL_HETERO_KV_PAGE_POOL_PAGES", "3")
    monkeypatch.setenv("VLLM_EXPERIMENTAL_HETERO_KV_PAGE_BYTES", str(64 * 1024))
    runner = make_runner()

    runner._initialize_hetero_kv_page_pool()

    assert runner.hetero_kv_page_pool is not None
    assert runner.hetero_kv_page_pool.shape == (3, 64 * 1024)
    assert runner.hetero_kv_page_pool.dtype is torch.uint8
    assert runner.hetero_kv_page_pool.device.type == "cpu"
    assert torch.count_nonzero(runner.hetero_kv_page_pool) == 0


def test_gpu_page_pool_is_cleared_after_disabling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_environment(monkeypatch)
    runner = make_runner()
    runner.hetero_kv_page_pool = torch.ones(
        (1, 128 * 1024),
        dtype=torch.uint8,
    )

    runner._initialize_hetero_kv_page_pool()

    assert runner.hetero_kv_page_pool is None


def test_gpu_page_pool_requires_explicit_page_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_environment(monkeypatch)
    monkeypatch.setenv("VLLM_EXPERIMENTAL_HETERO_KV_PAGE_PLANNING", "1")
    monkeypatch.setenv("VLLM_EXPERIMENTAL_HETERO_KV_PAGE_POOL", "1")
    runner = make_runner()

    with pytest.raises(ValueError, match="PAGE_POOL_PAGES"):
        runner._initialize_hetero_kv_page_pool()
