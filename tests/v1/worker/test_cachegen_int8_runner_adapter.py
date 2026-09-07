# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.worker.gpu_model_runner import GPUModelRunner


def make_runner() -> GPUModelRunner:
    runner = object.__new__(GPUModelRunner)
    runner.hetero_kv_page_pool = torch.zeros(
        (2, 128 * 1024),
        dtype=torch.uint8,
    )
    runner.cachegen_int8_page_adapter = None
    return runner


def make_attention_config(
    geometries: list[tuple[int, int]],
) -> SimpleNamespace:
    return SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(
                kv_cache_spec=FullAttentionSpec(
                    block_size=16,
                    num_kv_heads=num_kv_heads,
                    head_size=head_size,
                    dtype=torch.bfloat16,
                    use_mla=False,
                )
            )
            for num_kv_heads, head_size in geometries
        ]
    )


def clear_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(
        "VLLM_EXPERIMENTAL_HETERO_KV_PAGE_PLANNING",
        raising=False,
    )
    monkeypatch.delenv(
        "VLLM_EXPERIMENTAL_HETERO_KV_PAGE_POOL",
        raising=False,
    )
    monkeypatch.delenv(
        "VLLM_EXPERIMENTAL_HETERO_KV_PAGE_POOL_PAGES",
        raising=False,
    )
    monkeypatch.delenv(
        "VLLM_EXPERIMENTAL_CACHEGEN_INT8_PAGE_ADAPTER",
        raising=False,
    )


def enable_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_EXPERIMENTAL_HETERO_KV_PAGE_PLANNING", "1")
    monkeypatch.setenv("VLLM_EXPERIMENTAL_HETERO_KV_PAGE_POOL", "1")
    monkeypatch.setenv("VLLM_EXPERIMENTAL_HETERO_KV_PAGE_POOL_PAGES", "2")
    monkeypatch.setenv("VLLM_EXPERIMENTAL_CACHEGEN_INT8_PAGE_ADAPTER", "1")


def test_runner_adapter_is_absent_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_environment(monkeypatch)
    runner = make_runner()

    runner._initialize_cachegen_int8_page_adapter(
        make_attention_config([(8, 128)]),
    )

    assert runner.cachegen_int8_page_adapter is None


def test_runner_binds_adapter_to_real_byte_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_environment(monkeypatch)
    enable_adapter(monkeypatch)
    runner = make_runner()

    runner._initialize_cachegen_int8_page_adapter(
        make_attention_config([(8, 128)]),
    )

    adapter = runner.cachegen_int8_page_adapter
    assert adapter is not None
    assert adapter.page_pool is runner.hetero_kv_page_pool
    assert adapter.num_pages == 2
    assert adapter.page_bytes == 128 * 1024
    assert adapter.layout.num_kv_heads == 8
    assert adapter.layout.head_size == 128


def test_runner_rejects_missing_byte_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_environment(monkeypatch)
    enable_adapter(monkeypatch)
    runner = make_runner()
    runner.hetero_kv_page_pool = None

    with pytest.raises(AssertionError, match="requires byte-page storage"):
        runner._initialize_cachegen_int8_page_adapter(
            make_attention_config([(8, 128)]),
        )


def test_runner_rejects_mismatched_attention_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_environment(monkeypatch)
    enable_adapter(monkeypatch)
    runner = make_runner()

    with pytest.raises(ValueError, match="share num_kv_heads and head_size"):
        runner._initialize_cachegen_int8_page_adapter(
            make_attention_config([(8, 128), (16, 128)]),
        )
