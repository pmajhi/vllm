# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.worker.gpu_model_runner import GPUModelRunner


def make_runner() -> GPUModelRunner:
    runner = object.__new__(GPUModelRunner)
    runner.device = torch.device("cpu")
    runner.hetero_kv_page_pool = None
    runner.cachegen_int8_layer_page_pool = None
    runner.cachegen_int8_pages_per_layer = None
    return runner


def make_attention_config(layer_names: list[str]) -> SimpleNamespace:
    return SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(
                layer_names=layer_names,
                kv_cache_spec=FullAttentionSpec(
                    block_size=16,
                    num_kv_heads=8,
                    head_size=128,
                    dtype=torch.bfloat16,
                    use_mla=False,
                ),
            )
        ]
    )


def clear_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "VLLM_EXPERIMENTAL_HETERO_KV_PAGE_PLANNING",
        "VLLM_EXPERIMENTAL_HETERO_KV_PAGE_POOL",
        "VLLM_EXPERIMENTAL_HETERO_KV_PAGE_POOL_PAGES",
        "VLLM_EXPERIMENTAL_HETERO_KV_PAGE_BYTES",
        "VLLM_EXPERIMENTAL_CACHEGEN_INT8_PAGE_ADAPTER",
        "VLLM_EXPERIMENTAL_CACHEGEN_INT8_SHADOW_WRITE",
        "VLLM_EXPERIMENTAL_CACHEGEN_INT8_SHADOW_ALL_LAYERS",
    ):
        monkeypatch.delenv(name, raising=False)


def enable_page_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_EXPERIMENTAL_HETERO_KV_PAGE_PLANNING", "1")
    monkeypatch.setenv("VLLM_EXPERIMENTAL_HETERO_KV_PAGE_POOL", "1")
    monkeypatch.setenv("VLLM_EXPERIMENTAL_HETERO_KV_PAGE_POOL_PAGES", "3")
    monkeypatch.setenv("VLLM_EXPERIMENTAL_HETERO_KV_PAGE_BYTES", str(64 * 1024))


def enable_all_layers(monkeypatch: pytest.MonkeyPatch) -> None:
    enable_page_pool(monkeypatch)
    monkeypatch.setenv("VLLM_EXPERIMENTAL_CACHEGEN_INT8_PAGE_ADAPTER", "1")
    monkeypatch.setenv("VLLM_EXPERIMENTAL_CACHEGEN_INT8_SHADOW_WRITE", "1")
    monkeypatch.setenv(
        "VLLM_EXPERIMENTAL_CACHEGEN_INT8_SHADOW_ALL_LAYERS",
        "1",
    )


def test_gpu_page_pool_is_absent_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_environment(monkeypatch)
    runner = make_runner()

    runner._initialize_hetero_kv_page_pool()

    assert runner.hetero_kv_page_pool is None
    assert runner.cachegen_int8_layer_page_pool is None


def test_gpu_page_pool_has_expected_byte_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_environment(monkeypatch)
    enable_page_pool(monkeypatch)
    runner = make_runner()

    runner._initialize_hetero_kv_page_pool()

    assert runner.hetero_kv_page_pool is not None
    assert runner.hetero_kv_page_pool.shape == (3, 64 * 1024)
    assert runner.hetero_kv_page_pool.dtype is torch.uint8
    assert runner.hetero_kv_page_pool.device.type == "cpu"
    assert torch.count_nonzero(runner.hetero_kv_page_pool) == 0
    assert runner.cachegen_int8_layer_page_pool is None


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
    assert runner.cachegen_int8_layer_page_pool is None


def test_gpu_page_pool_requires_explicit_page_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_environment(monkeypatch)
    monkeypatch.setenv("VLLM_EXPERIMENTAL_HETERO_KV_PAGE_PLANNING", "1")
    monkeypatch.setenv("VLLM_EXPERIMENTAL_HETERO_KV_PAGE_POOL", "1")
    runner = make_runner()

    with pytest.raises(ValueError, match="PAGE_POOL_PAGES"):
        runner._initialize_hetero_kv_page_pool()


def test_all_layers_pool_uses_disjoint_layer_slices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_environment(monkeypatch)
    enable_all_layers(monkeypatch)
    runner = make_runner()
    runner.kv_cache_config = make_attention_config(
        ["layers.0.attn", "layers.1.attn"]
    )

    runner._initialize_hetero_kv_page_pool()

    layer_pool = runner.cachegen_int8_layer_page_pool
    assert layer_pool is not None
    assert runner.hetero_kv_page_pool is layer_pool.page_pool
    assert runner.cachegen_int8_pages_per_layer == 3
    assert layer_pool.total_pages == 6
    assert layer_pool.layer_page_pool("layers.0.attn").shape == (3, 64 * 1024)
    assert layer_pool.layer_page_pool("layers.1.attn").shape == (3, 64 * 1024)

    first = layer_pool.layer_page_pool("layers.0.attn")
    second = layer_pool.layer_page_pool("layers.1.attn")
    first[0, 0] = 17
    second[0, 0] = 29

    assert first[0, 0].item() == 17
    assert second[0, 0].item() == 29
