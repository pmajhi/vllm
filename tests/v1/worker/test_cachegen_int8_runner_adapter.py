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
    runner.cachegen_int8_layer_page_pool = None
    runner.cachegen_int8_page_adapters = {}
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
    monkeypatch.delenv(
        "VLLM_EXPERIMENTAL_CACHEGEN_INT8_SHADOW_DIAGNOSTICS",
        raising=False,
    )
    monkeypatch.delenv(
        "VLLM_EXPERIMENTAL_CACHEGEN_INT8_SHADOW_ALL_LAYERS",
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


def test_runner_binds_diagnostics_to_byte_pool_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_environment(monkeypatch)
    enable_adapter(monkeypatch)
    monkeypatch.setenv(
        "VLLM_EXPERIMENTAL_CACHEGEN_INT8_SHADOW_DIAGNOSTICS",
        "1",
    )
    runner = make_runner()

    runner._initialize_cachegen_int8_page_adapter(
        make_attention_config([(8, 128)]),
    )

    assert runner.cachegen_int8_shadow_callback_calls is not None
    assert runner.cachegen_int8_shadow_tokens_written is not None
    assert (
        runner.cachegen_int8_shadow_callback_calls.device
        == runner.hetero_kv_page_pool.device
    )
    assert (
        runner.cachegen_int8_shadow_tokens_written.device
        == runner.hetero_kv_page_pool.device
    )


def test_runner_binds_layer_isolated_adapters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_environment(monkeypatch)
    enable_adapter(monkeypatch)
    monkeypatch.setenv(
        "VLLM_EXPERIMENTAL_CACHEGEN_INT8_SHADOW_WRITE",
        "1",
    )
    monkeypatch.setenv(
        "VLLM_EXPERIMENTAL_CACHEGEN_INT8_SHADOW_ALL_LAYERS",
        "1",
    )

    runner = make_runner()
    layer_names = ["layers.0.attn", "layers.1.attn"]
    from vllm.v1.worker.experimental.cachegen_int8_layer_page_pool import (
        CacheGenInt8LayerPagePool,
    )

    runner.cachegen_int8_layer_page_pool = CacheGenInt8LayerPagePool.allocate(
        layer_names=layer_names,
        pages_per_layer=2,
        page_bytes=128 * 1024,
        device=torch.device("cpu"),
    )
    runner.hetero_kv_page_pool = runner.cachegen_int8_layer_page_pool.page_pool

    runner._initialize_cachegen_int8_page_adapter(
        make_attention_config([(8, 128)]),
    )

    assert runner.cachegen_int8_page_adapter is None
    assert set(runner.cachegen_int8_page_adapters) == set(layer_names)

    first = runner.cachegen_int8_page_adapters[layer_names[0]]
    second = runner.cachegen_int8_page_adapters[layer_names[1]]
    assert first.page_pool.shape == (2, 128 * 1024)
    assert second.page_pool.shape == (2, 128 * 1024)
    assert first.page_pool.data_ptr() != second.page_pool.data_ptr()


def test_layer_isolated_adapters_do_not_alias_logical_pages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_environment(monkeypatch)
    enable_adapter(monkeypatch)
    monkeypatch.setenv(
        "VLLM_EXPERIMENTAL_CACHEGEN_INT8_SHADOW_WRITE",
        "1",
    )
    monkeypatch.setenv(
        "VLLM_EXPERIMENTAL_CACHEGEN_INT8_SHADOW_ALL_LAYERS",
        "1",
    )

    runner = make_runner()
    layer_names = ["layers.0.attn", "layers.1.attn"]
    from vllm.v1.worker.experimental.cachegen_int8_layer_page_pool import (
        CacheGenInt8LayerPagePool,
    )

    runner.cachegen_int8_layer_page_pool = CacheGenInt8LayerPagePool.allocate(
        layer_names=layer_names,
        pages_per_layer=2,
        page_bytes=128 * 1024,
        device=torch.device("cpu"),
    )
    runner.hetero_kv_page_pool = runner.cachegen_int8_layer_page_pool.page_pool

    runner._initialize_cachegen_int8_page_adapter(
        make_attention_config([(8, 128)]),
    )

    first = runner.cachegen_int8_page_adapters[layer_names[0]]
    second = runner.cachegen_int8_page_adapters[layer_names[1]]

    first_key = torch.full((8, 128), 0.25, dtype=torch.float32)
    first_value = torch.full((8, 128), -0.50, dtype=torch.float32)
    second_key = torch.full((8, 128), 1.50, dtype=torch.float32)
    second_value = torch.full((8, 128), -1.25, dtype=torch.float32)

    first.write_token(
        page_id=0,
        page_offset=0,
        key=first_key,
        value=first_value,
    )
    second.write_token(
        page_id=0,
        page_offset=0,
        key=second_key,
        value=second_value,
    )

    decoded_first_key, decoded_first_value = first.read_token(
        page_id=0,
        page_offset=0,
    )
    decoded_second_key, decoded_second_value = second.read_token(
        page_id=0,
        page_offset=0,
    )

    assert torch.allclose(decoded_first_key, first_key, atol=0.01, rtol=0)
    assert torch.allclose(decoded_first_value, first_value, atol=0.01, rtol=0)
    assert torch.allclose(decoded_second_key, second_key, atol=0.01, rtol=0)
    assert torch.allclose(decoded_second_value, second_value, atol=0.01, rtol=0)


def test_all_layers_requires_layer_isolated_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_environment(monkeypatch)
    enable_adapter(monkeypatch)
    monkeypatch.setenv(
        "VLLM_EXPERIMENTAL_CACHEGEN_INT8_SHADOW_WRITE",
        "1",
    )
    monkeypatch.setenv(
        "VLLM_EXPERIMENTAL_CACHEGEN_INT8_SHADOW_ALL_LAYERS",
        "1",
    )

    runner = make_runner()

    with pytest.raises(
        AssertionError,
        match="requires layer-isolated byte-page storage",
    ):
        runner._initialize_cachegen_int8_page_adapter(
            make_attention_config([(8, 128)]),
        )


def test_runner_resolves_layer_local_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_environment(monkeypatch)
    enable_adapter(monkeypatch)
    monkeypatch.setenv(
        "VLLM_EXPERIMENTAL_CACHEGEN_INT8_SHADOW_WRITE",
        "1",
    )
    monkeypatch.setenv(
        "VLLM_EXPERIMENTAL_CACHEGEN_INT8_SHADOW_ALL_LAYERS",
        "1",
    )

    runner = make_runner()
    from vllm.v1.worker.experimental.cachegen_int8_layer_page_pool import (
        CacheGenInt8LayerPagePool,
    )

    layer_names = ["layers.0.attn", "layers.1.attn"]
    runner.cachegen_int8_layer_page_pool = CacheGenInt8LayerPagePool.allocate(
        layer_names=layer_names,
        pages_per_layer=2,
        page_bytes=128 * 1024,
        device=torch.device("cpu"),
    )
    runner.hetero_kv_page_pool = runner.cachegen_int8_layer_page_pool.page_pool
    runner._initialize_cachegen_int8_page_adapter(
        make_attention_config([(8, 128)]),
    )

    assert (
        runner.get_cachegen_int8_page_adapter(
            layer_name="layers.0.attn"
        )
        is runner.cachegen_int8_page_adapters["layers.0.attn"]
    )
    assert (
        runner.get_cachegen_int8_page_adapter(
            layer_name="layers.1.attn"
        )
        is runner.cachegen_int8_page_adapters["layers.1.attn"]
    )

    with pytest.raises(ValueError, match="No CacheGen-style byte-page adapter"):
        runner.get_cachegen_int8_page_adapter(layer_name="layers.2.attn")
