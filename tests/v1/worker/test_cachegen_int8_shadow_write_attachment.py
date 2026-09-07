# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import dataclasses
from types import SimpleNamespace

import pytest
import torch

from vllm.config import CUDAGraphMode
from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata
from vllm.v1.worker.experimental.cachegen_int8_byte_page_adapter import (
    CacheGenInt8FixedBytePageAdapter,
)
from vllm.v1.worker.gpu_model_runner import GPUModelRunner


PAGE_BYTES = 128 * 1024
NUM_KV_HEADS = 8
HEAD_SIZE = 128
SELECTED_LAYER = "model.decoder.layers.0.self_attn.attn"


def make_metadata(quantizer_id: int = 2) -> FlashAttentionMetadata:
    return FlashAttentionMetadata(
        num_actual_tokens=2,
        max_query_len=2,
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        max_seq_len=2,
        seq_lens=torch.tensor([2], dtype=torch.int32),
        block_table=torch.tensor([[0]], dtype=torch.int32),
        slot_mapping=torch.tensor([0, 1], dtype=torch.int64),
        use_cascade=False,
        common_prefix_len=0,
        cu_prefix_query_lens=None,
        prefix_kv_lens=None,
        suffix_kv_lens=None,
        quantizer_id=torch.tensor([quantizer_id], dtype=torch.int32),
        tokens_per_page=torch.tensor([1], dtype=torch.int32),
        quantized_page_ids=torch.tensor([0, 1], dtype=torch.int32),
        quantized_page_offsets=torch.tensor([0, 0], dtype=torch.int32),
    )


def make_runner(mode: CUDAGraphMode = CUDAGraphMode.NONE) -> GPUModelRunner:
    runner = object.__new__(GPUModelRunner)
    runner.compilation_config = SimpleNamespace(cudagraph_mode=mode)
    runner.cachegen_int8_page_adapter = CacheGenInt8FixedBytePageAdapter(
        page_pool=torch.zeros((2, PAGE_BYTES), dtype=torch.uint8),
        num_kv_heads=NUM_KV_HEADS,
        head_size=HEAD_SIZE,
    )
    return runner


def clear_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "VLLM_EXPERIMENTAL_HETERO_KV_PAGE_PLANNING",
        "VLLM_EXPERIMENTAL_HETERO_KV_PAGE_POOL",
        "VLLM_EXPERIMENTAL_HETERO_KV_PAGE_POOL_PAGES",
        "VLLM_EXPERIMENTAL_CACHEGEN_INT8_PAGE_ADAPTER",
        "VLLM_EXPERIMENTAL_CACHEGEN_INT8_SHADOW_WRITE",
        "VLLM_EXPERIMENTAL_CACHEGEN_INT8_SHADOW_LAYER",
        "VLLM_EXPERIMENTAL_CACHEGEN_INT8_SHADOW_DIAGNOSTICS",
        "VLLM_EXPERIMENTAL_CACHEGEN_INT8_SHADOW_SAMPLE",
        "VLLM_EXPERIMENTAL_CACHEGEN_INT8_SHADOW_ALL_LAYERS",
    ):
        monkeypatch.delenv(name, raising=False)


def enable_shadow_write(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_EXPERIMENTAL_HETERO_KV_PAGE_PLANNING", "1")
    monkeypatch.setenv("VLLM_EXPERIMENTAL_HETERO_KV_PAGE_POOL", "1")
    monkeypatch.setenv("VLLM_EXPERIMENTAL_HETERO_KV_PAGE_POOL_PAGES", "2")
    monkeypatch.setenv("VLLM_EXPERIMENTAL_CACHEGEN_INT8_PAGE_ADAPTER", "1")
    monkeypatch.setenv("VLLM_EXPERIMENTAL_CACHEGEN_INT8_SHADOW_WRITE", "1")
    monkeypatch.setenv(
        "VLLM_EXPERIMENTAL_CACHEGEN_INT8_SHADOW_LAYER",
        SELECTED_LAYER,
    )


def test_attachment_is_noop_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_environment(monkeypatch)
    runner = make_runner()
    metadata = make_metadata()

    runner._attach_cachegen_int8_shadow_write(
        layer_name=SELECTED_LAYER,
        attn_metadata=metadata,
    )

    assert metadata.cachegen_int8_shadow_write is None


def test_attachment_is_noop_for_nonselected_layer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_environment(monkeypatch)
    enable_shadow_write(monkeypatch)
    runner = make_runner()
    metadata = make_metadata()

    runner._attach_cachegen_int8_shadow_write(
        layer_name="model.decoder.layers.1.self_attn.attn",
        attn_metadata=metadata,
    )

    assert metadata.cachegen_int8_shadow_write is None


def test_attachment_writes_mapped_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_environment(monkeypatch)
    enable_shadow_write(monkeypatch)
    runner = make_runner()
    metadata = make_metadata()

    runner._attach_cachegen_int8_shadow_write(
        layer_name=SELECTED_LAYER,
        attn_metadata=metadata,
    )

    assert metadata.cachegen_int8_shadow_write is not None
    keys = torch.randn((2, NUM_KV_HEADS, HEAD_SIZE), dtype=torch.float32)
    values = torch.randn_like(keys)
    metadata.cachegen_int8_shadow_write(keys, values, metadata)

    for index in range(2):
        decoded_key, decoded_value = (
            runner.cachegen_int8_page_adapter.read_token(
                page_id=index,
                page_offset=0,
            )
        )
        assert torch.allclose(decoded_key, keys[index], atol=0.04, rtol=0.02)
        assert torch.allclose(
            decoded_value,
            values[index],
            atol=0.04,
            rtol=0.02,
        )


def test_attachment_rejects_cuda_graph_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_environment(monkeypatch)
    enable_shadow_write(monkeypatch)
    runner = make_runner(CUDAGraphMode.PIECEWISE)

    with pytest.raises(ValueError, match="cudagraph_mode=NONE"):
        runner._attach_cachegen_int8_shadow_write(
            layer_name=SELECTED_LAYER,
            attn_metadata=make_metadata(),
        )


def test_attachment_rejects_non_cachegen_quantizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_environment(monkeypatch)
    enable_shadow_write(monkeypatch)
    runner = make_runner()

    with pytest.raises(ValueError, match="quantizer_id=2"):
        runner._attach_cachegen_int8_shadow_write(
            layer_name=SELECTED_LAYER,
            attn_metadata=make_metadata(quantizer_id=0),
        )


def test_attachment_skips_out_of_range_page_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_environment(monkeypatch)
    enable_shadow_write(monkeypatch)

    runner = make_runner()
    runner.cachegen_int8_shadow_callback_calls = torch.zeros(
        (), dtype=torch.int64
    )
    runner.cachegen_int8_shadow_tokens_written = torch.zeros(
        (), dtype=torch.int64
    )
    runner.cachegen_int8_shadow_sample = None
    metadata = make_metadata()
    metadata.quantized_page_ids = torch.tensor([0, 2], dtype=torch.int32)
    metadata.quantized_page_offsets = torch.tensor([0, 0], dtype=torch.int32)

    runner._attach_cachegen_int8_shadow_write(
        layer_name=SELECTED_LAYER,
        attn_metadata=metadata,
    )

    keys = torch.randn((2, NUM_KV_HEADS, HEAD_SIZE), dtype=torch.float32)
    values = torch.randn_like(keys)
    metadata.cachegen_int8_shadow_write(keys, values, metadata)

    adapter = runner.cachegen_int8_page_adapter
    assert adapter is not None
    decoded_key, decoded_value = adapter.read_token(
        page_id=0,
        page_offset=0,
    )
    assert torch.allclose(decoded_key, keys[0], atol=0.04, rtol=0.02)
    assert torch.allclose(decoded_value, values[0], atol=0.04, rtol=0.02)
    assert runner.cachegen_int8_shadow_callback_calls.item() == 1
    assert runner.cachegen_int8_shadow_tokens_written.item() == 1


def test_attachment_rejects_out_of_range_page_offset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_environment(monkeypatch)
    enable_shadow_write(monkeypatch)
    runner = make_runner()
    metadata = make_metadata()
    metadata.quantized_page_offsets = torch.tensor(
        [0, runner.cachegen_int8_page_adapter.tokens_per_page],
        dtype=torch.int32,
    )

    runner._attach_cachegen_int8_shadow_write(
        layer_name=SELECTED_LAYER,
        attn_metadata=metadata,
    )

    keys = torch.randn((2, NUM_KV_HEADS, HEAD_SIZE), dtype=torch.float32)
    with pytest.raises(ValueError, match="page_offset must be in"):
        metadata.cachegen_int8_shadow_write(keys, keys, metadata)


def test_attachment_diagnostics_are_disabled_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_environment(monkeypatch)
    enable_shadow_write(monkeypatch)
    runner = make_runner()
    runner.cachegen_int8_shadow_callback_calls = None
    runner.cachegen_int8_shadow_tokens_written = None
    metadata = make_metadata()

    runner._attach_cachegen_int8_shadow_write(
        layer_name=SELECTED_LAYER,
        attn_metadata=metadata,
    )

    keys = torch.randn((2, NUM_KV_HEADS, HEAD_SIZE), dtype=torch.float32)
    metadata.cachegen_int8_shadow_write(keys, keys, metadata)

    assert runner.get_cachegen_int8_shadow_diagnostics() is None


def test_attachment_diagnostics_count_successful_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_environment(monkeypatch)
    enable_shadow_write(monkeypatch)
    monkeypatch.setenv(
        "VLLM_EXPERIMENTAL_CACHEGEN_INT8_SHADOW_DIAGNOSTICS",
        "1",
    )
    runner = make_runner()
    runner.cachegen_int8_shadow_callback_calls = torch.zeros(
        (), dtype=torch.int64
    )
    runner.cachegen_int8_shadow_tokens_written = torch.zeros(
        (), dtype=torch.int64
    )
    metadata = make_metadata()

    runner._attach_cachegen_int8_shadow_write(
        layer_name=SELECTED_LAYER,
        attn_metadata=metadata,
    )

    keys = torch.randn((2, NUM_KV_HEADS, HEAD_SIZE), dtype=torch.float32)
    metadata.cachegen_int8_shadow_write(keys, keys, metadata)

    assert runner.get_cachegen_int8_shadow_diagnostics() == {
        "callback_calls": 1,
        "tokens_written": 2,
        "sample_count": 0,
    }


def test_selected_layer_uses_private_shadow_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_environment(monkeypatch)
    enable_shadow_write(monkeypatch)
    monkeypatch.setenv(
        "VLLM_EXPERIMENTAL_CACHEGEN_INT8_SHADOW_DIAGNOSTICS",
        "1",
    )

    runner = make_runner()
    runner.cachegen_int8_shadow_callback_calls = torch.zeros(
        (), dtype=torch.int64
    )
    runner.cachegen_int8_shadow_tokens_written = torch.zeros(
        (), dtype=torch.int64
    )

    shared_metadata = make_metadata()
    selected_metadata = dataclasses.replace(shared_metadata)
    runner._attach_cachegen_int8_shadow_write(
        layer_name=SELECTED_LAYER,
        attn_metadata=selected_metadata,
    )

    metadata_by_layer = {
        SELECTED_LAYER: selected_metadata,
        "model.decoder.layers.1.self_attn.attn": shared_metadata,
    }

    assert metadata_by_layer[SELECTED_LAYER] is not (
        metadata_by_layer["model.decoder.layers.1.self_attn.attn"]
    )
    assert (
        metadata_by_layer[SELECTED_LAYER].cachegen_int8_shadow_write
        is not None
    )
    assert (
        metadata_by_layer[
            "model.decoder.layers.1.self_attn.attn"
        ].cachegen_int8_shadow_write
        is None
    )

    keys = torch.randn((2, NUM_KV_HEADS, HEAD_SIZE), dtype=torch.float32)
    metadata_by_layer[SELECTED_LAYER].cachegen_int8_shadow_write(
        keys,
        keys,
        metadata_by_layer[SELECTED_LAYER],
    )

    assert runner.get_cachegen_int8_shadow_diagnostics() == {
        "callback_calls": 1,
        "tokens_written": 2,
        "sample_count": 0,
    }


def test_attachment_records_one_round_trip_sample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_environment(monkeypatch)
    enable_shadow_write(monkeypatch)
    monkeypatch.setenv(
        "VLLM_EXPERIMENTAL_CACHEGEN_INT8_SHADOW_DIAGNOSTICS",
        "1",
    )
    monkeypatch.setenv(
        "VLLM_EXPERIMENTAL_CACHEGEN_INT8_SHADOW_SAMPLE",
        "1",
    )

    runner = make_runner()
    runner.cachegen_int8_shadow_callback_calls = torch.zeros(
        (), dtype=torch.int64
    )
    runner.cachegen_int8_shadow_tokens_written = torch.zeros(
        (), dtype=torch.int64
    )
    runner.cachegen_int8_shadow_sample = None
    metadata = make_metadata()

    runner._attach_cachegen_int8_shadow_write(
        layer_name=SELECTED_LAYER,
        attn_metadata=metadata,
    )

    keys = torch.randn((2, NUM_KV_HEADS, HEAD_SIZE), dtype=torch.float32)
    values = torch.randn_like(keys)
    metadata.cachegen_int8_shadow_write(keys, values, metadata)

    diagnostics = runner.get_cachegen_int8_shadow_diagnostics()
    assert diagnostics is not None
    assert diagnostics["callback_calls"] == 1
    assert diagnostics["tokens_written"] == 2
    assert diagnostics["sample_count"] == 1
    assert diagnostics["page_id"] == 0
    assert diagnostics["page_offset"] == 0
    assert diagnostics["key_max_abs_error"] < 0.05
    assert diagnostics["value_max_abs_error"] < 0.05
    assert diagnostics["key_rmse"] < 0.01
    assert diagnostics["value_rmse"] < 0.01


def test_all_layer_callbacks_use_isolated_adapters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_environment(monkeypatch)
    enable_shadow_write(monkeypatch)
    monkeypatch.setenv(
        "VLLM_EXPERIMENTAL_CACHEGEN_INT8_SHADOW_ALL_LAYERS",
        "1",
    )
    monkeypatch.setenv(
        "VLLM_EXPERIMENTAL_CACHEGEN_INT8_SHADOW_DIAGNOSTICS",
        "1",
    )

    layer_names = (
        SELECTED_LAYER,
        "model.decoder.layers.1.self_attn.attn",
    )
    runner = make_runner()
    runner.cachegen_int8_page_adapter = None
    runner.cachegen_int8_page_adapters = {
        layer_name: CacheGenInt8FixedBytePageAdapter(
            page_pool=torch.zeros((2, PAGE_BYTES), dtype=torch.uint8),
            num_kv_heads=NUM_KV_HEADS,
            head_size=HEAD_SIZE,
        )
        for layer_name in layer_names
    }
    runner.cachegen_int8_shadow_callback_calls = torch.zeros(
        (), dtype=torch.int64
    )
    runner.cachegen_int8_shadow_tokens_written = torch.zeros(
        (), dtype=torch.int64
    )
    runner.cachegen_int8_shadow_layer_counters = {
        layer_name: (
            torch.zeros((), dtype=torch.int64),
            torch.zeros((), dtype=torch.int64),
        )
        for layer_name in layer_names
    }
    runner.cachegen_int8_shadow_sample = None

    first_metadata = make_metadata()
    second_metadata = make_metadata()
    runner._attach_cachegen_int8_shadow_write(
        layer_name=layer_names[0],
        attn_metadata=first_metadata,
    )
    runner._attach_cachegen_int8_shadow_write(
        layer_name=layer_names[1],
        attn_metadata=second_metadata,
    )

    first_key = torch.full(
        (2, NUM_KV_HEADS, HEAD_SIZE),
        0.25,
        dtype=torch.float32,
    )
    first_value = torch.full_like(first_key, -0.50)
    second_key = torch.full_like(first_key, 1.50)
    second_value = torch.full_like(first_key, -1.25)

    first_metadata.cachegen_int8_shadow_write(
        first_key,
        first_value,
        first_metadata,
    )
    second_metadata.cachegen_int8_shadow_write(
        second_key,
        second_value,
        second_metadata,
    )

    first_adapter = runner.cachegen_int8_page_adapters[layer_names[0]]
    second_adapter = runner.cachegen_int8_page_adapters[layer_names[1]]

    decoded_first_key, decoded_first_value = first_adapter.read_token(
        page_id=0,
        page_offset=0,
    )
    decoded_second_key, decoded_second_value = second_adapter.read_token(
        page_id=0,
        page_offset=0,
    )

    assert torch.allclose(decoded_first_key, first_key[0], atol=0.01, rtol=0)
    assert torch.allclose(
        decoded_first_value,
        first_value[0],
        atol=0.01,
        rtol=0,
    )
    assert torch.allclose(
        decoded_second_key,
        second_key[0],
        atol=0.01,
        rtol=0,
    )
    assert torch.allclose(
        decoded_second_value,
        second_value[0],
        atol=0.01,
        rtol=0,
    )

    diagnostics = runner.get_cachegen_int8_shadow_diagnostics()
    assert diagnostics is not None
    assert diagnostics["callback_calls"] == 2
    assert diagnostics["tokens_written"] == 4
    assert diagnostics["layers_with_writes"] == 2
    assert diagnostics["layer_counters"] == {
        layer_names[0]: {
            "callback_calls": 1,
            "tokens_written": 2,
        },
        layer_names[1]: {
            "callback_calls": 1,
            "tokens_written": 2,
        },
    }


def test_all_layer_attachment_does_not_require_selected_layer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_environment(monkeypatch)
    enable_shadow_write(monkeypatch)
    monkeypatch.setenv(
        "VLLM_EXPERIMENTAL_CACHEGEN_INT8_SHADOW_ALL_LAYERS",
        "1",
    )
    monkeypatch.delenv(
        "VLLM_EXPERIMENTAL_CACHEGEN_INT8_SHADOW_LAYER",
        raising=False,
    )

    layer_names = (
        SELECTED_LAYER,
        "model.decoder.layers.1.self_attn.attn",
    )
    runner = make_runner()
    runner.cachegen_int8_page_adapter = None
    runner.cachegen_int8_page_adapters = {
        layer_name: CacheGenInt8FixedBytePageAdapter(
            page_pool=torch.zeros((2, PAGE_BYTES), dtype=torch.uint8),
            num_kv_heads=NUM_KV_HEADS,
            head_size=HEAD_SIZE,
        )
        for layer_name in layer_names
    }
    runner.cachegen_int8_shadow_callback_calls = torch.zeros(
        (), dtype=torch.int64
    )
    runner.cachegen_int8_shadow_tokens_written = torch.zeros(
        (), dtype=torch.int64
    )
    runner.cachegen_int8_shadow_layer_counters = {
        layer_name: (
            torch.zeros((), dtype=torch.int64),
            torch.zeros((), dtype=torch.int64),
        )
        for layer_name in layer_names
    }
    runner.cachegen_int8_shadow_sample = None

    for layer_name in layer_names:
        metadata = make_metadata()
        runner._attach_cachegen_int8_shadow_write(
            layer_name=layer_name,
            attn_metadata=metadata,
        )
        assert metadata.cachegen_int8_shadow_write is not None


def test_calibration_attachment_is_independent_of_shadow_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.v1.worker.experimental.cachegen_int8_calibration_config import (
        CACHEGEN_INT8_CALIBRATION_LAYER_ENV_VAR,
    )

    clear_environment(monkeypatch)
    monkeypatch.setenv(
        CACHEGEN_INT8_CALIBRATION_LAYER_ENV_VAR,
        SELECTED_LAYER,
    )

    runner = make_runner()
    runner.cachegen_int8_calibration_stats = {
        "layer_name": SELECTED_LAYER,
        "callback_calls": 0,
        "tokens_seen": 0,
        "key_abs_max_per_head": torch.zeros(
            NUM_KV_HEADS,
            dtype=torch.float32,
        ),
        "value_abs_max_per_head": torch.zeros(
            NUM_KV_HEADS,
            dtype=torch.float32,
        ),
    }
    metadata = make_metadata()

    runner._attach_cachegen_int8_calibration(
        layer_name=SELECTED_LAYER,
        attn_metadata=metadata,
    )

    assert metadata.cachegen_int8_calibration is not None
    assert metadata.cachegen_int8_shadow_write is None


def test_calibration_attachment_ignores_unselected_layer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.v1.worker.experimental.cachegen_int8_calibration_config import (
        CACHEGEN_INT8_CALIBRATION_LAYER_ENV_VAR,
    )

    clear_environment(monkeypatch)
    monkeypatch.setenv(
        CACHEGEN_INT8_CALIBRATION_LAYER_ENV_VAR,
        SELECTED_LAYER,
    )

    runner = make_runner()
    runner.cachegen_int8_calibration_stats = None
    metadata = make_metadata()

    runner._attach_cachegen_int8_calibration(
        layer_name="model.layers.99.self_attn.attn",
        attn_metadata=metadata,
    )

    assert metadata.cachegen_int8_calibration is None
    assert runner.cachegen_int8_calibration_stats is None


def test_layer_metadata_override_precedes_fast_prefill_source_order() -> None:
    from pathlib import Path

    source = Path("vllm/v1/worker/gpu_model_runner.py").read_text()

    loop_start = source.index(
        "                for layer_name in attn_group.layer_names:\n",
        source.index("layer_metadata_overrides"),
    )
    override_check = source.index(
        "if layer_name in layer_metadata_overrides:",
        loop_start,
    )
    fast_prefill_check = source.index(
        "elif (self.cache_config.kv_sharing_fast_prefill",
        loop_start,
    )

    assert override_check < fast_prefill_check
