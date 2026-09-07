# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.worker.experimental.cachegen_int8_byte_page_adapter import (
    CacheGenInt8FixedBytePageAdapter,
)
from vllm.v1.worker.gpu_model_runner import GPUModelRunner


NUM_KV_HEADS = 8
HEAD_SIZE = 128
PAGE_BYTES = 128 * 1024
LAYER_0 = "layers.0.attn"
LAYER_1 = "layers.1.attn"


def make_runner() -> GPUModelRunner:
    runner = object.__new__(GPUModelRunner)
    runner.cachegen_int8_page_adapters = {
        layer_name: CacheGenInt8FixedBytePageAdapter(
            page_pool=torch.zeros((3, PAGE_BYTES), dtype=torch.uint8),
            num_kv_heads=NUM_KV_HEADS,
            head_size=HEAD_SIZE,
        )
        for layer_name in (LAYER_0, LAYER_1)
    }
    runner.cachegen_int8_page_adapter = None
    return runner


def test_reads_layer_isolated_mapped_tokens() -> None:
    runner = make_runner()
    page_ids = torch.tensor([0, 2], dtype=torch.int32)
    page_offsets = torch.tensor([0, 1], dtype=torch.int32)

    first_keys = torch.full(
        (2, NUM_KV_HEADS, HEAD_SIZE),
        0.25,
        dtype=torch.float32,
    )
    first_values = torch.full_like(first_keys, -0.50)
    second_keys = torch.full_like(first_keys, 1.50)
    second_values = torch.full_like(first_keys, -1.25)

    for index in range(2):
        runner.cachegen_int8_page_adapters[LAYER_0].write_token(
            page_id=int(page_ids[index]),
            page_offset=int(page_offsets[index]),
            key=first_keys[index],
            value=first_values[index],
        )
        runner.cachegen_int8_page_adapters[LAYER_1].write_token(
            page_id=int(page_ids[index]),
            page_offset=int(page_offsets[index]),
            key=second_keys[index],
            value=second_values[index],
        )

    decoded_first_keys, decoded_first_values = (
        runner.read_cachegen_int8_shadow_kv(
            layer_name=LAYER_0,
            quantized_page_ids=page_ids,
            quantized_page_offsets=page_offsets,
            dtype=torch.float32,
        )
    )
    decoded_second_keys, decoded_second_values = (
        runner.read_cachegen_int8_shadow_kv(
            layer_name=LAYER_1,
            quantized_page_ids=page_ids,
            quantized_page_offsets=page_offsets,
            dtype=torch.float32,
        )
    )

    assert torch.allclose(decoded_first_keys, first_keys, atol=0.01, rtol=0)
    assert torch.allclose(
        decoded_first_values,
        first_values,
        atol=0.01,
        rtol=0,
    )
    assert torch.allclose(decoded_second_keys, second_keys, atol=0.01, rtol=0)
    assert torch.allclose(
        decoded_second_values,
        second_values,
        atol=0.01,
        rtol=0,
    )


def test_rejects_invalid_mapping_shapes() -> None:
    runner = make_runner()

    with pytest.raises(ValueError, match="one-dimensional"):
        runner.read_cachegen_int8_shadow_kv(
            layer_name=LAYER_0,
            quantized_page_ids=torch.zeros((1, 1), dtype=torch.int32),
            quantized_page_offsets=torch.zeros((1, 1), dtype=torch.int32),
            dtype=torch.float32,
        )

    with pytest.raises(ValueError, match="same shape"):
        runner.read_cachegen_int8_shadow_kv(
            layer_name=LAYER_0,
            quantized_page_ids=torch.tensor([0], dtype=torch.int32),
            quantized_page_offsets=torch.tensor([0, 1], dtype=torch.int32),
            dtype=torch.float32,
        )


def test_rejects_nonfloating_output_dtype() -> None:
    runner = make_runner()

    with pytest.raises(ValueError, match="dtype must be floating point"):
        runner.read_cachegen_int8_shadow_kv(
            layer_name=LAYER_0,
            quantized_page_ids=torch.tensor([], dtype=torch.int32),
            quantized_page_offsets=torch.tensor([], dtype=torch.int32),
            dtype=torch.int8,
        )


def test_reads_empty_mapping_with_expected_shape() -> None:
    runner = make_runner()

    keys, values = runner.read_cachegen_int8_shadow_kv(
        layer_name=LAYER_0,
        quantized_page_ids=torch.tensor([], dtype=torch.int32),
        quantized_page_offsets=torch.tensor([], dtype=torch.int32),
        dtype=torch.float16,
    )

    assert keys.shape == (0, NUM_KV_HEADS, HEAD_SIZE)
    assert values.shape == (0, NUM_KV_HEADS, HEAD_SIZE)
    assert keys.dtype is torch.float16
    assert values.dtype is torch.float16
