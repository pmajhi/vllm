# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

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
            page_pool=torch.zeros((2, PAGE_BYTES), dtype=torch.uint8),
            num_kv_heads=NUM_KV_HEADS,
            head_size=HEAD_SIZE,
        )
        for layer_name in (LAYER_0, LAYER_1)
    }
    page_ids = torch.tensor([1], dtype=torch.int32)
    page_offsets = torch.tensor([0], dtype=torch.int32)
    runner.input_batch = SimpleNamespace(
        block_table=[
            SimpleNamespace(
                quantized_page_ids=page_ids,
                quantized_page_offsets=page_offsets,
            )
        ]
    )
    return runner


def test_audit_reads_one_mapped_token_per_layer() -> None:
    runner = make_runner()
    first_key = torch.full(
        (NUM_KV_HEADS, HEAD_SIZE),
        0.25,
        dtype=torch.float32,
    )
    first_value = torch.full_like(first_key, -0.50)
    second_key = torch.full_like(first_key, 1.50)
    second_value = torch.full_like(first_key, -1.25)

    runner.cachegen_int8_page_adapters[LAYER_0].write_token(
        page_id=1,
        page_offset=0,
        key=first_key,
        value=first_value,
    )
    runner.cachegen_int8_page_adapters[LAYER_1].write_token(
        page_id=1,
        page_offset=0,
        key=second_key,
        value=second_value,
    )

    audit = runner.get_cachegen_int8_layer_page_audit()

    assert set(audit) == {LAYER_0, LAYER_1}
    assert audit[LAYER_0]["page_id"] == 1
    assert audit[LAYER_0]["page_offset"] == 0
    assert audit[LAYER_0]["key_abs_max"] == 0.25
    assert audit[LAYER_0]["value_abs_max"] == 0.5
    assert audit[LAYER_1]["key_abs_max"] == 1.5
    assert audit[LAYER_1]["value_abs_max"] == 1.25


def test_audit_returns_empty_without_layer_adapters() -> None:
    runner = object.__new__(GPUModelRunner)
    runner.cachegen_int8_page_adapters = {}

    assert runner.get_cachegen_int8_layer_page_audit() == {}
