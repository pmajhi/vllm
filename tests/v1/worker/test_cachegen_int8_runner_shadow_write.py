# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.worker.block_table import MultiGroupBlockTable
from vllm.v1.worker.experimental.cachegen_int8_byte_page_adapter import (
    CacheGenInt8FixedBytePageAdapter,
)
from vllm.v1.worker.gpu_model_runner import GPUModelRunner


PAGE_BYTES = 128 * 1024
NUM_KV_HEADS = 8
HEAD_SIZE = 128


def make_runner() -> GPUModelRunner:
    runner = object.__new__(GPUModelRunner)
    runner.cachegen_int8_page_adapter = CacheGenInt8FixedBytePageAdapter(
        page_pool=torch.zeros((3, PAGE_BYTES), dtype=torch.uint8),
        num_kv_heads=NUM_KV_HEADS,
        head_size=HEAD_SIZE,
    )
    runner.input_batch = type(
        "InputBatchStub",
        (),
        {
            "block_table": MultiGroupBlockTable(
                max_num_reqs=4,
                max_model_len=4096,
                max_num_batched_tokens=8,
                pin_memory=False,
                device=torch.device("cpu"),
                block_sizes=[16],
            )
        },
    )()
    return runner


def set_mapping(
    runner: GPUModelRunner,
    page_ids: list[int],
    page_offsets: list[int],
) -> None:
    block_table = runner.input_batch.block_table[0]
    block_table.set_quantized_mapping(
        torch.tensor(page_ids, dtype=torch.int32).numpy(),
        torch.tensor(page_offsets, dtype=torch.int32).numpy(),
    )
    block_table.commit_quantized_mapping(len(page_ids))


def test_runner_shadow_write_uses_block_table_mapping() -> None:
    torch.manual_seed(3)
    runner = make_runner()
    set_mapping(
        runner,
        page_ids=[2, 0, 2],
        page_offsets=[
            0,
            runner.cachegen_int8_page_adapter.tokens_per_page - 1,
            1,
        ],
    )
    keys = torch.randn((3, NUM_KV_HEADS, HEAD_SIZE), dtype=torch.float32)
    values = torch.randn_like(keys)

    runner.write_cachegen_int8_shadow_kv(
        kv_cache_group_id=0,
        keys=keys,
        values=values,
    )

    for index, (page_id, page_offset) in enumerate(
        zip([2, 0, 2], [0, runner.cachegen_int8_page_adapter.tokens_per_page - 1, 1])
    ):
        decoded_key, decoded_value = (
            runner.cachegen_int8_page_adapter.read_token(
                page_id=page_id,
                page_offset=page_offset,
            )
        )
        assert torch.allclose(decoded_key, keys[index], atol=0.04, rtol=0.02)
        assert torch.allclose(
            decoded_value,
            values[index],
            atol=0.04,
            rtol=0.02,
        )


def test_runner_shadow_write_requires_initialized_adapter() -> None:
    runner = make_runner()
    runner.cachegen_int8_page_adapter = None
    keys = torch.zeros((1, NUM_KV_HEADS, HEAD_SIZE), dtype=torch.float32)

    with pytest.raises(RuntimeError, match="not initialized"):
        runner.write_cachegen_int8_shadow_kv(
            kv_cache_group_id=0,
            keys=keys,
            values=keys,
        )


def test_runner_shadow_write_rejects_invalid_group_id() -> None:
    runner = make_runner()
    keys = torch.zeros((1, NUM_KV_HEADS, HEAD_SIZE), dtype=torch.float32)

    with pytest.raises(ValueError, match="kv_cache_group_id must be"):
        runner.write_cachegen_int8_shadow_kv(
            kv_cache_group_id=1,
            keys=keys,
            values=keys,
        )


def test_runner_shadow_write_rejects_value_shape_mismatch() -> None:
    runner = make_runner()
    keys = torch.zeros((2, NUM_KV_HEADS, HEAD_SIZE), dtype=torch.float32)
    values = torch.zeros((1, NUM_KV_HEADS, HEAD_SIZE), dtype=torch.float32)

    with pytest.raises(ValueError, match="values must have shape"):
        runner.write_cachegen_int8_shadow_kv(
            kv_cache_group_id=0,
            keys=keys,
            values=values,
        )
