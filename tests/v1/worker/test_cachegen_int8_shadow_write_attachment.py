# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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
    ):
        monkeypatch.delenv(name, raising=False)


def enable_shadow_write(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_EXPERIMENTAL_HETERO_KV_PAGE_PLANNING", "1")
    monkeypatch.setenv("VLLM_EXPERIMENTAL_HETERO_KV_PAGE_POOL", "1")
    monkeypatch.setenv("VLLM_EXPERIMENTAL_HETERO_KV_PAGE_POOL_PAGES", "2")
    monkeypatch.setenv("VLLM_EXPERIMENTAL_CACHEGEN_INT8_PAGE_ADAPTER", "1")
    monkeypatch.setenv("VLLM_EXPERIMENTAL_CACHEGEN_INT8_SHADOW_WRITE", "1")


def test_attachment_is_noop_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_environment(monkeypatch)
    runner = make_runner()
    metadata = make_metadata()

    runner._attach_cachegen_int8_shadow_write(attn_metadata=metadata)

    assert metadata.cachegen_int8_shadow_write is None


def test_attachment_writes_mapped_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_environment(monkeypatch)
    enable_shadow_write(monkeypatch)
    runner = make_runner()
    metadata = make_metadata()

    runner._attach_cachegen_int8_shadow_write(attn_metadata=metadata)

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
            attn_metadata=make_metadata(quantizer_id=0),
        )
