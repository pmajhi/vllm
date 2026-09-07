# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.utils import CommonAttentionMetadata


def make_builder() -> FlashAttentionMetadataBuilder:
    builder = object.__new__(FlashAttentionMetadataBuilder)
    builder.aot_schedule = False
    builder.aot_sliding_window = None
    builder.use_full_cuda_graph = False
    builder.max_cuda_graph_size = 0
    builder.max_num_splits = 0
    builder.cache_config = SimpleNamespace(cache_dtype="auto")
    builder.kv_cache_dtype = torch.bfloat16
    builder.device = torch.device("cpu")
    return builder


def make_common_metadata() -> CommonAttentionMetadata:
    return CommonAttentionMetadata(
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 2], dtype=torch.int32),
        seq_lens=torch.tensor([2], dtype=torch.int32),
        seq_lens_cpu=torch.tensor([2], dtype=torch.int32),
        num_computed_tokens_cpu=torch.tensor([0], dtype=torch.int32),
        num_reqs=1,
        num_actual_tokens=2,
        max_query_len=2,
        block_table_tensor=torch.tensor([[0]], dtype=torch.int32),
        slot_mapping=torch.tensor([0, 1], dtype=torch.int64),
        quantizer_id=torch.tensor([2], dtype=torch.int32),
        tokens_per_page=torch.tensor([62], dtype=torch.int32),
        quantized_page_ids=torch.tensor([3, 3], dtype=torch.int32),
        quantized_page_offsets=torch.tensor([0, 1], dtype=torch.int32),
        causal=True,
    )


def test_flash_attention_metadata_preserves_hetero_page_tensors() -> None:
    common_metadata = make_common_metadata()
    metadata = make_builder().build(
        common_prefix_len=0,
        common_attn_metadata=common_metadata,
    )

    assert metadata.quantizer_id is common_metadata.quantizer_id
    assert metadata.tokens_per_page is common_metadata.tokens_per_page
    assert metadata.quantized_page_ids is common_metadata.quantized_page_ids
    assert (
        metadata.quantized_page_offsets
        is common_metadata.quantized_page_offsets
    )


def test_flash_attention_metadata_defaults_are_optional() -> None:
    from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata

    metadata = FlashAttentionMetadata(
        num_actual_tokens=0,
        max_query_len=0,
        query_start_loc=torch.zeros(1, dtype=torch.int32),
        max_seq_len=0,
        seq_lens=torch.zeros(0, dtype=torch.int32),
        block_table=torch.zeros((0, 0), dtype=torch.int32),
        slot_mapping=torch.zeros(0, dtype=torch.int64),
        use_cascade=False,
        common_prefix_len=0,
        cu_prefix_query_lens=None,
        prefix_kv_lens=None,
        suffix_kv_lens=None,
    )

    assert metadata.quantizer_id is None
    assert metadata.tokens_per_page is None
    assert metadata.quantized_page_ids is None
    assert metadata.quantized_page_offsets is None


def test_flash_attention_metadata_accepts_shadow_write_callback() -> None:
    captured: list[tuple[torch.Tensor, torch.Tensor]] = []

    def callback(
        key: torch.Tensor,
        value: torch.Tensor,
        metadata,
    ) -> None:
        captured.append((key, value))
        assert metadata.quantized_page_ids is not None
        assert metadata.quantized_page_offsets is not None

    common_metadata = make_common_metadata()
    metadata = make_builder().build(
        common_prefix_len=0,
        common_attn_metadata=common_metadata,
    )
    metadata.cachegen_int8_shadow_write = callback

    key = torch.randn((2, 8, 128), dtype=torch.float32)
    value = torch.randn_like(key)
    metadata.cachegen_int8_shadow_write(key, value, metadata)

    assert len(captured) == 1
    assert captured[0][0] is key
    assert captured[0][1] is value


def test_shadow_write_helper_is_noop_without_callback() -> None:
    from vllm.v1.attention.backends.flash_attn import (
        _run_cachegen_int8_shadow_write,
    )

    metadata = make_builder().build(
        common_prefix_len=0,
        common_attn_metadata=make_common_metadata(),
    )
    key = torch.randn((3, 8, 128), dtype=torch.float32)
    value = torch.randn_like(key)

    _run_cachegen_int8_shadow_write(
        key=key,
        value=value,
        attn_metadata=metadata,
    )


def test_shadow_write_helper_slices_to_actual_tokens() -> None:
    from vllm.v1.attention.backends.flash_attn import (
        _run_cachegen_int8_shadow_write,
    )

    captured: list[tuple[torch.Tensor, torch.Tensor]] = []

    def callback(
        key: torch.Tensor,
        value: torch.Tensor,
        metadata,
    ) -> None:
        captured.append((key, value))
        assert metadata.quantized_page_ids is not None
        assert metadata.quantized_page_offsets is not None

    metadata = make_builder().build(
        common_prefix_len=0,
        common_attn_metadata=make_common_metadata(),
    )
    metadata.cachegen_int8_shadow_write = callback
    key = torch.randn((4, 8, 128), dtype=torch.float32)
    value = torch.randn_like(key)

    _run_cachegen_int8_shadow_write(
        key=key,
        value=value,
        attn_metadata=metadata,
    )

    assert len(captured) == 1
    assert captured[0][0].shape[0] == metadata.num_actual_tokens
    assert captured[0][1].shape[0] == metadata.num_actual_tokens
    assert torch.equal(captured[0][0], key[:metadata.num_actual_tokens])
    assert torch.equal(captured[0][1], value[:metadata.num_actual_tokens])


def test_shadow_write_helper_requires_mapping_when_enabled() -> None:
    from vllm.v1.attention.backends.flash_attn import (
        _run_cachegen_int8_shadow_write,
    )

    metadata = make_builder().build(
        common_prefix_len=0,
        common_attn_metadata=make_common_metadata(),
    )
    metadata.cachegen_int8_shadow_write = lambda key, value, metadata: None
    metadata.quantized_page_ids = None
    key = torch.randn((2, 8, 128), dtype=torch.float32)
    value = torch.randn_like(key)

    with pytest.raises(AssertionError, match="requires quantized page IDs"):
        _run_cachegen_int8_shadow_write(
            key=key,
            value=value,
            attn_metadata=metadata,
        )
