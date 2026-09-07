# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Fused adaptive-INT8 decode attention over fixed-byte common KV pages."""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _cachegen_int8_common_page_decode_kernel(
    query_ptr,
    page_bytes_ptr,
    key_scales_ptr,
    value_scales_ptr,
    page_ids_ptr,
    output_ptr,
    sequence_length,
    stride_query_head,
    stride_page,
    stride_output_head,
    num_kv_heads: tl.constexpr,
    num_query_heads: tl.constexpr,
    head_size: tl.constexpr,
    tokens_per_page: tl.constexpr,
    scale_count: tl.constexpr,
    key_scale_offset: tl.constexpr,
    value_scale_offset: tl.constexpr,
    key_payload_offset: tl.constexpr,
    value_payload_offset: tl.constexpr,
    softmax_scale: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """One program computes one query head's one-token decode attention."""
    query_head = tl.program_id(0)
    queries_per_kv_head = num_query_heads // num_kv_heads
    kv_head = query_head // queries_per_kv_head
    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < head_size

    query = tl.load(
        query_ptr + query_head * stride_query_head + offs_d,
        mask=d_mask,
        other=0.0,
    ).to(tl.float32)

    max_logit = -float("inf")
    sum_exp = 0.0
    accumulator = tl.zeros((BLOCK_D,), dtype=tl.float32)

    num_blocks = tl.cdiv(sequence_length, BLOCK_T)
    for block_index in range(0, num_blocks):
        offs_t = block_index * BLOCK_T + tl.arange(0, BLOCK_T)
        token_mask = offs_t < sequence_length

        page_indices = offs_t // tokens_per_page
        page_offsets = offs_t % tokens_per_page
        page_ids = tl.load(
            page_ids_ptr + page_indices,
            mask=token_mask,
            other=0,
        ).to(tl.int64)

        key_scales = tl.load(
            key_scales_ptr + page_ids * num_kv_heads + kv_head,
            mask=token_mask,
            other=0.0,
        ).to(tl.float32)

        value_scales = tl.load(
            value_scales_ptr + page_ids * num_kv_heads + kv_head,
            mask=token_mask,
            other=0.0,
        ).to(tl.float32)

        payload_offsets = (
            page_offsets[:, None] * num_kv_heads * head_size
            + kv_head * head_size
            + offs_d[None, :]
        )

        key_bytes = tl.load(
            page_bytes_ptr
            + page_ids[:, None] * stride_page
            + key_payload_offset
            + payload_offsets,
            mask=token_mask[:, None] & d_mask[None, :],
            other=0,
        ).to(tl.int8)
        key_values = key_bytes.to(tl.float32) * key_scales[:, None]

        logits = tl.sum(key_values * query[None, :], axis=1) * softmax_scale
        logits = tl.where(token_mask, logits, -float("inf"))

        block_max = tl.max(logits, axis=0)
        new_max = tl.maximum(max_logit, block_max)
        exp_logits = tl.exp(logits - new_max)
        exp_logits = tl.where(token_mask, exp_logits, 0.0)

        value_bytes = tl.load(
            page_bytes_ptr
            + page_ids[:, None] * stride_page
            + value_payload_offset
            + payload_offsets,
            mask=token_mask[:, None] & d_mask[None, :],
            other=0,
        ).to(tl.int8)
        value_values = value_bytes.to(tl.float32) * value_scales[:, None]

        rescale = tl.exp(max_logit - new_max)
        accumulator = accumulator * rescale + tl.sum(
            value_values * exp_logits[:, None],
            axis=0,
        )
        sum_exp = sum_exp * rescale + tl.sum(exp_logits, axis=0)
        max_logit = new_max

    tl.store(
        output_ptr + query_head * stride_output_head + offs_d,
        accumulator / sum_exp,
        mask=d_mask,
    )


def cachegen_int8_common_page_decode_attention_triton(
    *,
    query: torch.Tensor,
    page_bytes: torch.Tensor,
    key_scales: torch.Tensor,
    value_scales: torch.Tensor,
    page_ids: torch.Tensor,
    sequence_length: int,
    num_kv_heads: int,
    tokens_per_page: int,
    head_size: int,
    metadata_nbytes: int,
    key_payload_offset: int,
    value_payload_offset: int,
    scale: float | None = None,
) -> torch.Tensor:
    """Run fused adaptive-INT8 decode directly from fixed-byte page bytes."""
    if not torch.cuda.is_available():
        raise RuntimeError("Common-page Triton decode requires CUDA")
    if query.ndim != 2:
        raise ValueError(
            f"query must have shape [num_heads, head_size], got "
            f"{tuple(query.shape)}"
        )
    if page_bytes.ndim != 2 or page_bytes.dtype is not torch.uint8:
        raise ValueError(
            "page_bytes must have shape [num_pages, page_nbytes] and "
            "dtype torch.uint8"
        )
    if key_scales.shape != (page_bytes.shape[0], num_kv_heads):
        raise ValueError("key_scales has invalid shape")
    if value_scales.shape != (page_bytes.shape[0], num_kv_heads):
        raise ValueError("value_scales has invalid shape")
    if key_scales.dtype is not torch.float32:
        raise ValueError("key_scales must have dtype torch.float32")
    if value_scales.dtype is not torch.float32:
        raise ValueError("value_scales must have dtype torch.float32")
    if page_ids.ndim != 1 or page_ids.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError("page_ids must be a one-dimensional integer tensor")
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if num_kv_heads <= 0 or head_size <= 0 or tokens_per_page <= 0:
        raise ValueError("KV geometry must be positive")
    if query.shape[1] != head_size:
        raise ValueError(
            f"query head size must be {head_size}, got {query.shape[1]}"
        )
    if query.shape[0] % num_kv_heads:
        raise ValueError(
            "num_query_heads must be divisible by num_kv_heads"
        )
    if metadata_nbytes < 2 * num_kv_heads * 4:
        raise ValueError("metadata_nbytes is too small for K/V FP32 scales")
    if (
        query.device != page_bytes.device
        or query.device != key_scales.device
        or query.device != value_scales.device
        or query.device != page_ids.device
    ):
        raise ValueError(
            "query, page_bytes, scales, and page_ids must share a device"
        )
    if not query.is_cuda:
        raise ValueError("query must be a CUDA tensor")

    required_pages = (sequence_length + tokens_per_page - 1) // tokens_per_page
    if page_ids.numel() < required_pages:
        raise ValueError(
            "page_ids do not cover sequence length: "
            f"required={required_pages}, got={page_ids.numel()}"
        )
    if torch.any(page_ids[:required_pages] < 0):
        raise ValueError("page_ids must be non-negative")
    if torch.any(page_ids[:required_pages] >= page_bytes.shape[0]):
        raise ValueError("page_ids exceed page_bytes capacity")

    output = torch.empty_like(query)
    block_t = 16
    block_d = triton.next_power_of_2(head_size)
    grid = (query.shape[0],)

    _cachegen_int8_common_page_decode_kernel[grid](
        query,
        page_bytes,
        key_scales,
        value_scales,
        page_ids,
        output,
        sequence_length,
        query.stride(0),
        page_bytes.stride(0),
        output.stride(0),
        num_kv_heads=num_kv_heads,
        num_query_heads=query.shape[0],
        head_size=head_size,
        tokens_per_page=tokens_per_page,
        scale_count=num_kv_heads,
        key_scale_offset=0,
        value_scale_offset=num_kv_heads * 4,
        key_payload_offset=key_payload_offset,
        value_payload_offset=value_payload_offset,
        softmax_scale=(
            scale if scale is not None else 1.0 / math.sqrt(head_size)
        ),
        BLOCK_T=block_t,
        BLOCK_D=block_d,
    )
    return output
