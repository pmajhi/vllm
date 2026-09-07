# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Reference paged INT8-KV decode attention.

This module is intentionally eager and correctness-oriented. It establishes the
layout and numerical contract for a later fused Triton/CUDA implementation.

It does not modify vLLM's standard attention path.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


_INT8_QMAX = 127.0


@dataclass(frozen=True)
class CacheGenInt8PagedKV:
    """Paged symmetric-INT8 K/V cache with per-page, per-KV-head scales.

    key_pages and value_pages have shape:
        [num_pages, tokens_per_page, num_kv_heads, head_size]

    key_scales and value_scales have shape:
        [num_pages, num_kv_heads]

    Each scale applies to every token and head-dimension element for one
    `(page_id, kv_head)` region.
    """

    key_pages: torch.Tensor
    value_pages: torch.Tensor
    key_scales: torch.Tensor
    value_scales: torch.Tensor

    def __post_init__(self) -> None:
        if self.key_pages.dtype != torch.int8:
            raise ValueError(
                f"key_pages must have dtype torch.int8; got {self.key_pages.dtype}"
            )
        if self.value_pages.dtype != torch.int8:
            raise ValueError(
                "value_pages must have dtype torch.int8; "
                f"got {self.value_pages.dtype}"
            )
        if self.key_pages.shape != self.value_pages.shape:
            raise ValueError(
                "key_pages and value_pages must have identical shape; got "
                f"{tuple(self.key_pages.shape)} and "
                f"{tuple(self.value_pages.shape)}"
            )
        if self.key_pages.ndim != 4:
            raise ValueError(
                "key_pages must have shape "
                "[num_pages, tokens_per_page, num_kv_heads, head_size]; got "
                f"{tuple(self.key_pages.shape)}"
            )

        num_pages, _, num_kv_heads, _ = self.key_pages.shape
        expected_scale_shape = (num_pages, num_kv_heads)
        for name, scales in (
            ("key_scales", self.key_scales),
            ("value_scales", self.value_scales),
        ):
            if tuple(scales.shape) != expected_scale_shape:
                raise ValueError(
                    f"{name} must have shape {expected_scale_shape}; got "
                    f"{tuple(scales.shape)}"
                )
            if not scales.dtype.is_floating_point:
                raise ValueError(f"{name} must have floating-point dtype")
            if scales.device != self.key_pages.device:
                raise ValueError(
                    f"{name} must be on {self.key_pages.device}; got "
                    f"{scales.device}"
                )
            if torch.any(scales < 0):
                raise ValueError(f"{name} must be nonnegative")

    @property
    def num_pages(self) -> int:
        return self.key_pages.shape[0]

    @property
    def tokens_per_page(self) -> int:
        return self.key_pages.shape[1]

    @property
    def num_kv_heads(self) -> int:
        return self.key_pages.shape[2]

    @property
    def head_size(self) -> int:
        return self.key_pages.shape[3]

    @property
    def device(self) -> torch.device:
        return self.key_pages.device


def quantize_cachegen_int8_pages(
    keys: torch.Tensor,
    values: torch.Tensor,
) -> CacheGenInt8PagedKV:
    """Quantize floating K/V pages with symmetric per-page/per-KV-head scales.

    Args:
        keys: `[num_pages, tokens_per_page, num_kv_heads, head_size]`.
        values: Same shape and device as keys.

    Returns:
        A quantized paged cache and its K/V scale tensors.
    """
    if keys.ndim != 4:
        raise ValueError(
            "keys must have shape "
            "[num_pages, tokens_per_page, num_kv_heads, head_size]; got "
            f"{tuple(keys.shape)}"
        )
    if values.shape != keys.shape:
        raise ValueError(
            f"values must have shape {tuple(keys.shape)}; got "
            f"{tuple(values.shape)}"
        )
    if not keys.dtype.is_floating_point:
        raise ValueError("keys must have floating-point dtype")
    if not values.dtype.is_floating_point:
        raise ValueError("values must have floating-point dtype")
    if keys.device != values.device:
        raise ValueError(
            f"keys and values must share a device; got {keys.device} and "
            f"{values.device}"
        )

    scale_dtype = torch.float32
    key_abs_max = keys.abs().amax(dim=(1, 3)).to(scale_dtype)
    value_abs_max = values.abs().amax(dim=(1, 3)).to(scale_dtype)

    key_scales = key_abs_max / _INT8_QMAX
    value_scales = value_abs_max / _INT8_QMAX

    safe_key_scales = torch.where(
        key_scales > 0,
        key_scales,
        torch.ones_like(key_scales),
    )
    safe_value_scales = torch.where(
        value_scales > 0,
        value_scales,
        torch.ones_like(value_scales),
    )

    key_pages = torch.clamp(
        torch.round(keys.to(scale_dtype) / safe_key_scales[:, None, :, None]),
        min=-_INT8_QMAX,
        max=_INT8_QMAX,
    ).to(torch.int8)
    value_pages = torch.clamp(
        torch.round(
            values.to(scale_dtype) / safe_value_scales[:, None, :, None]
        ),
        min=-_INT8_QMAX,
        max=_INT8_QMAX,
    ).to(torch.int8)

    return CacheGenInt8PagedKV(
        key_pages=key_pages,
        value_pages=value_pages,
        key_scales=key_scales,
        value_scales=value_scales,
    )


def _validate_decode_inputs(
    query: torch.Tensor,
    cache: CacheGenInt8PagedKV,
    block_table: torch.Tensor,
    seq_len: int,
) -> None:
    if query.ndim != 2:
        raise ValueError(
            f"query must have shape [num_query_heads, head_size]; got "
            f"{tuple(query.shape)}"
        )
    if query.shape[1] != cache.head_size:
        raise ValueError(
            f"query head size must be {cache.head_size}; got {query.shape[1]}"
        )
    if query.shape[0] % cache.num_kv_heads != 0:
        raise ValueError(
            "num_query_heads must be divisible by num_kv_heads; got "
            f"{query.shape[0]} and {cache.num_kv_heads}"
        )
    if not query.dtype.is_floating_point:
        raise ValueError("query must have floating-point dtype")
    if query.device != cache.device:
        raise ValueError(
            f"query must be on {cache.device}; got {query.device}"
        )
    if block_table.ndim != 1:
        raise ValueError(
            f"block_table must be rank 1; got {tuple(block_table.shape)}"
        )
    if block_table.dtype not in (torch.int32, torch.int64):
        raise ValueError("block_table must have integer dtype")
    if block_table.device != cache.device:
        raise ValueError(
            f"block_table must be on {cache.device}; got {block_table.device}"
        )
    if seq_len <= 0:
        raise ValueError(f"seq_len must be positive; got {seq_len}")

    required_blocks = (
        seq_len + cache.tokens_per_page - 1
    ) // cache.tokens_per_page
    if block_table.numel() < required_blocks:
        raise ValueError(
            f"block_table needs at least {required_blocks} entries for "
            f"seq_len={seq_len}; got {block_table.numel()}"
        )
    active_pages = block_table[:required_blocks]
    if torch.any(active_pages < 0) or torch.any(active_pages >= cache.num_pages):
        raise ValueError(
            "active block_table page IDs must lie in "
            f"[0, {cache.num_pages})"
        )


def _dequantize_page(
    cache: CacheGenInt8PagedKV,
    page_id: int,
    *,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    keys = cache.key_pages[page_id].to(dtype)
    values = cache.value_pages[page_id].to(dtype)
    keys = keys * cache.key_scales[page_id].to(dtype)[None, :, None]
    values = values * cache.value_scales[page_id].to(dtype)[None, :, None]
    return keys, values


def cachegen_int8_paged_decode_attention_reference(
    query: torch.Tensor,
    cache: CacheGenInt8PagedKV,
    block_table: torch.Tensor,
    seq_len: int,
) -> torch.Tensor:
    """Run causal decode attention over paged INT8 K/V using online softmax.

    Args:
        query: `[num_query_heads, head_size]`, representing one decode token.
        cache: Quantized K/V pages.
        block_table: Logical KV block index to quantized page ID.
        seq_len: Number of causal K/V tokens visible to this query.

    Returns:
        `[num_query_heads, head_size]` in `query.dtype`.
    """
    _validate_decode_inputs(query, cache, block_table, seq_len)

    compute_dtype = torch.float32
    num_query_heads = query.shape[0]
    queries_per_kv_head = num_query_heads // cache.num_kv_heads
    scale = cache.head_size**-0.5

    output = torch.empty_like(query)
    for query_head in range(num_query_heads):
        kv_head = query_head // queries_per_kv_head
        q = query[query_head].to(compute_dtype)

        running_max = torch.tensor(
            float("-inf"),
            dtype=compute_dtype,
            device=query.device,
        )
        running_sum = torch.zeros(
            (),
            dtype=compute_dtype,
            device=query.device,
        )
        running_value = torch.zeros(
            (cache.head_size,),
            dtype=compute_dtype,
            device=query.device,
        )

        remaining_tokens = seq_len
        required_blocks = (
            seq_len + cache.tokens_per_page - 1
        ) // cache.tokens_per_page
        for logical_block in range(required_blocks):
            page_id = int(block_table[logical_block].item())
            tokens_in_page = min(remaining_tokens, cache.tokens_per_page)
            remaining_tokens -= tokens_in_page

            page_keys, page_values = _dequantize_page(
                cache,
                page_id,
                dtype=compute_dtype,
            )
            keys = page_keys[:tokens_in_page, kv_head]
            values = page_values[:tokens_in_page, kv_head]

            logits = torch.matmul(keys, q) * scale
            page_max = logits.max()
            next_max = torch.maximum(running_max, page_max)

            running_value = (
                running_value
                * torch.exp(running_max - next_max)
                + torch.sum(
                    torch.exp(logits - next_max)[:, None] * values,
                    dim=0,
                )
            )
            running_sum = (
                running_sum * torch.exp(running_max - next_max)
                + torch.exp(logits - next_max).sum()
            )
            running_max = next_max

        output[query_head] = (running_value / running_sum).to(query.dtype)

    return output


def dense_decode_attention_reference(
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
) -> torch.Tensor:
    """FP32 dense causal decode-attention oracle.

    Args:
        query: `[num_query_heads, head_size]`.
        keys: `[seq_len, num_kv_heads, head_size]`.
        values: Same shape as keys.
    """
    if keys.ndim != 3:
        raise ValueError(
            "keys must have shape [seq_len, num_kv_heads, head_size]; got "
            f"{tuple(keys.shape)}"
        )
    if values.shape != keys.shape:
        raise ValueError(
            f"values must have shape {tuple(keys.shape)}; got "
            f"{tuple(values.shape)}"
        )
    if query.ndim != 2:
        raise ValueError(
            f"query must be rank 2; got {tuple(query.shape)}"
        )
    if query.shape[1] != keys.shape[2]:
        raise ValueError("query and keys must have equal head size")
    if query.shape[0] % keys.shape[1] != 0:
        raise ValueError(
            "num_query_heads must be divisible by num_kv_heads"
        )
    if query.device != keys.device or values.device != keys.device:
        raise ValueError("query, keys, and values must share a device")

    compute_dtype = torch.float32
    num_query_heads = query.shape[0]
    num_kv_heads = keys.shape[1]
    queries_per_kv_head = num_query_heads // num_kv_heads
    scale = keys.shape[2]**-0.5

    outputs = []
    for query_head in range(num_query_heads):
        kv_head = query_head // queries_per_kv_head
        logits = (
            torch.matmul(
                keys[:, kv_head].to(compute_dtype),
                query[query_head].to(compute_dtype),
            )
            * scale
        )
        probs = torch.softmax(logits, dim=0)
        outputs.append(
            torch.matmul(
                probs,
                values[:, kv_head].to(compute_dtype),
            ).to(query.dtype)
        )
    return torch.stack(outputs)


try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _cachegen_int8_paged_decode_attention_kernel(
        query_ptr,
        key_pages_ptr,
        value_pages_ptr,
        key_scales_ptr,
        value_scales_ptr,
        block_table_ptr,
        output_ptr,
        seq_len,
        num_query_heads: tl.constexpr,
        num_kv_heads: tl.constexpr,
        head_size: tl.constexpr,
        tokens_per_page: tl.constexpr,
        queries_per_kv_head: tl.constexpr,
        query_stride_head: tl.constexpr,
        page_stride_page: tl.constexpr,
        page_stride_token: tl.constexpr,
        page_stride_kv_head: tl.constexpr,
        scale_stride_page: tl.constexpr,
        output_stride_head: tl.constexpr,
        BLOCK_TOKENS: tl.constexpr,
        BLOCK_HEAD_SIZE: tl.constexpr,
    ):
        query_head = tl.program_id(0)
        kv_head = query_head // queries_per_kv_head

        head_offsets = tl.arange(0, BLOCK_HEAD_SIZE)
        head_mask = head_offsets < head_size

        q_ptrs = (
            query_ptr
            + query_head * query_stride_head
            + head_offsets
        )
        q = tl.load(q_ptrs, mask=head_mask, other=0.0).to(tl.float32)

        running_max = -float("inf")
        running_sum = 0.0
        running_value = tl.zeros((BLOCK_HEAD_SIZE,), dtype=tl.float32)

        num_blocks = (
            seq_len + tokens_per_page - 1
        ) // tokens_per_page

        for logical_block in range(0, num_blocks):
            page_id = tl.load(block_table_ptr + logical_block).to(tl.int64)
            key_scale = tl.load(
                key_scales_ptr + page_id * scale_stride_page + kv_head
            ).to(tl.float32)
            value_scale = tl.load(
                value_scales_ptr + page_id * scale_stride_page + kv_head
            ).to(tl.float32)

            token_offsets = tl.arange(0, BLOCK_TOKENS)
            token_indices = logical_block * tokens_per_page + token_offsets
            token_mask = token_indices < seq_len
            page_offsets = token_indices - logical_block * tokens_per_page

            key_ptrs = (
                key_pages_ptr
                + page_id * page_stride_page
                + page_offsets[:, None] * page_stride_token
                + kv_head * page_stride_kv_head
                + head_offsets[None, :]
            )
            keys = tl.load(
                key_ptrs,
                mask=token_mask[:, None] & head_mask[None, :],
                other=0,
            ).to(tl.float32)
            keys *= key_scale

            logits = tl.sum(keys * q[None, :], axis=1)
            logits *= head_size**-0.5
            logits = tl.where(token_mask, logits, -float("inf"))

            block_max = tl.max(logits, axis=0)
            next_max = tl.maximum(running_max, block_max)
            alpha = tl.exp(running_max - next_max)
            weights = tl.exp(logits - next_max)

            value_ptrs = (
                value_pages_ptr
                + page_id * page_stride_page
                + page_offsets[:, None] * page_stride_token
                + kv_head * page_stride_kv_head
                + head_offsets[None, :]
            )
            values = tl.load(
                value_ptrs,
                mask=token_mask[:, None] & head_mask[None, :],
                other=0,
            ).to(tl.float32)
            values *= value_scale

            running_value = (
                running_value * alpha
                + tl.sum(weights[:, None] * values, axis=0)
            )
            running_sum = running_sum * alpha + tl.sum(weights, axis=0)
            running_max = next_max

        output_ptrs = (
            output_ptr
            + query_head * output_stride_head
            + head_offsets
        )
        tl.store(
            output_ptrs,
            running_value / running_sum,
            mask=head_mask,
        )


def cachegen_int8_paged_decode_attention_triton(
    query: torch.Tensor,
    cache: CacheGenInt8PagedKV,
    block_table: torch.Tensor,
    seq_len: int,
) -> torch.Tensor:
    """Fused paged INT8-KV decode attention for one query token.

    The kernel loads packed INT8 K/V vectors from paged storage, applies each
    page/KV-head scale in registers, and performs online-softmax attention.
    It intentionally supports only CUDA float16/bfloat16/float32 queries and
    a head size no larger than 256 during this initial integration stage.
    """
    _validate_decode_inputs(query, cache, block_table, seq_len)

    if triton is None:
        raise RuntimeError(
            "Triton is required for cachegen_int8_paged_decode_attention_triton"
        )
    if not query.is_cuda:
        raise ValueError("Triton decode attention requires CUDA tensors")
    if query.dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ):
        raise ValueError(
            "query must have dtype float16, bfloat16, or float32; got "
            f"{query.dtype}"
        )
    if cache.head_size > 256:
        raise ValueError(
            "Initial Triton decode kernel supports head_size <= 256; got "
            f"{cache.head_size}"
        )
    if cache.tokens_per_page > 256:
        raise ValueError(
            "Initial Triton decode kernel supports tokens_per_page <= 256; "
            f"got {cache.tokens_per_page}"
        )

    block_head_size = triton.next_power_of_2(cache.head_size)
    block_tokens = triton.next_power_of_2(cache.tokens_per_page)
    if block_head_size > 256 or block_tokens > 256:
        raise ValueError(
            "Initial Triton decode kernel requires power-of-two block sizes "
            "no larger than 256"
        )

    output = torch.empty_like(query)
    num_query_heads = query.shape[0]
    queries_per_kv_head = num_query_heads // cache.num_kv_heads

    _cachegen_int8_paged_decode_attention_kernel[
        (num_query_heads,)
    ](
        query,
        cache.key_pages,
        cache.value_pages,
        cache.key_scales,
        cache.value_scales,
        block_table,
        output,
        seq_len,
        num_query_heads=num_query_heads,
        num_kv_heads=cache.num_kv_heads,
        head_size=cache.head_size,
        tokens_per_page=cache.tokens_per_page,
        queries_per_kv_head=queries_per_kv_head,
        query_stride_head=query.stride(0),
        page_stride_page=cache.key_pages.stride(0),
        page_stride_token=cache.key_pages.stride(1),
        page_stride_kv_head=cache.key_pages.stride(2),
        scale_stride_page=cache.key_scales.stride(0),
        output_stride_head=output.stride(0),
        BLOCK_TOKENS=block_tokens,
        BLOCK_HEAD_SIZE=block_head_size,
        num_warps=4,
    )
    return output
