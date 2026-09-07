# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Strict eligibility checks for experimental fused INT8 decode attention."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CacheGenInt8DecodeRoutePlan:
    """Validated inputs for one-sequence, one-token decode attention."""

    sequence_length: int
    block_table: torch.Tensor


@dataclass(frozen=True)
class CacheGenInt8DecodeRouteDecision:
    """Either an eligible plan or a stable native-fallback reason."""

    plan: CacheGenInt8DecodeRoutePlan | None
    reason: str | None

    @property
    def is_eligible(self) -> bool:
        return self.plan is not None


def get_cachegen_int8_decode_route_decision(
    *,
    num_actual_tokens: int,
    max_query_len: int,
    seq_lens: torch.Tensor,
    query_start_loc: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    use_cascade: bool,
    sliding_window: tuple[int, int],
    has_kv_sharing: bool,
) -> CacheGenInt8DecodeRouteDecision:
    """Classify the strict initial fused-INT8 decode route.

    The initial route supports exactly one request with one query token.
    Every ineligible shape returns a stable reason so callers can count
    native fallbacks without affecting native FlashAttention output.
    """
    if num_actual_tokens != 1 or max_query_len != 1:
        return CacheGenInt8DecodeRouteDecision(
            plan=None,
            reason="not_single_token_decode",
        )
    if use_cascade:
        return CacheGenInt8DecodeRouteDecision(
            plan=None,
            reason="cascade_attention",
        )
    if sliding_window != (-1, -1):
        return CacheGenInt8DecodeRouteDecision(
            plan=None,
            reason="sliding_window_attention",
        )
    if has_kv_sharing:
        return CacheGenInt8DecodeRouteDecision(
            plan=None,
            reason="kv_sharing",
        )
    if block_size <= 0:
        raise ValueError(f"block_size must be positive; got {block_size}")
    if seq_lens.ndim != 1 or seq_lens.numel() != 1:
        return CacheGenInt8DecodeRouteDecision(
            plan=None,
            reason="not_single_sequence",
        )
    if query_start_loc.ndim != 1 or query_start_loc.numel() != 2:
        return CacheGenInt8DecodeRouteDecision(
            plan=None,
            reason="invalid_query_start_loc_shape",
        )
    if int(query_start_loc[0].item()) != 0:
        return CacheGenInt8DecodeRouteDecision(
            plan=None,
            reason="invalid_query_start_loc_start",
        )
    if int(query_start_loc[1].item()) != 1:
        return CacheGenInt8DecodeRouteDecision(
            plan=None,
            reason="invalid_query_start_loc_end",
        )

    sequence_length = int(seq_lens[0].item())
    if sequence_length <= 0:
        return CacheGenInt8DecodeRouteDecision(
            plan=None,
            reason="nonpositive_sequence_length",
        )
    if block_table.ndim != 2 or block_table.shape[0] != 1:
        return CacheGenInt8DecodeRouteDecision(
            plan=None,
            reason="invalid_block_table_shape",
        )

    required_blocks = (sequence_length + block_size - 1) // block_size
    if block_table.shape[1] < required_blocks:
        return CacheGenInt8DecodeRouteDecision(
            plan=None,
            reason="insufficient_block_table",
        )

    active_blocks = block_table[0, :required_blocks]
    if active_blocks.dtype not in (torch.int32, torch.int64):
        return CacheGenInt8DecodeRouteDecision(
            plan=None,
            reason="invalid_block_table_dtype",
        )
    if torch.any(active_blocks < 0):
        return CacheGenInt8DecodeRouteDecision(
            plan=None,
            reason="negative_active_block_id",
        )

    return CacheGenInt8DecodeRouteDecision(
        plan=CacheGenInt8DecodeRoutePlan(
            sequence_length=sequence_length,
            block_table=active_blocks,
        ),
        reason=None,
    )


def get_cachegen_int8_decode_route_plan(
    *,
    num_actual_tokens: int,
    max_query_len: int,
    seq_lens: torch.Tensor,
    query_start_loc: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    use_cascade: bool,
    sliding_window: tuple[int, int],
    has_kv_sharing: bool,
) -> CacheGenInt8DecodeRoutePlan | None:
    """Return a route plan only for the initial supported decode shape."""
    return get_cachegen_int8_decode_route_decision(
        num_actual_tokens=num_actual_tokens,
        max_query_len=max_query_len,
        seq_lens=seq_lens,
        query_start_loc=query_start_loc,
        block_table=block_table,
        block_size=block_size,
        use_cascade=use_cascade,
        sliding_window=sliding_window,
        has_kv_sharing=has_kv_sharing,
    ).plan
