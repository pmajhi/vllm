# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Batch-level planning helpers for heterogeneous fixed-byte KV codecs."""

from collections.abc import Sequence
from dataclasses import dataclass

from vllm.v1.worker.experimental.hetero_kv_codec_registry import (
    DEFAULT_HETERO_KV_PAGE_BYTES,
    HeteroKVCodecPlan,
    resolve_hetero_kv_codec_plan,
)


@dataclass(frozen=True)
class HeteroKVBatchPlan:
    """Resolved fixed-byte KV plans for the active rows of one batch.

    The order of ``codec_plans`` and ``tokens_per_page`` exactly matches the
    supplied active-row order. This helper only performs host-side planning;
    it does not alter standard VLLM KV allocation or attention execution.
    """

    page_bytes: int
    codec_plans: tuple[HeteroKVCodecPlan, ...]
    tokens_per_page: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.page_bytes <= 0:
            raise ValueError("page_bytes must be positive")
        if len(self.codec_plans) != len(self.tokens_per_page):
            raise ValueError(
                "codec_plans and tokens_per_page must have the same length"
            )
        if any(plan.page_bytes != self.page_bytes for plan in self.codec_plans):
            raise ValueError("all codec plans must use the batch page_bytes")
        if any(tokens <= 0 for tokens in self.tokens_per_page):
            raise ValueError("tokens_per_page values must be positive")

    @property
    def num_active_rows(self) -> int:
        return len(self.codec_plans)


def plan_hetero_kv_batch(
    *,
    quantizer_ids: Sequence[int],
    num_kv_heads: int,
    head_size: int,
    page_bytes: int = DEFAULT_HETERO_KV_PAGE_BYTES,
) -> HeteroKVBatchPlan:
    """Resolve active-row quantizer IDs into fixed-byte KV codec plans.

    ``quantizer_ids`` must be ordered in the same way as the active request
    rows that will consume the result. The helper intentionally has no GPU
    dependency so the scheduler/runner integration can validate geometry
    before experimental byte-page allocation is enabled.
    """
    if page_bytes <= 0:
        raise ValueError("page_bytes must be positive")
    if num_kv_heads <= 0:
        raise ValueError("num_kv_heads must be positive")
    if head_size <= 0:
        raise ValueError("head_size must be positive")

    codec_plans = tuple(
        resolve_hetero_kv_codec_plan(
            codec_id=quantizer_id,
            page_bytes=page_bytes,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
        )
        for quantizer_id in quantizer_ids
    )

    return HeteroKVBatchPlan(
        page_bytes=page_bytes,
        codec_plans=codec_plans,
        tokens_per_page=tuple(
            codec_plan.tokens_per_page for codec_plan in codec_plans
        ),
    )
