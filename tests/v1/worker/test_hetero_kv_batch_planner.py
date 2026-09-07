# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.v1.worker.experimental.hetero_kv_batch_planner import (
    HeteroKVBatchPlan,
    plan_hetero_kv_batch,
)
from vllm.v1.worker.experimental.hetero_kv_codec_registry import (
    DEFAULT_HETERO_KV_PAGE_BYTES,
    HeteroKVCodecId,
)


def test_mixed_batch_uses_registry_derived_tokens_per_page() -> None:
    quantizer_ids = (
        HeteroKVCodecId.CACHEGEN_INT8,
        HeteroKVCodecId.KIVI_K2_V2,
        HeteroKVCodecId.TURBOQUANT_K3_V2,
        HeteroKVCodecId.CACHEGEN_INT8,
    )

    batch_plan = plan_hetero_kv_batch(
        quantizer_ids=quantizer_ids,
        page_bytes=128 * 1024,
        num_kv_heads=8,
        head_size=128,
    )

    assert batch_plan.page_bytes == 128 * 1024
    assert batch_plan.num_active_rows == 4
    assert tuple(plan.codec_id for plan in batch_plan.codec_plans) == quantizer_ids
    assert batch_plan.tokens_per_page == tuple(
        plan.tokens_per_page for plan in batch_plan.codec_plans
    )
    assert len(set(batch_plan.tokens_per_page[:3])) == 3
    assert batch_plan.tokens_per_page[0] == batch_plan.tokens_per_page[3]


def test_default_page_size_is_128kib() -> None:
    batch_plan = plan_hetero_kv_batch(
        quantizer_ids=(HeteroKVCodecId.CACHEGEN_INT8,),
        num_kv_heads=8,
        head_size=128,
    )

    assert batch_plan.page_bytes == DEFAULT_HETERO_KV_PAGE_BYTES
    assert batch_plan.page_bytes == 128 * 1024


def test_page_bytes_are_configurable_for_every_active_row() -> None:
    smaller = plan_hetero_kv_batch(
        quantizer_ids=(
            HeteroKVCodecId.CACHEGEN_INT8,
            HeteroKVCodecId.KIVI_K2_V2,
            HeteroKVCodecId.TURBOQUANT_K3_V2,
        ),
        page_bytes=64 * 1024,
        num_kv_heads=8,
        head_size=128,
    )
    larger = plan_hetero_kv_batch(
        quantizer_ids=(
            HeteroKVCodecId.CACHEGEN_INT8,
            HeteroKVCodecId.KIVI_K2_V2,
            HeteroKVCodecId.TURBOQUANT_K3_V2,
        ),
        page_bytes=128 * 1024,
        num_kv_heads=8,
        head_size=128,
    )

    assert smaller.page_bytes == 64 * 1024
    assert larger.page_bytes == 128 * 1024
    assert all(
        larger_tokens >= smaller_tokens
        for smaller_tokens, larger_tokens in zip(
            smaller.tokens_per_page,
            larger.tokens_per_page,
            strict=True,
        )
    )


def test_empty_batch_is_valid() -> None:
    batch_plan = plan_hetero_kv_batch(
        quantizer_ids=(),
        num_kv_heads=8,
        head_size=128,
    )

    assert batch_plan.num_active_rows == 0
    assert batch_plan.codec_plans == ()
    assert batch_plan.tokens_per_page == ()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"quantizer_ids": (99,)}, "Unsupported hetero KV codec ID"),
        ({"page_bytes": 0}, "page_bytes must be positive"),
        ({"num_kv_heads": 0}, "num_kv_heads must be positive"),
        ({"head_size": 0}, "head_size must be positive"),
    ],
)
def test_batch_planner_rejects_invalid_inputs(
    kwargs: dict[str, object],
    message: str,
) -> None:
    values: dict[str, object] = {
        "quantizer_ids": (HeteroKVCodecId.CACHEGEN_INT8,),
        "page_bytes": 128 * 1024,
        "num_kv_heads": 8,
        "head_size": 128,
    }
    values.update(kwargs)

    with pytest.raises(ValueError, match=message):
        plan_hetero_kv_batch(**values)  # type: ignore[arg-type]


def test_batch_plan_rejects_mismatched_plan_lengths() -> None:
    plan = plan_hetero_kv_batch(
        quantizer_ids=(HeteroKVCodecId.CACHEGEN_INT8,),
        num_kv_heads=8,
        head_size=128,
    ).codec_plans[0]

    with pytest.raises(
        ValueError,
        match="codec_plans and tokens_per_page must have the same length",
    ):
        HeteroKVBatchPlan(
            page_bytes=128 * 1024,
            codec_plans=(plan,),
            tokens_per_page=(),
        )
