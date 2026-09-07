# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.v1.worker.experimental.hetero_kv_codec_registry import (
    DEFAULT_HETERO_KV_PAGE_BYTES,
    HeteroKVCodecId,
    resolve_hetero_kv_codec_plan,
    supported_hetero_kv_codec_ids,
)


def test_registry_lists_all_initial_codec_ids() -> None:
    assert supported_hetero_kv_codec_ids() == (
        HeteroKVCodecId.BASELINE,
        HeteroKVCodecId.UNIFORM_INT8,
        HeteroKVCodecId.CACHEGEN_INT8,
        HeteroKVCodecId.KIVI_K2_V2,
        HeteroKVCodecId.KIVI_K4_V4,
        HeteroKVCodecId.KIVI_K8_V8,
        HeteroKVCodecId.TURBOQUANT_K3_V2,
        HeteroKVCodecId.TURBOQUANT_K3_V4,
    )


@pytest.mark.parametrize("codec_id", list(HeteroKVCodecId))
def test_all_codec_plans_use_one_configurable_physical_page_size(
    codec_id: HeteroKVCodecId,
) -> None:
    plan = resolve_hetero_kv_codec_plan(
        codec_id=codec_id,
        page_bytes=128 * 1024,
        num_kv_heads=8,
        head_size=128,
    )

    assert plan.codec_id is codec_id
    assert plan.page_bytes == 128 * 1024
    assert plan.tokens_per_page > 0
    assert plan.external_bytes_per_sequence_per_layer >= 0


def test_registry_default_page_size_is_128kib() -> None:
    plan = resolve_hetero_kv_codec_plan(
        codec_id=HeteroKVCodecId.CACHEGEN_INT8,
        num_kv_heads=8,
        head_size=128,
    )

    assert DEFAULT_HETERO_KV_PAGE_BYTES == 128 * 1024
    assert plan.page_bytes == DEFAULT_HETERO_KV_PAGE_BYTES


def test_page_size_is_configurable() -> None:
    smaller = resolve_hetero_kv_codec_plan(
        codec_id=HeteroKVCodecId.CACHEGEN_INT8,
        page_bytes=64 * 1024,
        num_kv_heads=8,
        head_size=128,
    )
    larger = resolve_hetero_kv_codec_plan(
        codec_id=HeteroKVCodecId.CACHEGEN_INT8,
        page_bytes=128 * 1024,
        num_kv_heads=8,
        head_size=128,
    )

    assert smaller.page_bytes == 64 * 1024
    assert larger.page_bytes == 128 * 1024
    assert larger.tokens_per_page >= smaller.tokens_per_page


def test_codec_families_have_distinct_page_capacities() -> None:
    plans = [
        resolve_hetero_kv_codec_plan(
            codec_id=codec_id,
            page_bytes=128 * 1024,
            num_kv_heads=8,
            head_size=128,
        )
        for codec_id in (
            HeteroKVCodecId.CACHEGEN_INT8,
            HeteroKVCodecId.KIVI_K2_V2,
            HeteroKVCodecId.TURBOQUANT_K3_V2,
        )
    ]

    assert len({plan.tokens_per_page for plan in plans}) == len(plans)


def test_kivi_and_turboquant_external_state_is_reported_separately() -> None:
    cachegen = resolve_hetero_kv_codec_plan(
        codec_id=HeteroKVCodecId.CACHEGEN_INT8,
        num_kv_heads=8,
        head_size=128,
    )
    kivi = resolve_hetero_kv_codec_plan(
        codec_id=HeteroKVCodecId.KIVI_K2_V2,
        num_kv_heads=8,
        head_size=128,
    )
    turboquant = resolve_hetero_kv_codec_plan(
        codec_id=HeteroKVCodecId.TURBOQUANT_K3_V2,
        num_kv_heads=8,
        head_size=128,
    )

    assert cachegen.external_bytes_per_sequence_per_layer == 0
    assert kivi.external_bytes_per_sequence_per_layer == 131_072
    assert turboquant.external_bytes_per_sequence_per_layer == 524_288


def test_kivi_and_turboquant_plan_parameters_are_explicit() -> None:
    kivi = resolve_hetero_kv_codec_plan(
        codec_id=HeteroKVCodecId.KIVI_K4_V4,
        num_kv_heads=8,
        head_size=128,
    )
    turboquant = resolve_hetero_kv_codec_plan(
        codec_id=HeteroKVCodecId.TURBOQUANT_K3_V4,
        num_kv_heads=8,
        head_size=128,
    )

    assert (kivi.key_bits, kivi.value_bits) == (4, 4)
    assert kivi.key_group_size == 32
    assert kivi.value_group_size == 32
    assert kivi.residual_tokens == 32

    assert (turboquant.key_bits, turboquant.value_bits) == (3, 4)
    assert turboquant.value_group_size == 32
    assert turboquant.ring_capacity == 128


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"codec_id": 99}, "Unsupported hetero KV codec ID"),
        ({"page_bytes": 0}, "page_bytes must be positive"),
        ({"num_kv_heads": 0}, "num_kv_heads must be positive"),
        ({"head_size": 0}, "head_size must be positive"),
    ],
)
def test_registry_rejects_invalid_inputs(
    kwargs: dict[str, int],
    message: str,
) -> None:
    values = {
        "codec_id": HeteroKVCodecId.CACHEGEN_INT8,
        "page_bytes": 128 * 1024,
        "num_kv_heads": 8,
        "head_size": 128,
    }
    values.update(kwargs)

    with pytest.raises(ValueError, match=message):
        resolve_hetero_kv_codec_plan(**values)
