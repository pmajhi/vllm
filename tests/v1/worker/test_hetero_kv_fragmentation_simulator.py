# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.v1.worker.experimental.hetero_kv_codec_registry import (
    HeteroKVCodecId,
)
from vllm.v1.worker.experimental.hetero_kv_fragmentation_simulator import (
    HeteroKVTraceRequest,
    PrivateHeteroKVPagePolicy,
    SharedHeteroKVPagePolicy,
    pages_required,
    resolve_trace_plans,
    simulate_hetero_kv_trace,
)


PAGE_BYTES = 128 * 1024
NUM_KV_HEADS = 8
HEAD_SIZE = 128


def test_pages_required_rounds_up() -> None:
    assert pages_required(cached_tokens=1, tokens_per_page=16) == 1
    assert pages_required(cached_tokens=16, tokens_per_page=16) == 1
    assert pages_required(cached_tokens=17, tokens_per_page=16) == 2


def test_shared_pool_admits_request_stranded_by_private_partition() -> None:
    plans = resolve_trace_plans(
        codec_ids=(
            HeteroKVCodecId.CACHEGEN_INT8,
            HeteroKVCodecId.KIVI_K2_V2,
        ),
        page_bytes=PAGE_BYTES,
        num_kv_heads=NUM_KV_HEADS,
        head_size=HEAD_SIZE,
    )
    cachegen_tokens = plans[
        HeteroKVCodecId.CACHEGEN_INT8
    ].tokens_per_page + 1
    kivi_tokens = plans[
        HeteroKVCodecId.KIVI_K2_V2
    ].tokens_per_page + 1

    requests = (
        HeteroKVTraceRequest(
            request_id="cachegen-0",
            arrival_step=0,
            departure_step=4,
            cached_tokens=cachegen_tokens,
            codec_id=HeteroKVCodecId.CACHEGEN_INT8,
        ),
        HeteroKVTraceRequest(
            request_id="kivi-0",
            arrival_step=1,
            departure_step=3,
            cached_tokens=kivi_tokens,
            codec_id=HeteroKVCodecId.KIVI_K2_V2,
        ),
    )

    private_result = simulate_hetero_kv_trace(
        requests=requests,
        policy=PrivateHeteroKVPagePolicy(
            pages_by_codec={
                HeteroKVCodecId.CACHEGEN_INT8: 2,
                HeteroKVCodecId.KIVI_K2_V2: 1,
                HeteroKVCodecId.TURBOQUANT_K3_V2: 1,
            },
            page_bytes=PAGE_BYTES,
        ),
        page_bytes=PAGE_BYTES,
        num_kv_heads=NUM_KV_HEADS,
        head_size=HEAD_SIZE,
    )
    shared_result = simulate_hetero_kv_trace(
        requests=requests,
        policy=SharedHeteroKVPagePolicy(
            total_pages=4,
            page_bytes=PAGE_BYTES,
        ),
        page_bytes=PAGE_BYTES,
        num_kv_heads=NUM_KV_HEADS,
        head_size=HEAD_SIZE,
    )

    assert private_result.total_pages == shared_result.total_pages == 4
    assert private_result.admitted_requests == 1
    assert private_result.rejected_requests == 1
    assert shared_result.admitted_requests == 2
    assert shared_result.rejected_requests == 0
    assert any(
        sample.stranded_private_pages > 0
        for sample in private_result.samples
    )


def test_shared_pool_records_cross_codec_reuse() -> None:
    requests = (
        HeteroKVTraceRequest(
            request_id="cachegen-0",
            arrival_step=0,
            departure_step=1,
            cached_tokens=1,
            codec_id=HeteroKVCodecId.CACHEGEN_INT8,
        ),
        HeteroKVTraceRequest(
            request_id="turbo-0",
            arrival_step=1,
            departure_step=2,
            cached_tokens=1,
            codec_id=HeteroKVCodecId.TURBOQUANT_K3_V2,
        ),
    )

    result = simulate_hetero_kv_trace(
        requests=requests,
        policy=SharedHeteroKVPagePolicy(
            total_pages=1,
            page_bytes=PAGE_BYTES,
        ),
        page_bytes=PAGE_BYTES,
        num_kv_heads=NUM_KV_HEADS,
        head_size=HEAD_SIZE,
    )

    assert result.admitted_requests == 2
    assert result.rejected_requests == 0
    assert result.cross_codec_reuse_events == 1


def test_external_state_is_reported_outside_compressed_page_pool() -> None:
    requests = (
        HeteroKVTraceRequest(
            request_id="cachegen-0",
            arrival_step=0,
            departure_step=2,
            cached_tokens=1,
            codec_id=HeteroKVCodecId.CACHEGEN_INT8,
        ),
        HeteroKVTraceRequest(
            request_id="kivi-0",
            arrival_step=0,
            departure_step=2,
            cached_tokens=1,
            codec_id=HeteroKVCodecId.KIVI_K2_V2,
        ),
        HeteroKVTraceRequest(
            request_id="turbo-0",
            arrival_step=0,
            departure_step=2,
            cached_tokens=1,
            codec_id=HeteroKVCodecId.TURBOQUANT_K3_V2,
        ),
    )

    result = simulate_hetero_kv_trace(
        requests=requests,
        policy=SharedHeteroKVPagePolicy(
            total_pages=3,
            page_bytes=PAGE_BYTES,
        ),
        page_bytes=PAGE_BYTES,
        num_kv_heads=NUM_KV_HEADS,
        head_size=HEAD_SIZE,
    )

    active_sample = result.samples[0]
    assert active_sample.live_pages == 3
    assert active_sample.allocated_page_bytes == 3 * PAGE_BYTES
    assert active_sample.external_bytes > 0


def test_requests_released_at_step_are_reusable_by_same_step_arrival() -> None:
    requests = (
        HeteroKVTraceRequest(
            request_id="first",
            arrival_step=0,
            departure_step=1,
            cached_tokens=1,
            codec_id=HeteroKVCodecId.CACHEGEN_INT8,
        ),
        HeteroKVTraceRequest(
            request_id="second",
            arrival_step=1,
            departure_step=2,
            cached_tokens=1,
            codec_id=HeteroKVCodecId.KIVI_K2_V2,
        ),
    )

    result = simulate_hetero_kv_trace(
        requests=requests,
        policy=SharedHeteroKVPagePolicy(
            total_pages=1,
            page_bytes=PAGE_BYTES,
        ),
        page_bytes=PAGE_BYTES,
        num_kv_heads=NUM_KV_HEADS,
        head_size=HEAD_SIZE,
    )

    assert result.admitted_requests == 2
    assert result.rejected_requests == 0


def test_private_stranded_pages_keep_step_maximum() -> None:
    plans = resolve_trace_plans(
        codec_ids=(
            HeteroKVCodecId.CACHEGEN_INT8,
            HeteroKVCodecId.KIVI_K2_V2,
            HeteroKVCodecId.TURBOQUANT_K3_V2,
        ),
        page_bytes=PAGE_BYTES,
        num_kv_heads=NUM_KV_HEADS,
        head_size=HEAD_SIZE,
    )
    two_kivi_pages = plans[
        HeteroKVCodecId.KIVI_K2_V2
    ].tokens_per_page + 1
    two_turboquant_pages = plans[
        HeteroKVCodecId.TURBOQUANT_K3_V2
    ].tokens_per_page + 1

    requests = (
        HeteroKVTraceRequest(
            request_id="cachegen-0",
            arrival_step=0,
            departure_step=3,
            cached_tokens=1,
            codec_id=HeteroKVCodecId.CACHEGEN_INT8,
        ),
        HeteroKVTraceRequest(
            request_id="kivi-0",
            arrival_step=1,
            departure_step=2,
            cached_tokens=two_kivi_pages,
            codec_id=HeteroKVCodecId.KIVI_K2_V2,
        ),
        HeteroKVTraceRequest(
            request_id="turbo-0",
            arrival_step=1,
            departure_step=2,
            cached_tokens=two_turboquant_pages,
            codec_id=HeteroKVCodecId.TURBOQUANT_K3_V2,
        ),
    )

    result = simulate_hetero_kv_trace(
        requests=requests,
        policy=PrivateHeteroKVPagePolicy(
            pages_by_codec={
                HeteroKVCodecId.CACHEGEN_INT8: 2,
                HeteroKVCodecId.KIVI_K2_V2: 1,
                HeteroKVCodecId.TURBOQUANT_K3_V2: 1,
            },
            page_bytes=PAGE_BYTES,
        ),
        page_bytes=PAGE_BYTES,
        num_kv_heads=NUM_KV_HEADS,
        head_size=HEAD_SIZE,
    )

    step_one = result.samples[1]
    assert step_one.rejected_requests == 2
    assert step_one.stranded_private_pages == 2
