# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("VLLM_RUN_CACHEGEN_INT8_LIVE_TEST") != "1",
    reason=(
        "Set VLLM_RUN_CACHEGEN_INT8_LIVE_TEST=1 to run the "
        "GPU/model-backed CacheGen INT8 live smoke test."
    ),
)


def test_cachegen_int8_live_decode_validation() -> None:
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=os.environ.get(
            "VLLM_CACHEGEN_INT8_TEST_MODEL",
            "Qwen/Qwen2.5-0.5B-Instruct",
        ),
        dtype="bfloat16",
        tensor_parallel_size=1,
        enforce_eager=True,
        max_model_len=512,
        gpu_memory_utilization=0.60,
    )
    try:
        outputs = llm.generate(
            [
                "Explain why adaptive per-page INT8 scales reduce "
                "KV-cache quantization error."
            ],
            SamplingParams(
                temperature=0.0,
                max_tokens=32,
                ignore_eos=True,
            ),
        )

        assert outputs[0].outputs
        assert outputs[0].outputs[0].text

        results = llm.collective_rpc(
            "get_cachegen_int8_decode_validation_stats"
        )
        assert len(results) == 1

        stats = results[0]
        assert stats is not None
        assert stats["layer_name"]
        assert int(stats["callback_calls"]) > 0
        assert int(stats["route_eligible_calls"]) > 0
        assert int(stats["tokens_written"]) > 0
        assert int(stats["mapped_pages"]) > 0
        assert int(stats["int8_page_store_bytes"]) > 0
        assert int(stats["route_capacity_fallbacks"]) == 0
        assert stats["route_fallback_reasons"] == {
            "not_single_token_decode": 1,
        }
        assert int(stats["int8_output_calls"]) == 0
        assert int(stats["native_output_calls"]) == int(
            stats["route_eligible_calls"]
        )

        max_abs_error = float(stats["max_abs_error"])
        mean_abs_error = float(stats["mean_abs_error"])
        assert max_abs_error >= 0.0
        assert mean_abs_error >= 0.0
        assert max_abs_error < float("inf")
        assert mean_abs_error < float("inf")
    finally:
        del llm
