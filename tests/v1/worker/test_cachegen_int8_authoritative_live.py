# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("VLLM_RUN_CACHEGEN_INT8_LIVE_TEST") != "1",
    reason=(
        "Set VLLM_RUN_CACHEGEN_INT8_LIVE_TEST=1 to run the GPU/model-backed "
        "CacheGen INT8 authoritative decode test."
    ),
)

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPT = (
    "Explain why adaptive per-page INT8 scales can reduce KV-cache "
    "quantization error."
)


def _run(mode: str) -> tuple[list[int], str, dict[str, object]]:
    from vllm import LLM, SamplingParams

    os.environ["VLLM_EXPERIMENTAL_CACHEGEN_INT8_ATTENTION_MODE"] = mode

    llm = LLM(
        model=os.environ.get("VLLM_CACHEGEN_INT8_TEST_MODEL", MODEL),
        dtype="bfloat16",
        tensor_parallel_size=1,
        enforce_eager=True,
        max_model_len=512,
        gpu_memory_utilization=0.60,
    )
    try:
        result = llm.generate(
            [PROMPT],
            SamplingParams(
                temperature=0.0,
                max_tokens=32,
                ignore_eos=True,
            ),
        )[0].outputs[0]

        stats = llm.collective_rpc(
            "get_cachegen_int8_decode_validation_stats"
        )[0]
        assert stats is not None
        return list(result.token_ids), result.text, stats
    finally:
        del llm


def _assert_common_route_properties(stats: dict[str, object]) -> None:
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

    max_abs_error = float(stats["max_abs_error"])
    mean_abs_error = float(stats["mean_abs_error"])
    assert max_abs_error >= 0.0
    assert mean_abs_error >= 0.0
    assert max_abs_error < float("inf")
    assert mean_abs_error < float("inf")


def test_cachegen_int8_authoritative_decode_matches_validate() -> None:
    validate_ids, validate_text, validate_stats = _run("validate")
    int8_ids, int8_text, int8_stats = _run("int8")

    _assert_common_route_properties(validate_stats)
    _assert_common_route_properties(int8_stats)

    assert int(validate_stats["int8_output_calls"]) == 0
    assert int(validate_stats["native_output_calls"]) == int(
        validate_stats["route_eligible_calls"]
    )

    assert int(int8_stats["native_output_calls"]) == 0
    assert int(int8_stats["int8_output_calls"]) == int(
        int8_stats["route_eligible_calls"]
    )

    assert int8_ids == validate_ids
    assert int8_text == validate_text
