# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm import SamplingParams
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.request import Request, RequestStatus


def test_request_status_fmt_str():
    """Test that the string representation of RequestStatus is correct."""
    assert f"{RequestStatus.WAITING}" == "WAITING"
    assert f"{RequestStatus.WAITING_FOR_FSM}" == "WAITING_FOR_FSM"
    assert f"{RequestStatus.WAITING_FOR_REMOTE_KVS}" == "WAITING_FOR_REMOTE_KVS"
    assert f"{RequestStatus.RUNNING}" == "RUNNING"
    assert f"{RequestStatus.PREEMPTED}" == "PREEMPTED"
    assert f"{RequestStatus.FINISHED_STOPPED}" == "FINISHED_STOPPED"
    assert f"{RequestStatus.FINISHED_LENGTH_CAPPED}" == "FINISHED_LENGTH_CAPPED"
    assert f"{RequestStatus.FINISHED_ABORTED}" == "FINISHED_ABORTED"
    assert f"{RequestStatus.FINISHED_IGNORED}" == "FINISHED_IGNORED"


def make_engine_core_request(
        quantizer_id: int = 0) -> EngineCoreRequest:
    return EngineCoreRequest(
        request_id="test-request",
        prompt_token_ids=[1, 2, 3],
        mm_kwargs=None,
        mm_hashes=None,
        mm_placeholders=None,
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        eos_token_id=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
        quantizer_id=quantizer_id,
    )


def test_quantizer_id_propagates_from_engine_core_request():
    """Test that quantizer_id survives EngineCoreRequest conversion."""
    engine_request = make_engine_core_request(quantizer_id=1)

    request = Request.from_engine_core_request(
        engine_request,
        block_hasher=None,
    )

    assert request.quantizer_id == 1


def test_engine_core_request_quantizer_id_defaults_to_zero():
    """Test the backward-compatible quantizer_id default."""
    engine_request = make_engine_core_request()

    assert engine_request.quantizer_id == 0
