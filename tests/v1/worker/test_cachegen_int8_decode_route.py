# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.worker.experimental.cachegen_int8_decode_route import (
    get_cachegen_int8_decode_route_plan,
)


def make_inputs(
    *,
    sequence_length: int = 11,
) -> dict[str, object]:
    block_size = 16
    required_blocks = (sequence_length + block_size - 1) // block_size
    return {
        "num_actual_tokens": 1,
        "max_query_len": 1,
        "seq_lens": torch.tensor([sequence_length], dtype=torch.int32),
        "query_start_loc": torch.tensor([0, 1], dtype=torch.int32),
        "block_table": torch.tensor(
            [[1] * required_blocks],
            dtype=torch.int32,
        ),
        "block_size": block_size,
        "use_cascade": False,
        "sliding_window": (-1, -1),
        "has_kv_sharing": False,
    }


def test_route_plan_accepts_observed_qwen_decode_metadata() -> None:
    inputs = make_inputs(sequence_length=11)

    plan = get_cachegen_int8_decode_route_plan(**inputs)

    assert plan is not None
    assert plan.sequence_length == 11
    assert plan.block_table.tolist() == [1]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("num_actual_tokens", 2),
        ("max_query_len", 2),
        ("use_cascade", True),
        ("sliding_window", (15, 0)),
        ("has_kv_sharing", True),
    ],
)
def test_route_plan_rejects_unsupported_execution_modes(
    field: str,
    value: object,
) -> None:
    inputs = make_inputs()
    inputs[field] = value

    assert get_cachegen_int8_decode_route_plan(**inputs) is None


def test_route_plan_rejects_multiple_sequences() -> None:
    inputs = make_inputs()
    inputs["seq_lens"] = torch.tensor([11, 7], dtype=torch.int32)

    assert get_cachegen_int8_decode_route_plan(**inputs) is None


def test_route_plan_rejects_invalid_query_start_locations() -> None:
    inputs = make_inputs()
    inputs["query_start_loc"] = torch.tensor([0, 2], dtype=torch.int32)

    assert get_cachegen_int8_decode_route_plan(**inputs) is None


def test_route_plan_rejects_insufficient_block_table() -> None:
    inputs = make_inputs(sequence_length=17)
    inputs["block_table"] = torch.tensor([[1]], dtype=torch.int32)

    assert get_cachegen_int8_decode_route_plan(**inputs) is None


def test_route_plan_rejects_negative_block_id() -> None:
    inputs = make_inputs()
    inputs["block_table"] = torch.tensor([[-1]], dtype=torch.int32)

    assert get_cachegen_int8_decode_route_plan(**inputs) is None


def test_route_plan_rejects_nonpositive_block_size() -> None:
    inputs = make_inputs()
    inputs["block_size"] = 0

    with pytest.raises(ValueError, match="block_size must be positive"):
        get_cachegen_int8_decode_route_plan(**inputs)


def test_route_decision_reports_prefill_reason() -> None:
    from vllm.v1.worker.experimental.cachegen_int8_decode_route import (
        get_cachegen_int8_decode_route_decision,
    )

    decision = get_cachegen_int8_decode_route_decision(
        num_actual_tokens=4,
        max_query_len=4,
        seq_lens=torch.tensor([4], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 4], dtype=torch.int32),
        block_table=torch.tensor([[0]], dtype=torch.int32),
        block_size=16,
        use_cascade=False,
        sliding_window=(-1, -1),
        has_kv_sharing=False,
    )

    assert not decision.is_eligible
    assert decision.plan is None
    assert decision.reason == "not_single_token_decode"


def test_route_decision_reports_multi_sequence_reason() -> None:
    from vllm.v1.worker.experimental.cachegen_int8_decode_route import (
        get_cachegen_int8_decode_route_decision,
    )

    decision = get_cachegen_int8_decode_route_decision(
        num_actual_tokens=1,
        max_query_len=1,
        seq_lens=torch.tensor([1, 1], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
        block_table=torch.tensor([[0], [1]], dtype=torch.int32),
        block_size=16,
        use_cascade=False,
        sliding_window=(-1, -1),
        has_kv_sharing=False,
    )

    assert not decision.is_eligible
    assert decision.plan is None
    assert decision.reason == "not_single_sequence"
