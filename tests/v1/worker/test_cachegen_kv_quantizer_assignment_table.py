# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.v1.worker.experimental.cachegen_kv_quantizer_assignment_table import (
    CacheGenKVQuantizerAssignmentTable,
)
from vllm.v1.worker.experimental.cachegen_quantizer_types import (
    CacheGenKVQuantizer,
)


def test_multiple_requests_can_use_different_assignments() -> None:
    table = CacheGenKVQuantizerAssignmentTable()

    table.assign(request_id="request-native", quantizer="native")
    table.assign(request_id="request-int8", quantizer="int8_adaptive")
    table.assign(request_id="request-fp8", quantizer="fp8")

    assert table.num_assignments == 3
    assert table.quantizer_for("request-native") is CacheGenKVQuantizer.NATIVE
    assert table.quantizer_for(
        "request-int8"
    ) is CacheGenKVQuantizer.INT8_ADAPTIVE
    assert table.quantizer_for("request-fp8") is CacheGenKVQuantizer.FP8
    assert table.snapshot() == {
        "request-native": "native",
        "request-int8": "int8_adaptive",
        "request-fp8": "fp8",
    }


def test_unknown_request_defaults_to_native() -> None:
    table = CacheGenKVQuantizerAssignmentTable()

    assert table.quantizer_for("unknown") is CacheGenKVQuantizer.NATIVE
    assert table.get("unknown") is None


def test_reassignment_replaces_existing_choice() -> None:
    table = CacheGenKVQuantizerAssignmentTable()

    table.assign(request_id="request-1", quantizer="native")
    replacement = table.assign(
        request_id="request-1",
        quantizer="int8_adaptive",
    )

    assert table.num_assignments == 1
    assert replacement.quantizer is CacheGenKVQuantizer.INT8_ADAPTIVE
    assert table.quantizer_for(
        "request-1"
    ) is CacheGenKVQuantizer.INT8_ADAPTIVE


def test_removal_releases_request_assignment() -> None:
    table = CacheGenKVQuantizerAssignmentTable()
    table.assign(request_id="request-1", quantizer="int8_adaptive")

    removed = table.remove("request-1")

    assert removed is not None
    assert removed.quantizer is CacheGenKVQuantizer.INT8_ADAPTIVE
    assert table.num_assignments == 0
    assert table.quantizer_for("request-1") is CacheGenKVQuantizer.NATIVE
