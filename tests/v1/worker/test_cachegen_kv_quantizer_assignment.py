# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.v1.worker.experimental.cachegen_kv_quantizer_assignment import (
    CacheGenKVQuantizerAssignment,
    make_cachegen_kv_quantizer_assignment,
)
from vllm.v1.worker.experimental.cachegen_quantizer_types import (
    CacheGenKVQuantizer,
)


def test_native_assignment_is_accepted() -> None:
    assignment = make_cachegen_kv_quantizer_assignment(
        request_id="request-native",
        quantizer="native",
    )

    assert assignment == CacheGenKVQuantizerAssignment(
        request_id="request-native",
        quantizer=CacheGenKVQuantizer.NATIVE,
    )
    assert assignment.quantizer_id == "native"


def test_int8_adaptive_assignment_is_accepted() -> None:
    assignment = make_cachegen_kv_quantizer_assignment(
        request_id="request-int8",
        quantizer="int8_adaptive",
    )

    assert assignment.quantizer is CacheGenKVQuantizer.INT8_ADAPTIVE
    assert assignment.quantizer_id == "int8_adaptive"


def test_fp8_assignment_is_accepted_without_cachegen_pages() -> None:
    assignment = make_cachegen_kv_quantizer_assignment(
        request_id="request-fp8",
        quantizer="fp8",
    )

    assert assignment.quantizer is CacheGenKVQuantizer.FP8
    assert assignment.quantizer_id == "fp8"


@pytest.mark.parametrize(
    "quantizer",
    [
        "kivi",
        "turboquant",
        "int4_groupwise",
    ],
)
def test_unimplemented_quantizer_assignment_is_rejected(
    quantizer: str,
) -> None:
    with pytest.raises(ValueError, match="not implemented"):
        make_cachegen_kv_quantizer_assignment(
            request_id="request-unimplemented",
            quantizer=quantizer,
        )


def test_empty_request_id_is_rejected() -> None:
    with pytest.raises(ValueError, match="request_id must not be empty"):
        make_cachegen_kv_quantizer_assignment(
            request_id="",
            quantizer="native",
        )


def test_enum_quantizer_assignment_is_accepted() -> None:
    assignment = make_cachegen_kv_quantizer_assignment(
        request_id="request-enum",
        quantizer=CacheGenKVQuantizer.INT8_ADAPTIVE,
    )

    assert assignment.quantizer is CacheGenKVQuantizer.INT8_ADAPTIVE
