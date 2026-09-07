# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.v1.worker.experimental.cachegen_quantizer_types import (
    CacheGenKVQuantizer,
    parse_cachegen_kv_quantizer,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("native", CacheGenKVQuantizer.NATIVE),
        ("INT8_ADAPTIVE", CacheGenKVQuantizer.INT8_ADAPTIVE),
        ("fp8", CacheGenKVQuantizer.FP8),
        ("kivi", CacheGenKVQuantizer.KIVI),
        ("turboquant", CacheGenKVQuantizer.TURBOQUANT),
        ("int4_groupwise", CacheGenKVQuantizer.INT4_GROUPWISE),
    ],
)
def test_parse_cachegen_kv_quantizer(
    value: str,
    expected: CacheGenKVQuantizer,
) -> None:
    assert parse_cachegen_kv_quantizer(value) is expected


def test_parse_cachegen_kv_quantizer_rejects_unknown_value() -> None:
    with pytest.raises(ValueError, match="Unknown CacheGen KV quantizer"):
        parse_cachegen_kv_quantizer("int2")
