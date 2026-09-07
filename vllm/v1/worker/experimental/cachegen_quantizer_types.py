# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Common user-facing identities for experimental paged KV quantizers."""

from __future__ import annotations

from enum import Enum


class CacheGenKVQuantizer(str, Enum):
    """KV-cache storage/attention formats selectable per sequence."""

    NATIVE = "native"
    INT8_ADAPTIVE = "int8_adaptive"
    FP8 = "fp8"
    KIVI = "kivi"
    TURBOQUANT = "turboquant"
    INT4_GROUPWISE = "int4_groupwise"


def parse_cachegen_kv_quantizer(value: str) -> CacheGenKVQuantizer:
    """Parse a stable user-facing experimental KV quantizer identity."""
    try:
        return CacheGenKVQuantizer(value.strip().lower())
    except ValueError as exc:
        valid = ", ".join(item.value for item in CacheGenKVQuantizer)
        raise ValueError(
            f"Unknown CacheGen KV quantizer {value!r}; expected one of {valid}"
        ) from exc
