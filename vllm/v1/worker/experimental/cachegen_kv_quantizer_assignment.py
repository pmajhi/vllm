# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Per-sequence experimental paged-KV quantizer assignments."""

from __future__ import annotations

from dataclasses import dataclass

from vllm.v1.worker.experimental.cachegen_kv_quantizer_registry import (
    CacheGenKVQuantizerRegistry,
)
from vllm.v1.worker.experimental.cachegen_quantizer_types import (
    CacheGenKVQuantizer,
    parse_cachegen_kv_quantizer,
)


@dataclass(frozen=True)
class CacheGenKVQuantizerAssignment:
    """One request's selected paged-KV quantizer and route eligibility."""

    request_id: str
    quantizer: CacheGenKVQuantizer

    @property
    def quantizer_id(self) -> str:
        """Return a stable serialized quantizer identity."""
        return self.quantizer.value


def make_cachegen_kv_quantizer_assignment(
    *,
    request_id: str,
    quantizer: str | CacheGenKVQuantizer,
    registry: CacheGenKVQuantizerRegistry | None = None,
) -> CacheGenKVQuantizerAssignment:
    """Create a validated per-request quantizer assignment.

    ``native`` and vLLM-native ``fp8`` may be assigned without CacheGen
    page storage. ``int8_adaptive`` selects the implemented CacheGen page
    store/fused route. Unsupported experimental formats are rejected.
    """
    if not request_id:
        raise ValueError("request_id must not be empty")

    resolved_quantizer = (
        quantizer
        if isinstance(quantizer, CacheGenKVQuantizer)
        else parse_cachegen_kv_quantizer(quantizer)
    )

    active_registry = registry or CacheGenKVQuantizerRegistry()
    capabilities = active_registry.capabilities(resolved_quantizer)

    if not capabilities.supports_assignment:
        raise ValueError(
            "CacheGen KV quantizer is not implemented for per-sequence "
            f"assignment: {resolved_quantizer.value}"
        )

    return CacheGenKVQuantizerAssignment(
        request_id=request_id,
        quantizer=resolved_quantizer,
    )
