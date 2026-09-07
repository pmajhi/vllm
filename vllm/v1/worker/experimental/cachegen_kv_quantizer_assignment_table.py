# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Mutable request-to-quantizer assignments for experimental paged KV."""

from __future__ import annotations

from vllm.v1.worker.experimental.cachegen_kv_quantizer_assignment import (
    CacheGenKVQuantizerAssignment,
    make_cachegen_kv_quantizer_assignment,
)
from vllm.v1.worker.experimental.cachegen_kv_quantizer_registry import (
    CacheGenKVQuantizerRegistry,
)
from vllm.v1.worker.experimental.cachegen_quantizer_types import (
    CacheGenKVQuantizer,
)


class CacheGenKVQuantizerAssignmentTable:
    """Own active request quantizer assignments without global coupling."""

    def __init__(
        self,
        registry: CacheGenKVQuantizerRegistry | None = None,
    ) -> None:
        self._registry = registry or CacheGenKVQuantizerRegistry()
        self._assignments: dict[str, CacheGenKVQuantizerAssignment] = {}

    def assign(
        self,
        *,
        request_id: str,
        quantizer: str | CacheGenKVQuantizer,
    ) -> CacheGenKVQuantizerAssignment:
        """Create or replace one request's assignment."""
        assignment = make_cachegen_kv_quantizer_assignment(
            request_id=request_id,
            quantizer=quantizer,
            registry=self._registry,
        )
        self._assignments[request_id] = assignment
        return assignment

    def get(
        self,
        request_id: str,
    ) -> CacheGenKVQuantizerAssignment | None:
        """Return one request's assignment, if active."""
        return self._assignments.get(request_id)

    def remove(
        self,
        request_id: str,
    ) -> CacheGenKVQuantizerAssignment | None:
        """Remove and return one completed request's assignment."""
        return self._assignments.pop(request_id, None)

    def quantizer_for(
        self,
        request_id: str,
    ) -> CacheGenKVQuantizer:
        """Return the request quantizer, defaulting safely to native."""
        assignment = self.get(request_id)
        if assignment is None:
            return CacheGenKVQuantizer.NATIVE
        return assignment.quantizer

    @property
    def num_assignments(self) -> int:
        """Return the number of active request assignments."""
        return len(self._assignments)

    def snapshot(self) -> dict[str, str]:
        """Return a serializable request-to-quantizer view."""
        return {
            request_id: assignment.quantizer_id
            for request_id, assignment in self._assignments.items()
        }
