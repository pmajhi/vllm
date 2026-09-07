# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Trace-driven fixed-byte KV-page allocator simulator.

This module intentionally models only compressed fixed-byte page allocation.
Codec-specific external state, such as KIVI residuals and TurboQuant rings, is
reported separately and is not charged to the shared compressed-page pool.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from math import ceil
from typing import Iterable

from vllm.v1.worker.experimental.hetero_fixed_byte_page_pool import (
    HeteroFixedBytePagePool,
)
from vllm.v1.worker.experimental.hetero_kv_codec_registry import (
    HeteroKVCodecId,
    HeteroKVCodecPlan,
    resolve_hetero_kv_codec_plan,
)


@dataclass(frozen=True)
class HeteroKVTraceRequest:
    """One synthetic or replayed request in discrete simulator time."""

    request_id: str
    arrival_step: int
    departure_step: int
    cached_tokens: int
    codec_id: HeteroKVCodecId

    def __post_init__(self) -> None:
        if self.arrival_step < 0:
            raise ValueError("arrival_step must be nonnegative")
        if self.departure_step <= self.arrival_step:
            raise ValueError("departure_step must exceed arrival_step")
        if self.cached_tokens <= 0:
            raise ValueError("cached_tokens must be positive")


@dataclass(frozen=True)
class HeteroKVRequestAllocation:
    """Resolved shared-page allocation and external-state accounting."""

    request: HeteroKVTraceRequest
    plan: HeteroKVCodecPlan
    page_ids: tuple[int, ...]

    @property
    def compressed_page_count(self) -> int:
        return len(self.page_ids)

    @property
    def external_bytes(self) -> int:
        return self.plan.external_bytes_per_sequence_per_layer


@dataclass(frozen=True)
class HeteroKVSimulationSample:
    """One time-step sample suitable for CSV export and plotting."""

    step: int
    active_requests: int
    admitted_requests: int
    rejected_requests: int
    live_pages: int
    free_pages: int
    allocated_page_bytes: int
    free_page_bytes: int
    external_bytes: int
    stranded_private_pages: int
    cross_codec_reuse_events: int


@dataclass(frozen=True)
class HeteroKVSimulationResult:
    """Aggregate and time-series results for one allocator policy."""

    policy_name: str
    page_bytes: int
    total_pages: int
    total_arrivals: int
    admitted_requests: int
    rejected_requests: int
    cross_codec_reuse_events: int
    samples: tuple[HeteroKVSimulationSample, ...]

    @property
    def admission_rate(self) -> float:
        if self.total_arrivals == 0:
            return 1.0
        return self.admitted_requests / self.total_arrivals

    @property
    def peak_live_pages(self) -> int:
        return max((sample.live_pages for sample in self.samples), default=0)

    @property
    def mean_live_pages(self) -> float:
        if not self.samples:
            return 0.0
        return sum(sample.live_pages for sample in self.samples) / len(
            self.samples
        )

    @property
    def peak_external_bytes(self) -> int:
        return max(
            (sample.external_bytes for sample in self.samples),
            default=0,
        )

    @property
    def mean_external_bytes(self) -> float:
        if not self.samples:
            return 0.0
        return sum(sample.external_bytes for sample in self.samples) / len(
            self.samples
        )


def pages_required(
    *,
    cached_tokens: int,
    tokens_per_page: int,
) -> int:
    """Return compressed fixed-byte pages needed for a request."""
    if cached_tokens <= 0:
        raise ValueError("cached_tokens must be positive")
    if tokens_per_page <= 0:
        raise ValueError("tokens_per_page must be positive")
    return ceil(cached_tokens / tokens_per_page)


def resolve_trace_plans(
    *,
    codec_ids: Iterable[HeteroKVCodecId],
    page_bytes: int,
    num_kv_heads: int,
    head_size: int,
) -> dict[HeteroKVCodecId, HeteroKVCodecPlan]:
    """Resolve each codec used by a trace exactly once."""
    return {
        codec_id: resolve_hetero_kv_codec_plan(
            codec_id=codec_id,
            page_bytes=page_bytes,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
        )
        for codec_id in set(codec_ids)
    }


class SharedHeteroKVPagePolicy:
    """One physical fixed-byte pool shared by all codec families."""

    name = "shared"

    def __init__(self, *, total_pages: int, page_bytes: int) -> None:
        self.page_bytes = page_bytes
        self.pool = HeteroFixedBytePagePool(
            num_pages=total_pages,
            page_bytes=page_bytes,
        )
        self._last_codec_by_page_id: dict[int, HeteroKVCodecId] = {}
        self.cross_codec_reuse_events = 0

    @property
    def total_pages(self) -> int:
        return self.pool.num_pages

    @property
    def live_pages(self) -> int:
        return self.pool.num_allocated_pages

    @property
    def free_pages(self) -> int:
        return self.pool.num_free_pages

    @property
    def stranded_private_pages(self) -> int:
        return 0

    def allocate(
        self,
        *,
        codec_id: HeteroKVCodecId,
        num_pages: int,
    ) -> tuple[int, ...] | None:
        if num_pages > self.pool.num_free_pages:
            return None

        page_ids = tuple(self.pool.allocate_many(num_pages))
        for page_id in page_ids:
            previous_codec = self._last_codec_by_page_id.get(page_id)
            if previous_codec is not None and previous_codec != codec_id:
                self.cross_codec_reuse_events += 1
            self._last_codec_by_page_id[page_id] = codec_id
        return page_ids

    def free(
        self,
        *,
        codec_id: HeteroKVCodecId,
        page_ids: tuple[int, ...],
    ) -> None:
        del codec_id
        self.pool.free_many(list(page_ids))


class PrivateHeteroKVPagePolicy:
    """Codec-private fixed-byte pools with identical total page budget.

    ``pages_by_codec`` is intentionally explicit. This avoids hiding a
    partitioning policy inside the baseline and enables sensitivity experiments
    over equal, demand-proportional, and deliberately mismatched partitions.
    """

    name = "private"

    def __init__(
        self,
        *,
        pages_by_codec: dict[HeteroKVCodecId, int],
        page_bytes: int,
    ) -> None:
        if not pages_by_codec:
            raise ValueError("pages_by_codec must not be empty")
        if any(num_pages <= 0 for num_pages in pages_by_codec.values()):
            raise ValueError("every codec-private pool must have positive size")

        self.page_bytes = page_bytes
        self.pools = {
            codec_id: HeteroFixedBytePagePool(
                num_pages=num_pages,
                page_bytes=page_bytes,
            )
            for codec_id, num_pages in pages_by_codec.items()
        }
        self._stranded_private_pages = 0

    @property
    def total_pages(self) -> int:
        return sum(pool.num_pages for pool in self.pools.values())

    @property
    def live_pages(self) -> int:
        return sum(pool.num_allocated_pages for pool in self.pools.values())

    @property
    def free_pages(self) -> int:
        return sum(pool.num_free_pages for pool in self.pools.values())

    @property
    def stranded_private_pages(self) -> int:
        """Free pages unavailable to a currently rejected codec are sampled.

        The simulator sets this value at each admission failure. It measures
        free capacity in *other* private pools that could have served the
        request under a shared pool.
        """
        return self._stranded_private_pages

    @property
    def cross_codec_reuse_events(self) -> int:
        return 0

    def reset_step_metrics(self) -> None:
        self._stranded_private_pages = 0

    def allocate(
        self,
        *,
        codec_id: HeteroKVCodecId,
        num_pages: int,
    ) -> tuple[int, ...] | None:
        pool = self.pools.get(codec_id)
        if pool is None:
            raise ValueError(
                f"No codec-private pool is configured for {codec_id.name}"
            )
        if num_pages > pool.num_free_pages:
            stranded_pages = self.free_pages - pool.num_free_pages
            self._stranded_private_pages = max(
                self._stranded_private_pages,
                stranded_pages,
            )
            return None
        return tuple(pool.allocate_many(num_pages))

    def free(
        self,
        *,
        codec_id: HeteroKVCodecId,
        page_ids: tuple[int, ...],
    ) -> None:
        self.pools[codec_id].free_many(list(page_ids))


def simulate_hetero_kv_trace(
    *,
    requests: Iterable[HeteroKVTraceRequest],
    policy: SharedHeteroKVPagePolicy | PrivateHeteroKVPagePolicy,
    page_bytes: int,
    num_kv_heads: int,
    head_size: int,
) -> HeteroKVSimulationResult:
    """Replay a discrete request trace through one page-allocation policy.

    At each step, requests whose departure time equals the step are released
    before arrivals at that step are admitted. This convention makes pages
    freed at a step immediately usable by an arrival at the same step.
    """
    requests = tuple(requests)
    if not requests:
        return HeteroKVSimulationResult(
            policy_name=policy.name,
            page_bytes=page_bytes,
            total_pages=policy.total_pages,
            total_arrivals=0,
            admitted_requests=0,
            rejected_requests=0,
            cross_codec_reuse_events=0,
            samples=(),
        )

    plans = resolve_trace_plans(
        codec_ids=(request.codec_id for request in requests),
        page_bytes=page_bytes,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
    )
    arrivals: dict[int, list[HeteroKVTraceRequest]] = defaultdict(list)
    departures: dict[int, list[str]] = defaultdict(list)
    for request in requests:
        arrivals[request.arrival_step].append(request)

    active: dict[str, HeteroKVRequestAllocation] = {}
    total_arrivals = 0
    admitted_requests = 0
    rejected_requests = 0
    samples: list[HeteroKVSimulationSample] = []

    for step in range(max(request.departure_step for request in requests) + 1):
        if isinstance(policy, PrivateHeteroKVPagePolicy):
            policy.reset_step_metrics()

        for request_id in departures.pop(step, []):
            allocation = active.pop(request_id)
            policy.free(
                codec_id=allocation.request.codec_id,
                page_ids=allocation.page_ids,
            )

        for request in arrivals.get(step, []):
            total_arrivals += 1
            plan = plans[request.codec_id]
            requested_pages = pages_required(
                cached_tokens=request.cached_tokens,
                tokens_per_page=plan.tokens_per_page,
            )
            page_ids = policy.allocate(
                codec_id=request.codec_id,
                num_pages=requested_pages,
            )
            if page_ids is None:
                rejected_requests += 1
                continue

            admitted_requests += 1
            active[request.request_id] = HeteroKVRequestAllocation(
                request=request,
                plan=plan,
                page_ids=page_ids,
            )
            departures[request.departure_step].append(request.request_id)

        external_bytes = sum(
            allocation.external_bytes for allocation in active.values()
        )
        samples.append(
            HeteroKVSimulationSample(
                step=step,
                active_requests=len(active),
                admitted_requests=admitted_requests,
                rejected_requests=rejected_requests,
                live_pages=policy.live_pages,
                free_pages=policy.free_pages,
                allocated_page_bytes=policy.live_pages * page_bytes,
                free_page_bytes=policy.free_pages * page_bytes,
                external_bytes=external_bytes,
                stranded_private_pages=policy.stranded_private_pages,
                cross_codec_reuse_events=policy.cross_codec_reuse_events,
            )
        )

    return HeteroKVSimulationResult(
        policy_name=policy.name,
        page_bytes=page_bytes,
        total_pages=policy.total_pages,
        total_arrivals=total_arrivals,
        admitted_requests=admitted_requests,
        rejected_requests=rejected_requests,
        cross_codec_reuse_events=policy.cross_codec_reuse_events,
        samples=tuple(samples),
    )
