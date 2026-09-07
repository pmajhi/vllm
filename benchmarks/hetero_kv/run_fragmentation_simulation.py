#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run deterministic heterogeneous KV-page fragmentation experiments."""

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter
from pathlib import Path

from vllm.v1.worker.experimental.hetero_kv_codec_registry import (
    HeteroKVCodecId,
)
from vllm.v1.worker.experimental.hetero_kv_fragmentation_simulator import (
    HeteroKVTraceRequest,
    PrivateHeteroKVPagePolicy,
    SharedHeteroKVPagePolicy,
    simulate_hetero_kv_trace,
)


CODEC_BY_NAME = {
    "cachegen": HeteroKVCodecId.CACHEGEN_INT8,
    "kivi": HeteroKVCodecId.KIVI_K2_V2,
    "turboquant": HeteroKVCodecId.TURBOQUANT_K3_V2,
}


def parse_codec_mix(value: str) -> dict[HeteroKVCodecId, float]:
    parts = value.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "codec mix must be cachegen,kivi,turboquant"
        )

    try:
        weights = [float(part) for part in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "codec mix values must be numeric"
        ) from exc

    if any(weight < 0 for weight in weights) or sum(weights) <= 0:
        raise argparse.ArgumentTypeError(
            "codec mix weights must be nonnegative with positive sum"
        )

    total = sum(weights)
    return {
        codec_id: weight / total
        for codec_id, weight in zip(CODEC_BY_NAME.values(), weights)
    }


def choose_codec(
    rng: random.Random,
    codec_mix: dict[HeteroKVCodecId, float],
) -> HeteroKVCodecId:
    codecs = list(codec_mix)
    weights = [codec_mix[codec_id] for codec_id in codecs]
    return rng.choices(codecs, weights=weights, k=1)[0]


def generate_trace(
    *,
    num_requests: int,
    arrival_gap: int,
    min_cached_tokens: int,
    max_cached_tokens: int,
    min_lifetime: int,
    max_lifetime: int,
    codec_mix: dict[HeteroKVCodecId, float],
    seed: int,
) -> tuple[HeteroKVTraceRequest, ...]:
    rng = random.Random(seed)
    requests = []
    for request_index in range(num_requests):
        arrival_step = request_index * arrival_gap
        cached_tokens = rng.randint(min_cached_tokens, max_cached_tokens)
        lifetime = rng.randint(min_lifetime, max_lifetime)
        codec_id = choose_codec(rng, codec_mix)
        requests.append(
            HeteroKVTraceRequest(
                request_id=f"request-{request_index}",
                arrival_step=arrival_step,
                departure_step=arrival_step + lifetime,
                cached_tokens=cached_tokens,
                codec_id=codec_id,
            )
        )
    return tuple(requests)


def equal_private_partition(
    *,
    total_pages: int,
    codec_ids: tuple[HeteroKVCodecId, ...],
) -> dict[HeteroKVCodecId, int]:
    if total_pages < len(codec_ids):
        raise ValueError(
            "total_pages must be at least the number of codec-private pools"
        )

    pages_per_codec, remainder = divmod(total_pages, len(codec_ids))
    return {
        codec_id: pages_per_codec + int(index < remainder)
        for index, codec_id in enumerate(codec_ids)
    }


def write_samples_csv(
    *,
    output_path: Path,
    result,
    request_counts: Counter[HeteroKVCodecId],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=(
                "policy",
                "page_bytes",
                "total_pages",
                "total_arrivals",
                "admitted_requests",
                "rejected_requests",
                "admission_rate",
                "cross_codec_reuse_events",
                "trace_cachegen_requests",
                "trace_kivi_requests",
                "trace_turboquant_requests",
                "step",
                "active_requests",
                "live_pages",
                "free_pages",
                "allocated_page_bytes",
                "free_page_bytes",
                "external_bytes",
                "stranded_private_pages",
                "sample_cross_codec_reuse_events",
            ),
        )
        writer.writeheader()
        for sample in result.samples:
            writer.writerow(
                {
                    "policy": result.policy_name,
                    "page_bytes": result.page_bytes,
                    "total_pages": result.total_pages,
                    "total_arrivals": result.total_arrivals,
                    "admitted_requests": result.admitted_requests,
                    "rejected_requests": result.rejected_requests,
                    "admission_rate": result.admission_rate,
                    "cross_codec_reuse_events": result.cross_codec_reuse_events,
                    "trace_cachegen_requests": request_counts[
                        HeteroKVCodecId.CACHEGEN_INT8
                    ],
                    "trace_kivi_requests": request_counts[
                        HeteroKVCodecId.KIVI_K2_V2
                    ],
                    "trace_turboquant_requests": request_counts[
                        HeteroKVCodecId.TURBOQUANT_K3_V2
                    ],
                    "step": sample.step,
                    "active_requests": sample.active_requests,
                    "live_pages": sample.live_pages,
                    "free_pages": sample.free_pages,
                    "allocated_page_bytes": sample.allocated_page_bytes,
                    "free_page_bytes": sample.free_page_bytes,
                    "external_bytes": sample.external_bytes,
                    "stranded_private_pages": sample.stranded_private_pages,
                    "sample_cross_codec_reuse_events":
                    sample.cross_codec_reuse_events,
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-requests", type=int, default=500)
    parser.add_argument("--total-pages", type=int, default=256)
    parser.add_argument("--page-bytes", type=int, default=128 * 1024)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--head-size", type=int, default=128)
    parser.add_argument("--arrival-gap", type=int, default=1)
    parser.add_argument("--min-cached-tokens", type=int, default=256)
    parser.add_argument("--max-cached-tokens", type=int, default=8192)
    parser.add_argument("--min-lifetime", type=int, default=8)
    parser.add_argument("--max-lifetime", type=int, default=64)
    parser.add_argument(
        "--codec-mix",
        type=parse_codec_mix,
        default=parse_codec_mix("1,1,1"),
    )
    args = parser.parse_args()

    if args.num_requests <= 0:
        parser.error("--num-requests must be positive")
    if args.total_pages <= 0:
        parser.error("--total-pages must be positive")
    if args.page_bytes <= 0:
        parser.error("--page-bytes must be positive")
    if args.arrival_gap <= 0:
        parser.error("--arrival-gap must be positive")
    if args.min_cached_tokens <= 0:
        parser.error("--min-cached-tokens must be positive")
    if args.max_cached_tokens < args.min_cached_tokens:
        parser.error(
            "--max-cached-tokens must be >= --min-cached-tokens"
        )
    if args.min_lifetime <= 0:
        parser.error("--min-lifetime must be positive")
    if args.max_lifetime < args.min_lifetime:
        parser.error("--max-lifetime must be >= --min-lifetime")

    trace = generate_trace(
        num_requests=args.num_requests,
        arrival_gap=args.arrival_gap,
        min_cached_tokens=args.min_cached_tokens,
        max_cached_tokens=args.max_cached_tokens,
        min_lifetime=args.min_lifetime,
        max_lifetime=args.max_lifetime,
        codec_mix=args.codec_mix,
        seed=args.seed,
    )
    codec_ids = tuple(args.codec_mix)
    request_counts = Counter(request.codec_id for request in trace)

    private_result = simulate_hetero_kv_trace(
        requests=trace,
        policy=PrivateHeteroKVPagePolicy(
            pages_by_codec=equal_private_partition(
                total_pages=args.total_pages,
                codec_ids=codec_ids,
            ),
            page_bytes=args.page_bytes,
        ),
        page_bytes=args.page_bytes,
        num_kv_heads=args.num_kv_heads,
        head_size=args.head_size,
    )
    shared_result = simulate_hetero_kv_trace(
        requests=trace,
        policy=SharedHeteroKVPagePolicy(
            total_pages=args.total_pages,
            page_bytes=args.page_bytes,
        ),
        page_bytes=args.page_bytes,
        num_kv_heads=args.num_kv_heads,
        head_size=args.head_size,
    )

    write_samples_csv(
        output_path=args.output_dir / "private.csv",
        result=private_result,
        request_counts=request_counts,
    )
    write_samples_csv(
        output_path=args.output_dir / "shared.csv",
        result=shared_result,
        request_counts=request_counts,
    )

    for result in (private_result, shared_result):
        print(
            f"{result.policy_name}: "
            f"admitted={result.admitted_requests}/{result.total_arrivals} "
            f"rejected={result.rejected_requests} "
            f"admission_rate={result.admission_rate:.4f} "
            f"peak_live_pages={result.peak_live_pages} "
            f"mean_live_pages={result.mean_live_pages:.2f} "
            f"peak_external_bytes={result.peak_external_bytes} "
            f"cross_codec_reuse={result.cross_codec_reuse_events}"
        )


if __name__ == "__main__":
    main()
