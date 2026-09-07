from __future__ import annotations

import argparse
import csv
import random
from dataclasses import asdict
from pathlib import Path

from vllm.v1.worker.experimental.cachegen_fixed_byte_pages import (
    CacheGenFixedBytePagePool,
    CacheGenIntegrityMode,
    CacheGenQuantizer,
)


QUANTIZERS = list(CacheGenQuantizer)


def parse_csv_ints(value: str) -> list[int]:
    return [int(x) for x in value.split(",") if x.strip()]


def parse_csv_floats(value: str) -> list[float]:
    values = [float(x) for x in value.split(",") if x.strip()]
    if len(values) != len(QUANTIZERS):
        raise ValueError(
            "mix needs 4 entries: bf16,fp8_e4m3,int8,int4"
        )
    return values


def make_pool(
    *,
    physical_page_bytes: int,
    total_budget_bytes: int,
    integrity_mode: CacheGenIntegrityMode,
    num_kv_heads: int,
    head_size: int,
) -> CacheGenFixedBytePagePool:
    return CacheGenFixedBytePagePool(
        physical_page_bytes=physical_page_bytes,
        page_count=total_budget_bytes // physical_page_bytes,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        integrity_mode=integrity_mode,
    )


def fixed_token_baseline_capacity(
    *,
    total_budget_bytes: int,
    page_tokens: int,
    num_kv_heads: int,
    head_size: int,
    mix: list[float],
) -> dict[CacheGenQuantizer, int]:
    bytes_per_token = {
        CacheGenQuantizer.BF16: 2 * num_kv_heads * head_size * 2,
        CacheGenQuantizer.FP8: 2 * num_kv_heads * head_size,
        CacheGenQuantizer.INT8: 2 * num_kv_heads * head_size,
        CacheGenQuantizer.INT4: num_kv_heads * head_size,
    }
    capacities = {}
    for quantizer, weight in zip(QUANTIZERS, mix):
        quantizer_budget = total_budget_bytes * weight / sum(mix)
        page_bytes = page_tokens * bytes_per_token[quantizer] + 64
        capacities[quantizer] = max(1, int(quantizer_budget // page_bytes))
    return capacities


def run_shared_trial(
    *,
    seed: int,
    physical_page_bytes: int,
    total_budget_bytes: int,
    integrity_mode: CacheGenIntegrityMode,
    sequence_count: int,
    min_tokens: int,
    max_tokens: int,
    mix: list[float],
    num_kv_heads: int,
    head_size: int,
) -> dict[str, object]:
    rng = random.Random(seed)
    pool = make_pool(
        physical_page_bytes=physical_page_bytes,
        total_budget_bytes=total_budget_bytes,
        integrity_mode=integrity_mode,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
    )

    admitted = 0
    for sequence_id in range(sequence_count):
        quantizer = rng.choices(QUANTIZERS, weights=mix, k=1)[0]
        token_count = rng.randint(min_tokens, max_tokens)
        allocation = pool.allocate(
            sequence_id=sequence_id,
            quantizer=quantizer,
            token_count=token_count,
        )
        admitted += allocation is not None

    stats = asdict(pool.stats())
    return {
        "design": "shared_fixed_byte",
        "seed": seed,
        "integrity_mode": integrity_mode.value,
        "physical_page_bytes": physical_page_bytes,
        "requested_sequences": sequence_count,
        "admitted_sequences": admitted,
        "admission_rate": admitted / sequence_count,
        **stats,
    }


def run_partitioned_trial(
    *,
    seed: int,
    physical_page_bytes: int,
    total_budget_bytes: int,
    integrity_mode: CacheGenIntegrityMode,
    sequence_count: int,
    min_tokens: int,
    max_tokens: int,
    mix: list[float],
    num_kv_heads: int,
    head_size: int,
) -> dict[str, object]:
    rng = random.Random(seed)
    template = make_pool(
        physical_page_bytes=physical_page_bytes,
        total_budget_bytes=total_budget_bytes,
        integrity_mode=integrity_mode,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
    )

    per_quantizer_budget = {
        q: int(total_budget_bytes * weight / sum(mix))
        for q, weight in zip(QUANTIZERS, mix)
    }
    pools = {
        q: CacheGenFixedBytePagePool(
            physical_page_bytes=physical_page_bytes,
            page_count=max(
                1,
                per_quantizer_budget[q] // physical_page_bytes,
            ),
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            integrity_mode=integrity_mode,
        )
        for q in QUANTIZERS
    }

    admitted = 0
    for sequence_id in range(sequence_count):
        quantizer = rng.choices(QUANTIZERS, weights=mix, k=1)[0]
        token_count = rng.randint(min_tokens, max_tokens)
        allocation = pools[quantizer].allocate(
            sequence_id=sequence_id,
            quantizer=quantizer,
            token_count=token_count,
        )
        admitted += allocation is not None

    pool_stats = [pool.stats() for pool in pools.values()]
    return {
        "design": "partitioned_fixed_byte",
        "seed": seed,
        "integrity_mode": integrity_mode.value,
        "physical_page_bytes": physical_page_bytes,
        "requested_sequences": sequence_count,
        "admitted_sequences": admitted,
        "admission_rate": admitted / sequence_count,
        "total_pages": sum(s.total_pages for s in pool_stats),
        "free_pages": sum(s.free_pages for s in pool_stats),
        "active_pages": sum(s.active_pages for s in pool_stats),
        "active_sequences": sum(s.active_sequences for s in pool_stats),
        "allocation_failures": sum(s.allocation_failures for s in pool_stats),
        "allocated_page_bytes": sum(s.allocated_page_bytes for s in pool_stats),
        "requested_payload_bytes": sum(s.requested_payload_bytes for s in pool_stats),
        "internal_fragmentation_bytes": sum(
            s.internal_fragmentation_bytes for s in pool_stats
        ),
        "tail_slack_bytes": sum(s.tail_slack_bytes for s in pool_stats),
        "checksum_bytes_reserved": sum(
            s.checksum_bytes_reserved for s in pool_stats
        ),
    }


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--page-bytes", default="16384,32768,65536,131072")
    parser.add_argument("--sequence-counts", default="16,32,64,128,256,512")
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--budget-gib", type=float, default=8.0)
    parser.add_argument("--min-tokens", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--mix", default="1,1,1,1")
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--head-size", type=int, default=128)
    args = parser.parse_args()

    rows = []
    budget_bytes = int(args.budget_gib * 1024**3)
    mix = parse_csv_floats(args.mix)

    for page_bytes in parse_csv_ints(args.page_bytes):
        for sequence_count in parse_csv_ints(args.sequence_counts):
            for trial in range(args.trials):
                for integrity_mode in (
                    CacheGenIntegrityMode.OFF,
                    CacheGenIntegrityMode.CRC32,
                ):
                    common = dict(
                        seed=trial,
                        physical_page_bytes=page_bytes,
                        total_budget_bytes=budget_bytes,
                        integrity_mode=integrity_mode,
                        sequence_count=sequence_count,
                        min_tokens=args.min_tokens,
                        max_tokens=args.max_tokens,
                        mix=mix,
                        num_kv_heads=args.num_kv_heads,
                        head_size=args.head_size,
                    )
                    rows.append(run_shared_trial(**common))
                    rows.append(run_partitioned_trial(**common))

    write_rows(Path(args.output), rows)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
