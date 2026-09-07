from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from math import ceil
from pathlib import Path


QUANTIZERS = ("bf16", "fp8", "int8", "int4")
BITS = {
    "bf16": 16,
    "fp8": 8,
    "int8": 8,
    "int4": 4,
}
FORMAT_METADATA_BYTES = {
    "bf16": 0,
    "fp8": 32,
    "int8": 32,
    "int4": 48,
}


def parse_ints(text: str) -> list[int]:
    return [int(value) for value in text.split(",") if value.strip()]


def parse_floats(text: str) -> list[float]:
    values = [float(value) for value in text.split(",") if value.strip()]
    if len(values) != len(QUANTIZERS):
        raise ValueError("mix needs bf16,fp8,int8,int4 weights")
    if sum(values) <= 0:
        raise ValueError("mix must have positive total weight")
    return values


@dataclass(frozen=True)
class Layout:
    quantizer: str
    physical_page_bytes: int
    token_page_size: int
    bytes_per_token: int
    header_bytes: int
    format_metadata_bytes: int
    checksum_bytes: int
    tokens_per_page: int
    used_payload_bytes: int
    layout_slack_bytes: int

    @property
    def per_page_metadata_bytes(self) -> int:
        return (
            self.header_bytes
            + self.format_metadata_bytes
            + self.checksum_bytes
        )


def bytes_per_token(
    quantizer: str,
    num_kv_heads: int,
    head_size: int,
) -> int:
    return (
        2 * num_kv_heads * head_size * BITS[quantizer]
    ) // 8


def fixed_byte_layout(
    *,
    quantizer: str,
    physical_page_bytes: int,
    num_kv_heads: int,
    head_size: int,
    header_bytes: int,
    checksum_bytes: int,
) -> Layout:
    bpt = bytes_per_token(quantizer, num_kv_heads, head_size)
    overhead = (
        header_bytes
        + FORMAT_METADATA_BYTES[quantizer]
        + checksum_bytes
    )
    token_capacity = (physical_page_bytes - overhead) // bpt
    if token_capacity <= 0:
        raise ValueError(
            f"{physical_page_bytes} bytes cannot fit {quantizer}"
        )
    used_payload = token_capacity * bpt
    layout_slack = (
        physical_page_bytes - overhead - used_payload
    )
    return Layout(
        quantizer=quantizer,
        physical_page_bytes=physical_page_bytes,
        token_page_size=token_capacity,
        bytes_per_token=bpt,
        header_bytes=header_bytes,
        format_metadata_bytes=FORMAT_METADATA_BYTES[quantizer],
        checksum_bytes=checksum_bytes,
        tokens_per_page=token_capacity,
        used_payload_bytes=used_payload,
        layout_slack_bytes=layout_slack,
    )


def fixed_token_layout(
    *,
    quantizer: str,
    token_page_size: int,
    num_kv_heads: int,
    head_size: int,
    header_bytes: int,
    checksum_bytes: int,
) -> Layout:
    bpt = bytes_per_token(quantizer, num_kv_heads, head_size)
    page_bytes = (
        header_bytes
        + FORMAT_METADATA_BYTES[quantizer]
        + checksum_bytes
        + token_page_size * bpt
    )
    return Layout(
        quantizer=quantizer,
        physical_page_bytes=page_bytes,
        token_page_size=token_page_size,
        bytes_per_token=bpt,
        header_bytes=header_bytes,
        format_metadata_bytes=FORMAT_METADATA_BYTES[quantizer],
        checksum_bytes=checksum_bytes,
        tokens_per_page=token_page_size,
        used_payload_bytes=token_page_size * bpt,
        layout_slack_bytes=0,
    )


@dataclass
class Allocation:
    quantizer: str
    tokens: int
    pages: int
    layout: Layout

    @property
    def allocated_bytes(self) -> int:
        return self.pages * self.layout.physical_page_bytes

    @property
    def requested_payload_bytes(self) -> int:
        return self.tokens * self.layout.bytes_per_token

    @property
    def tail_internal_bytes(self) -> int:
        return (
            self.pages * self.layout.used_payload_bytes
            - self.requested_payload_bytes
        )

    @property
    def layout_slack_bytes(self) -> int:
        return self.pages * self.layout.layout_slack_bytes

    @property
    def metadata_bytes(self) -> int:
        return self.pages * self.layout.per_page_metadata_bytes


class SharedFixedByteAllocator:
    def __init__(self, page_bytes: int, total_budget_bytes: int) -> None:
        self.page_bytes = page_bytes
        self.total_pages = total_budget_bytes // page_bytes
        self.free_pages = self.total_pages
        self.allocations: list[Allocation] = []
        self.rejections = 0
        self.external_fragmentation_bytes = 0
        self.globally_satisfiable_rejections = 0

    def allocate(self, quantizer: str, tokens: int, layout: Layout) -> bool:
        pages = ceil(tokens / layout.tokens_per_page)
        if self.free_pages < pages:
            self.rejections += 1
            return False
        self.free_pages -= pages
        self.allocations.append(
            Allocation(quantizer, tokens, pages, layout)
        )
        return True


class PartitionedAllocator:
    def __init__(
        self,
        capacities: dict[str, int],
        layouts: dict[str, Layout],
    ) -> None:
        self.capacities = capacities
        self.layouts = layouts
        self.free_pages = dict(capacities)
        self.allocations: list[Allocation] = []
        self.rejections = 0
        self.external_fragmentation_bytes = 0
        self.globally_satisfiable_rejections = 0

    def allocate(self, quantizer: str, tokens: int, layout: Layout) -> bool:
        pages = ceil(tokens / layout.tokens_per_page)
        if self.free_pages[quantizer] >= pages:
            self.free_pages[quantizer] -= pages
            self.allocations.append(
                Allocation(quantizer, tokens, pages, layout)
            )
            return True

        self.rejections += 1
        free_bytes_elsewhere = sum(
            self.free_pages[q] * self.layouts[q].physical_page_bytes
            for q in QUANTIZERS
            if q != quantizer
        )
        self.external_fragmentation_bytes += free_bytes_elsewhere

        total_free_bytes = sum(
            self.free_pages[q] * self.layouts[q].physical_page_bytes
            for q in QUANTIZERS
        )
        request_bytes = pages * layout.physical_page_bytes
        if total_free_bytes >= request_bytes:
            self.globally_satisfiable_rejections += 1
        return False


def summarize(
    *,
    design: str,
    allocator: SharedFixedByteAllocator | PartitionedAllocator,
    total_budget_bytes: int,
    requested_sequences: int,
    requested_tokens: int,
    checksum_mode: str,
    physical_page_bytes: int,
    fixed_token_page_size: int,
) -> dict[str, object]:
    allocations = allocator.allocations
    admitted_tokens = sum(a.tokens for a in allocations)
    allocated_bytes = sum(a.allocated_bytes for a in allocations)
    requested_payload_bytes = sum(
        a.requested_payload_bytes for a in allocations
    )
    tail_internal_bytes = sum(a.tail_internal_bytes for a in allocations)
    layout_slack_bytes = sum(a.layout_slack_bytes for a in allocations)
    metadata_bytes = sum(a.metadata_bytes for a in allocations)
    checksum_bytes = sum(
        a.pages * a.layout.checksum_bytes for a in allocations
    )

    if isinstance(allocator, SharedFixedByteAllocator):
        free_bytes = allocator.free_pages * physical_page_bytes
    else:
        free_bytes = sum(
            allocator.free_pages[q] * allocator.layouts[q].physical_page_bytes
            for q in QUANTIZERS
        )

    return {
        "design": design,
        "checksum_mode": checksum_mode,
        "physical_page_bytes": physical_page_bytes,
        "fixed_token_page_size": fixed_token_page_size,
        "total_budget_bytes": total_budget_bytes,
        "requested_sequences": requested_sequences,
        "admitted_sequences": len(allocations),
        "rejected_sequences": allocator.rejections,
        "admission_rate": len(allocations) / requested_sequences,
        "requested_tokens": requested_tokens,
        "admitted_tokens": admitted_tokens,
        "token_admission_rate": admitted_tokens / requested_tokens,
        "allocated_page_bytes": allocated_bytes,
        "requested_payload_bytes": requested_payload_bytes,
        "tail_internal_bytes": tail_internal_bytes,
        "tail_internal_ratio": (
            tail_internal_bytes / allocated_bytes
            if allocated_bytes else 0.0
        ),
        "layout_slack_bytes": layout_slack_bytes,
        "layout_slack_ratio": (
            layout_slack_bytes / allocated_bytes
            if allocated_bytes else 0.0
        ),
        "metadata_bytes": metadata_bytes,
        "metadata_ratio": (
            metadata_bytes / allocated_bytes
            if allocated_bytes else 0.0
        ),
        "checksum_bytes": checksum_bytes,
        "checksum_ratio": (
            checksum_bytes / allocated_bytes
            if allocated_bytes else 0.0
        ),
        "external_fragmentation_bytes": (
            allocator.external_fragmentation_bytes
        ),
        "external_fragmentation_ratio": (
            allocator.external_fragmentation_bytes
            / total_budget_bytes
        ),
        "globally_satisfiable_rejections": (
            allocator.globally_satisfiable_rejections
        ),
        "free_bytes": free_bytes,
    }


def run_trial(
    *,
    seed: int,
    total_budget_bytes: int,
    physical_page_bytes: int,
    fixed_token_page_size: int,
    sequence_count: int,
    min_tokens: int,
    max_tokens: int,
    mix: list[float],
    num_kv_heads: int,
    head_size: int,
    checksum_mode: str,
) -> list[dict[str, object]]:
    rng = random.Random(seed)
    checksum_bytes = 4 if checksum_mode == "crc32" else 0
    header_bytes = 32

    shared_layouts = {
        q: fixed_byte_layout(
            quantizer=q,
            physical_page_bytes=physical_page_bytes,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            header_bytes=header_bytes,
            checksum_bytes=checksum_bytes,
        )
        for q in QUANTIZERS
    }
    token_layouts = {
        q: fixed_token_layout(
            quantizer=q,
            token_page_size=fixed_token_page_size,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            header_bytes=header_bytes,
            checksum_bytes=checksum_bytes,
        )
        for q in QUANTIZERS
    }

    shared = SharedFixedByteAllocator(
        page_bytes=physical_page_bytes,
        total_budget_bytes=total_budget_bytes,
    )

    shared_partition_capacity = {
        q: max(
            1,
            int(
                (total_budget_bytes / len(QUANTIZERS))
                // physical_page_bytes
            ),
        )
        for q in QUANTIZERS
    }
    fixed_byte_partitioned = PartitionedAllocator(
        shared_partition_capacity,
        shared_layouts,
    )

    token_partition_capacity = {
        q: max(
            1,
            int(
                (total_budget_bytes / len(QUANTIZERS))
                // token_layouts[q].physical_page_bytes
            ),
        )
        for q in QUANTIZERS
    }
    fixed_token_partitioned = PartitionedAllocator(
        token_partition_capacity,
        token_layouts,
    )

    requests = [
        (
            rng.choices(QUANTIZERS, weights=mix, k=1)[0],
            rng.randint(min_tokens, max_tokens),
        )
        for _ in range(sequence_count)
    ]

    for quantizer, tokens in requests:
        shared.allocate(quantizer, tokens, shared_layouts[quantizer])
        fixed_byte_partitioned.allocate(
            quantizer,
            tokens,
            shared_layouts[quantizer],
        )
        fixed_token_partitioned.allocate(
            quantizer,
            tokens,
            token_layouts[quantizer],
        )

    requested_tokens = sum(tokens for _, tokens in requests)
    common = dict(
        total_budget_bytes=total_budget_bytes,
        requested_sequences=sequence_count,
        requested_tokens=requested_tokens,
        checksum_mode=checksum_mode,
        physical_page_bytes=physical_page_bytes,
        fixed_token_page_size=fixed_token_page_size,
    )
    return [
        summarize(
            design="shared_fixed_byte",
            allocator=shared,
            **common,
        ),
        summarize(
            design="partitioned_fixed_byte",
            allocator=fixed_byte_partitioned,
            **common,
        ),
        summarize(
            design="partitioned_fixed_token",
            allocator=fixed_token_partitioned,
            **common,
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--page-bytes", default="16384,32768,65536,131072")
    parser.add_argument("--token-page-size", type=int, default=16)
    parser.add_argument("--sequence-counts", default="64,128,256,512,1024,2048")
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--budget-gib", type=float, default=1.0)
    parser.add_argument("--min-tokens", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--mix", default="1,1,1,1")
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--head-size", type=int, default=128)
    args = parser.parse_args()

    rows = []
    for page_bytes in parse_ints(args.page_bytes):
        for sequence_count in parse_ints(args.sequence_counts):
            for seed in range(args.trials):
                for checksum_mode in ("off", "crc32"):
                    rows.extend(
                        run_trial(
                            seed=seed,
                            total_budget_bytes=int(args.budget_gib * 1024**3),
                            physical_page_bytes=page_bytes,
                            fixed_token_page_size=args.token_page_size,
                            sequence_count=sequence_count,
                            min_tokens=args.min_tokens,
                            max_tokens=args.max_tokens,
                            mix=parse_floats(args.mix),
                            num_kv_heads=args.num_kv_heads,
                            head_size=args.head_size,
                            checksum_mode=checksum_mode,
                        )
                    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0]),
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
