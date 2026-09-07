#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Summarize shared-versus-private heterogeneous KV-page simulations."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def first_row(rows: list[dict[str, str]]) -> dict[str, str]:
    if not rows:
        raise ValueError("CSV has no data rows")
    return rows[0]


def as_int(row: dict[str, str], field: str) -> int:
    return int(row[field])


def as_float(row: dict[str, str], field: str) -> float:
    return float(row[field])


def summarize_run(run_dir: Path) -> dict[str, object]:
    private_rows = read_rows(run_dir / "private.csv")
    shared_rows = read_rows(run_dir / "shared.csv")

    private = first_row(private_rows)
    shared = first_row(shared_rows)

    if private["page_bytes"] != shared["page_bytes"]:
        raise ValueError(f"{run_dir}: private/shared page sizes differ")
    if private["total_pages"] != shared["total_pages"]:
        raise ValueError(f"{run_dir}: private/shared page counts differ")
    if private["total_arrivals"] != shared["total_arrivals"]:
        raise ValueError(f"{run_dir}: private/shared trace sizes differ")

    private_admitted = as_int(private, "admitted_requests")
    shared_admitted = as_int(shared, "admitted_requests")
    private_rate = as_float(private, "admission_rate")
    shared_rate = as_float(shared, "admission_rate")

    private_live_pages = [as_int(row, "live_pages") for row in private_rows]
    shared_live_pages = [as_int(row, "live_pages") for row in shared_rows]
    stranded_pages = [
        as_int(row, "stranded_private_pages") for row in private_rows
    ]

    return {
        "run": run_dir.name,
        "page_bytes": as_int(private, "page_bytes"),
        "total_pages": as_int(private, "total_pages"),
        "total_arrivals": as_int(private, "total_arrivals"),
        "trace_cachegen_requests": as_int(
            private,
            "trace_cachegen_requests",
        ),
        "trace_kivi_requests": as_int(private, "trace_kivi_requests"),
        "trace_turboquant_requests": as_int(
            private,
            "trace_turboquant_requests",
        ),
        "private_admitted": private_admitted,
        "shared_admitted": shared_admitted,
        "admitted_delta": shared_admitted - private_admitted,
        "private_admission_rate": private_rate,
        "shared_admission_rate": shared_rate,
        "admission_rate_delta": shared_rate - private_rate,
        "relative_admission_improvement": (
            (shared_admitted - private_admitted) / private_admitted
            if private_admitted > 0
            else 0.0
        ),
        "private_peak_live_pages": max(private_live_pages),
        "shared_peak_live_pages": max(shared_live_pages),
        "private_mean_live_pages": (
            sum(private_live_pages) / len(private_live_pages)
        ),
        "shared_mean_live_pages": (
            sum(shared_live_pages) / len(shared_live_pages)
        ),
        "private_peak_stranded_pages": max(stranded_pages),
        "private_mean_stranded_pages": (
            sum(stranded_pages) / len(stranded_pages)
        ),
        "shared_cross_codec_reuse_events": as_int(
            shared,
            "cross_codec_reuse_events",
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "run_dirs",
        nargs="+",
        type=Path,
        help="Directories that each contain private.csv and shared.csv",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    summaries = [summarize_run(run_dir) for run_dir in args.run_dirs]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    for summary in summaries:
        print(
            f"{summary['run']}: "
            f"private={summary['private_admitted']}/"
            f"{summary['total_arrivals']} "
            f"shared={summary['shared_admitted']}/"
            f"{summary['total_arrivals']} "
            f"delta={summary['admitted_delta']} "
            f"relative_gain="
            f"{100 * summary['relative_admission_improvement']:.2f}% "
            f"peak_stranded_pages="
            f"{summary['private_peak_stranded_pages']} "
            f"cross_codec_reuse="
            f"{summary['shared_cross_codec_reuse_events']}"
        )


if __name__ == "__main__":
    main()
