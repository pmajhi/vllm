from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="artifacts/fixed_byte")
    parser.add_argument("--output-dir", default="artifacts/fixed_byte/plots")
    args = parser.parse_args()

    files = sorted(Path(args.input_dir).glob("*.csv"))
    if not files:
        raise SystemExit("No CSV files found")

    frames = []
    for file in files:
        frame = pd.read_csv(file)
        frame["workload"] = file.stem
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)

    numeric = [
        "admission_rate",
        "allocated_page_bytes",
        "requested_payload_bytes",
        "internal_fragmentation_bytes",
        "tail_slack_bytes",
        "checksum_bytes_reserved",
        "allocation_failures",
    ]
    summary = (
        df.groupby(
            [
                "workload",
                "design",
                "integrity_mode",
                "physical_page_bytes",
                "requested_sequences",
            ],
            as_index=False,
        )[numeric]
        .mean()
    )
    summary["internal_fragmentation_ratio"] = (
        summary["internal_fragmentation_bytes"]
        / summary["allocated_page_bytes"].clip(lower=1)
    )

    out = Path(args.output_dir)

    for workload, workload_df in summary.groupby("workload"):
        for integrity, data in workload_df.groupby("integrity_mode"):
            fig, ax = plt.subplots(figsize=(7.4, 4.6))
            for (design, page_bytes), series in data.groupby(
                ["design", "physical_page_bytes"]
            ):
                series = series.sort_values("requested_sequences")
                ax.plot(
                    series["requested_sequences"],
                    series["admission_rate"],
                    marker="o",
                    label=(
                        f"{design}, "
                        f"{int(page_bytes) // 1024} KiB"
                    ),
                )
            ax.set_xlabel("Requested concurrent sequences")
            ax.set_ylabel("Admission rate")
            ax.set_ylim(-0.03, 1.03)
            ax.set_title(
                f"Mixed-quantizer capacity: {workload}, integrity={integrity}"
            )
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8, ncol=2)
            save(
                fig,
                out / f"admission_{workload}_{integrity}",
            )

            fig, ax = plt.subplots(figsize=(7.4, 4.6))
            for (design, page_bytes), series in data.groupby(
                ["design", "physical_page_bytes"]
            ):
                series = series.sort_values("requested_sequences")
                ax.plot(
                    series["requested_sequences"],
                    100 * series["internal_fragmentation_ratio"],
                    marker="o",
                    label=(
                        f"{design}, "
                        f"{int(page_bytes) // 1024} KiB"
                    ),
                )
            ax.set_xlabel("Requested concurrent sequences")
            ax.set_ylabel("Internal fragmentation (%)")
            ax.set_title(
                f"Internal fragmentation: {workload}, integrity={integrity}"
            )
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8, ncol=2)
            save(
                fig,
                out / f"fragmentation_{workload}_{integrity}",
            )

    checksum = (
        summary[summary["integrity_mode"] == "crc32"]
        .groupby(
            ["workload", "design", "physical_page_bytes"],
            as_index=False,
        )["checksum_bytes_reserved"]
        .mean()
    )
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    for design, series in checksum.groupby("design"):
        grouped = series.groupby(
            "physical_page_bytes",
            as_index=False,
        )["checksum_bytes_reserved"].mean()
        ax.plot(
            grouped["physical_page_bytes"] / 1024,
            grouped["checksum_bytes_reserved"] / 1024,
            marker="o",
            label=design,
        )
    ax.set_xlabel("Physical page size (KiB)")
    ax.set_ylabel("Reserved CRC metadata (KiB)")
    ax.set_title("Checksum metadata cost")
    ax.grid(alpha=0.25)
    ax.legend()
    save(fig, out / "checksum_overhead")

    summary.to_csv(out / "summary.csv", index=False)
    print(f"wrote plots and summary to {out}")


if __name__ == "__main__":
    main()
