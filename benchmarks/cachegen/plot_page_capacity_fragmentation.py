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


def plot_metric(
    df: pd.DataFrame,
    *,
    metric: str,
    ylabel: str,
    title: str,
    path: Path,
    multiplier: float = 1.0,
) -> None:
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    grouped = (
        df.groupby(
            [
                "design",
                "physical_page_bytes",
                "requested_sequences",
            ],
            as_index=False,
        )[metric]
        .mean()
    )
    for (design, page_bytes), series in grouped.groupby(
        ["design", "physical_page_bytes"]
    ):
        series = series.sort_values("requested_sequences")
        ax.plot(
            series["requested_sequences"],
            multiplier * series[metric],
            marker="o",
            label=(
                f"{design}, "
                f"{int(page_bytes) // 1024} KiB"
            ),
        )
    ax.set_xlabel("Requested concurrent sequences")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    save(fig, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    out = Path(args.output_dir)

    for checksum_mode, subset in df.groupby("checksum_mode"):
        plot_metric(
            subset,
            metric="token_admission_rate",
            ylabel="Admitted token fraction",
            title=(
                "Token capacity under heterogeneous quantizer demand "
                f"(checksum={checksum_mode})"
            ),
            path=out / f"token_capacity_{checksum_mode}",
            multiplier=100,
        )

        plot_metric(
            subset,
            metric="external_fragmentation_ratio",
            ylabel="External fragmentation (% of total pool)",
            title=(
                "External fragmentation from quantizer-specific pools "
                f"(checksum={checksum_mode})"
            ),
            path=out / f"external_fragmentation_{checksum_mode}",
            multiplier=100,
        )

        plot_metric(
            subset,
            metric="tail_internal_ratio",
            ylabel="Tail/internal fragmentation (% of allocated bytes)",
            title=(
                "Tail fragmentation from finite page granularity "
                f"(checksum={checksum_mode})"
            ),
            path=out / f"tail_fragmentation_{checksum_mode}",
            multiplier=100,
        )

        plot_metric(
            subset,
            metric="globally_satisfiable_rejections",
            ylabel="Rejected requests despite sufficient total free bytes",
            title=(
                "Stranded capacity events "
                f"(checksum={checksum_mode})"
            ),
            path=out / f"stranded_capacity_events_{checksum_mode}",
        )

    summary = (
        df.groupby(
            [
                "design",
                "checksum_mode",
                "physical_page_bytes",
                "requested_sequences",
            ],
            as_index=False,
        )
        .mean(numeric_only=True)
    )
    summary.to_csv(out / "summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
