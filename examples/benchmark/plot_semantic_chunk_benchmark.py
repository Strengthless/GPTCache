import argparse
import csv
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional

try:
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "Missing dependency: install matplotlib to plot benchmark graphs."
    ) from exc


VARIANT_RE = re.compile(
    r"target_rate=([0-9.]+),\s*order=([a-z_]+),\s*prefix_context=(True|False),\s*mixed_per_pair=([0-9]+)"
)


def _parse_report(report_path: str) -> List[dict]:
    records: List[dict] = []
    current: Optional[dict] = None

    with open(report_path, "r", encoding="utf-8") as handler:
        for raw_line in handler:
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith("Dataset:"):
                if current and "hit_no_chunk" in current and "hit_chunk" in current:
                    records.append(current)
                current = {"dataset": line.split(":", 1)[1].strip()}
                continue

            if current is None:
                continue

            if line.startswith("Sessions:"):
                current["sessions"] = int(line.split(":", 1)[1].strip())
                continue

            if line.startswith("Variant:"):
                payload = line.split(":", 1)[1].strip()
                match = VARIANT_RE.search(payload)
                if match:
                    current["target_rate"] = float(match.group(1))
                    current["order"] = match.group(2)
                    current["prefix_context"] = match.group(3) == "True"
                    current["mixed_per_pair"] = int(match.group(4))
                continue

            if line.startswith("No chunking total hit rate:"):
                current["hit_no_chunk"] = float(line.split(":", 1)[1].split()[0])
                continue

            if line.startswith("Chunking total hit rate:"):
                current["hit_chunk"] = float(line.split(":", 1)[1].split()[0])
                continue

            if line.startswith("Max hit rate (primed upper bound):"):
                current["upper_bound"] = float(line.split(":", 1)[1].strip())
                continue

            if line.startswith("Expected max hit rate (shuffle):"):
                current["upper_bound"] = float(line.split(":", 1)[1].strip())
                continue

            if line.startswith("no_chunk precision/recall:"):
                numbers = line.split(":", 1)[1].strip().split("/")
                current["precision_no_chunk"] = float(numbers[0])
                current["recall_no_chunk"] = float(numbers[1])
                continue

            if line.startswith("chunk precision/recall:"):
                numbers = line.split(":", 1)[1].strip().split("/")
                current["precision_chunk"] = float(numbers[0])
                current["recall_chunk"] = float(numbers[1])
                continue

    if current and "hit_no_chunk" in current and "hit_chunk" in current:
        records.append(current)

    return records


def _read_speed_csv(path: str) -> List[dict]:
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as handler:
        reader = csv.DictReader(handler)
        for row in reader:
            rows.append(
                {
                    "turns": int(float(row["turns"])),
                    "nodes": int(float(row["nodes"])),
                    "chunk_ms": float(row["chunk_ms"]),
                    "embed_ms": float(row["embed_ms"]),
                    "search_ms": float(row["search_ms"]),
                    "save_ms": float(row["save_ms"]),
                    "total_ms": float(row["total_ms"]),
                }
            )
    rows.sort(key=lambda item: item["turns"])
    return rows


def _variant_label(record: dict) -> str:
    key = (
        round(float(record.get("target_rate", 0.0)), 3),
        record.get("order"),
        bool(record.get("prefix_context")),
        int(record.get("mixed_per_pair", 0)),
    )
    pretty_names = {
        (0.5, "primed", False, 0): "Baseline",
        (0.5, "primed", False, 1): "+Mixed",
        (0.5, "primed", True, 0): "+Prefix",
        (0.5, "primed", True, 1): "+Mixed +Prefix",
        (0.5, "shuffle", False, 1): "Shuffle order",
        (0.3, "primed", False, 1): "Sparse similarity",
        (0.7, "primed", False, 1): "Dense similarity",
    }
    if key in pretty_names:
        return pretty_names[key]
    order = "P" if record.get("order") == "primed" else "S"
    prefix = 1 if record.get("prefix_context") else 0
    return (
        f"r={record.get('target_rate', 0.0):.1f}|"
        f"o={order}|px={prefix}|m={record.get('mixed_per_pair', 0)}"
    )


def _plot_quality(records: List[dict], out_dir: str, title_prefix: str, dpi: int):
    grouped = defaultdict(list)
    for record in records:
        grouped[record["dataset"]].append(record)

    for dataset, dataset_records in grouped.items():
        _VARIANT_ORDER = {
            (0.5, "primed", False, 0): 0,  # Baseline
            (0.5, "primed", False, 1): 1,  # +Mixed
            (0.5, "primed", True, 0): 2,   # +Prefix
            (0.5, "primed", True, 1): 3,   # +Mixed +Prefix
            (0.5, "shuffle", False, 1): 4,  # Shuffle order
            (0.3, "primed", False, 1): 5,  # Sparse similarity
            (0.7, "primed", False, 1): 6,  # Dense similarity
        }
        dataset_records.sort(
            key=lambda r: _VARIANT_ORDER.get(
                (
                    round(float(r.get("target_rate", 0.0)), 3),
                    r.get("order"),
                    bool(r.get("prefix_context")),
                    int(r.get("mixed_per_pair", 0)),
                ),
                99,
            )
        )

        labels = [_variant_label(r) for r in dataset_records]
        hit_no_chunk = [r["hit_no_chunk"] for r in dataset_records]
        hit_chunk = [r["hit_chunk"] for r in dataset_records]
        upper_bound = [r.get("upper_bound") for r in dataset_records]

        x = list(range(len(dataset_records)))
        width = 0.38

        fig, ax = plt.subplots(figsize=(14, 5.5))
        ax.bar([i - width / 2 for i in x], hit_no_chunk, width=width, label="No chunking")
        ax.bar([i + width / 2 for i in x], hit_chunk, width=width, label="Chunking")
        if any(value is not None for value in upper_bound):
            y = [value if value is not None else 0.0 for value in upper_bound]
            ax.plot(x, y, color="black", linestyle="--", marker="o", label="Upper bound")

        max_val = max(hit_no_chunk + hit_chunk + [value for value in upper_bound if value is not None] + [0.1])
        ax.set_ylim(0.0, min(1.0, max_val * 1.25))
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_ylabel("Hit rate")
        ax.set_title(f"{title_prefix} - Quality Hit Rate ({dataset})")
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
        fig.tight_layout()

        output_path = os.path.join(out_dir, f"quality_hit_rate_{dataset}.png")
        fig.savefig(output_path, dpi=dpi)
        plt.close(fig)

        primed = [r for r in dataset_records if "precision_chunk" in r and "precision_no_chunk" in r]
        if primed:
            fig, ax = plt.subplots(figsize=(7.5, 6.5))
            for record in primed:
                label = _variant_label(record)
                ax.scatter(
                    record["precision_no_chunk"],
                    record["recall_no_chunk"],
                    color="tab:blue",
                    marker="o",
                )
                ax.scatter(
                    record["precision_chunk"],
                    record["recall_chunk"],
                    color="tab:orange",
                    marker="^",
                )
                ax.annotate(
                    label,
                    (record["precision_chunk"], record["recall_chunk"]),
                    fontsize=8,
                    xytext=(4, 4),
                    textcoords="offset points",
                )

            ax.scatter([], [], color="tab:blue", marker="o", label="No chunking")
            ax.scatter([], [], color="tab:orange", marker="^", label="Chunking")
            ax.set_xlim(0.0, 1.02)
            ax.set_ylim(0.0, 1.02)
            ax.set_xlabel("Precision")
            ax.set_ylabel("Recall")
            ax.set_title(f"{title_prefix} - Precision/Recall ({dataset}, primed)")
            ax.grid(alpha=0.25)
            ax.legend(loc="lower left")
            fig.tight_layout()

            output_path = os.path.join(out_dir, f"quality_precision_recall_{dataset}.png")
            fig.savefig(output_path, dpi=dpi)
            plt.close(fig)


def _plot_latency(
    chunk_rows: List[dict],
    no_chunk_rows: List[dict],
    out_dir: str,
    title_prefix: str,
    dpi: int,
):
    x_chunk = [row["turns"] for row in chunk_rows]
    x_no_chunk = [row["turns"] for row in no_chunk_rows]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=False)
    metrics = [
        ("total_ms", "Total latency (ms)"),
        ("search_ms", "Search latency (ms)"),
        ("save_ms", "Save latency (ms)"),
        ("nodes", "Cache nodes"),
    ]

    for ax, (metric, label) in zip(axes.flatten(), metrics):
        ax.plot(
            x_no_chunk,
            [row[metric] for row in no_chunk_rows],
            marker="o",
            linewidth=1.6,
            label="No chunking",
        )
        ax.plot(
            x_chunk,
            [row[metric] for row in chunk_rows],
            marker="o",
            linewidth=1.6,
            label="Chunking",
        )
        ax.set_xlabel("Turns processed")
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)

    axes[0][0].legend()
    fig.suptitle(f"{title_prefix} - Latency Growth")
    fig.tight_layout()
    fig.subplots_adjust(top=0.92)
    fig.savefig(os.path.join(out_dir, "latency_growth_vs_turns.png"), dpi=dpi)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(
        [row["nodes"] for row in no_chunk_rows],
        [row["total_ms"] for row in no_chunk_rows],
        marker="o",
        linewidth=1.8,
        label="No chunking",
    )
    ax.plot(
        [row["nodes"] for row in chunk_rows],
        [row["total_ms"] for row in chunk_rows],
        marker="o",
        linewidth=1.8,
        label="Chunking",
    )
    ax.set_xlabel("Cache nodes")
    ax.set_ylabel("Total latency (ms)")
    ax.set_title(f"{title_prefix} - Total Latency vs Cache Size")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "latency_total_vs_nodes.png"), dpi=dpi)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot semantic chunk benchmark quality and latency graphs.")
    parser.add_argument("--report-path", required=True, help="Path to semantic chunk benchmark report.txt")
    parser.add_argument("--speed-chunk-csv", required=True, help="CSV from speed benchmark with chunking enabled")
    parser.add_argument("--speed-nochunk-csv", required=True, help="CSV from speed benchmark with chunking disabled")
    parser.add_argument("--out-dir", required=True, help="Directory to write output PNG graphs")
    parser.add_argument("--title-prefix", default="Semantic Chunking Benchmark")
    parser.add_argument("--dpi", type=int, default=140)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    quality_records = _parse_report(args.report_path)
    if not quality_records:
        raise RuntimeError(f"No benchmark records parsed from report: {args.report_path}")
    _plot_quality(quality_records, args.out_dir, args.title_prefix, args.dpi)

    speed_chunk = _read_speed_csv(args.speed_chunk_csv)
    speed_nochunk = _read_speed_csv(args.speed_nochunk_csv)
    if not speed_chunk or not speed_nochunk:
        raise RuntimeError("Speed CSV inputs are empty; cannot plot latency.")
    _plot_latency(speed_chunk, speed_nochunk, args.out_dir, args.title_prefix, args.dpi)

    print(f"Wrote plots to {args.out_dir}")


if __name__ == "__main__":
    main()
