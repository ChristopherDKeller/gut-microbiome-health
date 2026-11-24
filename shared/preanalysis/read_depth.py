"""Histogram of reads per sample after bloom removal.

One bar per log-spaced bin; the dashed line marks the depth filter used in process.py.
Uses the chosen prep per sample, before the depth filter, so the samples
that get dropped are visible left of the line.

Input:  data/features/samples.tsv   (shared/features/process.py)
Output: results/shared/read_depth/read_depth.png
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from shared.features.process import MIN_READS, OUT_SAMPLES
from shared.paths import COHORT_DIR, SHARED_RESULTS

OUT_DIR = SHARED_RESULTS / "read_depth"
THRESHOLDS = sorted({MIN_READS, 1000, 2000, 3000, 4000, 5000})

if __name__ == "__main__":
    reads = pd.read_csv(OUT_SAMPLES, sep="\t")["reads"].clip(lower=1)
    dropped = (reads < MIN_READS).sum()

    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    bins = np.logspace(0, np.log10(reads.max()), 60)
    ax.hist(reads, bins=bins, color="#2a78d6")
    ax.axvline(MIN_READS, color="#52514e", linestyle="--", linewidth=1)
    ax.text(MIN_READS * 0.9, ax.get_ylim()[1] * 0.95,
            f"{MIN_READS:,} reads\n{dropped:,} samples below",
            va="top", ha="right", fontsize=8, color="#52514e")

    ax.set_xscale("log")
    ax.set_xlabel("Reads per sample after bloom removal (log scale)")
    ax.set_ylabel("Samples")
    ax.set_title(f"Read depth, {len(reads):,} samples (median {reads.median():,.0f})", loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "read_depth.png")
    print(reads.describe().round(0).to_string())

    # Samples left per cohort group for different depth thresholds.
    # Samples without 16S data in the context count as missing ("in_table").
    depth = pd.read_csv(OUT_SAMPLES, sep="\t").set_index("sample_name")["reads"]
    rows = []
    for path in sorted(COHORT_DIR.glob("*/*.tsv")):
        cohort = pd.read_csv(path, sep="\t", dtype=str)
        cohort["reads"] = cohort["sample_name"].map(depth)
        for group, g in cohort.groupby("group"):
            row = {"cohort": f"{path.parent.name}/{path.stem}", "group": group,
                   "n": len(g), "in_table": int(g["reads"].notna().sum())}
            row.update({f">={t}": int((g["reads"] >= t).sum()) for t in THRESHOLDS})
            rows.append(row)

    per_cohort = pd.DataFrame(rows)
    per_cohort.to_csv(OUT_DIR / "samples_per_threshold.tsv", sep="\t", index=False)
    print("\nSamples left per cohort and read threshold:")
    print(per_cohort.to_string(index=False))
