"""Read depth and sample flow for the depth filter.

Uses the chosen prep per sample (shared/features/process.py), before the
depth filter. Prints to the console only:
- reads per sample before and after bloom removal
- samples below MIN_READS before/after bloom removal
- sample flow per cohort group: in cohort -> with 16S data -> >= MIN_READS

Run:  python shared/preanalysis/read_depth.py
"""

import pandas as pd

from shared.features.process import MIN_READS, OUT_SAMPLES
from shared.paths import COHORT_DIR

if __name__ == "__main__":
    samples = pd.read_csv(OUT_SAMPLES, sep="\t")

    depth_summary = pd.DataFrame({
        "before bloom removal": samples["reads_raw"].describe(),
        "after bloom removal": samples["reads"].describe(),
    }).round(0).astype(int)
    print("Reads per sample (chosen prep, before the depth filter):")
    print(depth_summary.to_string())
    print(f"Samples below {MIN_READS:,} reads: before bloom removal "
          f"{(samples['reads_raw'] < MIN_READS).sum():,}, after {(samples['reads'] < MIN_READS).sum():,}")

    # Sample flow per cohort group. Samples without 16S data in the context
    # are missing from samples.tsv ("with 16S data").
    depth = samples.set_index("sample_name")["reads"]
    rows = []
    for path in sorted(COHORT_DIR.glob("*/*.tsv")):
        cohort = pd.read_csv(path, sep="\t", dtype=str)
        cohort["reads"] = cohort["sample_name"].map(depth)
        for group, g in cohort.groupby("group"):
            rows.append({
                "cohort": f"{path.parent.name}/{path.stem}",
                "group": group,
                "in cohort": len(g),
                "with 16S data": int(g["reads"].notna().sum()),
                f">= {MIN_READS} reads": int((g["reads"] >= MIN_READS).sum()),
            })

    print("\nSample flow per cohort:")
    print(pd.DataFrame(rows).to_string(index=False))
