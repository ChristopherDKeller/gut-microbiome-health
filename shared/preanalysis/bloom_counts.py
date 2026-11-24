"""Read counts of the 20 bloom ASVs (Amir et al. 2017) in the raw feature table.

Counts are summed over all sample-preps (no filtering), i.e. what bloom removal
would take out. Bars are sorted by read count; blue = Amir et al.'s top 10.
Y labels are the bloom rank in Amir et al. (order of the FASTA file).

Output: results/shared/bloom_counts/bloom_counts.{tsv,png}
"""

import biom
import matplotlib.pyplot as plt
import pandas as pd

from shared.paths import FEATURES_RAW_DIR, REFERENCE_DIR, SHARED_RESULTS

OUT_DIR = SHARED_RESULTS / "bloom_counts"


def read_sequences(path) -> list[str]:
    return [line.strip().upper() for line in path.read_text().splitlines()
            if line.strip() and not line.startswith(">")]


if __name__ == "__main__":
    all20 = read_sequences(REFERENCE_DIR / "bloom_seqs_amir2017_all20.fna")
    top10 = set(read_sequences(REFERENCE_DIR / "bloom_seqs_amir2017_top10.fna"))

    table = biom.load_table(str(FEATURES_RAW_DIR / "table.biom"))
    features = set(table.ids(axis="observation"))

    df = pd.DataFrame({"rank": range(1, len(all20) + 1), "sequence": all20})
    df["reads"] = [table.data(s, axis="observation", dense=False).sum() if s in features else 0
                   for s in df["sequence"]]
    df["share_of_all_reads"] = df["reads"] / table.sum()
    df["top10"] = df["sequence"].isin(top10)
    df = df.sort_values("reads", ascending=False)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "bloom_counts.tsv", sep="\t", index=False)

    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
    ax.barh([f"Rank {r}" for r in df["rank"]], df["reads"].clip(lower=1),
            color=["#2a78d6" if t else "#b5b4ae" for t in df["top10"]])
    ax.invert_yaxis()  # largest at the top
    ax.set_xscale("log")
    ax.set_xlabel("Total reads (log scale)")
    ax.set_title(f"Bloom ASVs in {table.shape[1]:,} sample-preps (blue = top 10)", loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "bloom_counts.png")

    print(df[["rank", "reads", "share_of_all_reads", "top10"]].to_string(index=False))
