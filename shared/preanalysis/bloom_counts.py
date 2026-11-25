"""How many reads the bloom filter removes (Amir et al. 2017; decisions D3).

Uses the raw feature table (all sample-preps, no filtering), i.e. exactly what
bloom removal takes out. Prints to the console only:
- reads per bloom ASV (rank = order in Amir et al.'s list; top10 = their AGP recommendation)
- share of all reads removed by the top 10 (pooled and per sample)
- share held by ranks 11-20 (answers "why not all 20?")

Run:  python shared/preanalysis/bloom_counts.py
"""

import biom
import pandas as pd

from shared.paths import FEATURES_RAW_DIR, REFERENCE_DIR


def read_sequences(path) -> list[str]:
    return [line.strip().upper() for line in path.read_text().splitlines()
            if line.strip() and not line.startswith(">")]


if __name__ == "__main__":
    all20 = read_sequences(REFERENCE_DIR / "bloom_seqs_amir2017_all20.fna")
    top10 = set(read_sequences(REFERENCE_DIR / "bloom_seqs_amir2017_top10.fna"))

    table = biom.load_table(str(FEATURES_RAW_DIR / "table.biom"))
    features = set(table.ids(axis="observation"))
    total = table.sum()

    df = pd.DataFrame({"rank": range(1, len(all20) + 1), "sequence": all20})
    df["reads"] = [table.data(s, axis="observation", dense=False).sum() if s in features else 0
                   for s in df["sequence"]]
    df["share_of_all_reads"] = df["reads"] / total
    df["top10"] = df["sequence"].isin(top10)

    print(df.sort_values("reads", ascending=False)[["rank", "reads", "share_of_all_reads", "top10"]]
          .to_string(index=False, formatters={"reads": "{:,.0f}".format,
                                              "share_of_all_reads": "{:.3%}".format}))

    # Pooled = top-10 reads / all reads (dominated by deep samples);
    # per sample = top-10 fraction of each sample-prep.
    top10_reads = table.filter([s for s in top10 if s in features], axis="observation",
                               inplace=False).sum(axis="sample")
    depth = table.sum(axis="sample")
    per_sample = pd.Series(top10_reads[depth > 0] / depth[depth > 0])

    print(f"\nTop-10 removal: {top10_reads.sum() / total:.2%} of all reads "
          f"(per sample: mean {per_sample.mean():.2%}, median {per_sample.median():.2%})")
    print(f"Ranks 11-20 (not removed): {df.loc[~df['top10'], 'reads'].sum() / total:.2%} of all reads")
