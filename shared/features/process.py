"""Process the raw feature table into one ASV count table for all fecal samples.

Step 2 of the feature pipeline (fetch -> process -> load). Steps, in order:

1. Remove the top-10 bloom ASVs.
2. Resolve samples with several preps: keep the prep with the most reads
   after bloom removal.
3. Drop samples with fewer than MIN_READS reads after bloom removal.
4. Drop ASVs with zero reads in the remaining samples.

Order follows McDonald et al. 2018 ("16S data processing"): bloom removal
before any depth filtering. Prep resolution comes after bloom removal so the
choice is based on non-bloom reads.

NOT done here (cohort- or analysis-specific): prevalence filtering,
rarefaction/normalization, collapsing to taxonomic levels. Taxonomy (D5) is a
separate step once the reference is decided.

Input:  data/raw/features/<context>/table.biom          (fetch.py)
Output: data/features/asv_counts.biom    ASVs x samples, IDs = sample_name
        data/features/samples.tsv        per sample: chosen prep, reads, bloom share
        results/shared/feature_qc.tsv    samples / features / reads after each step
"""

import biom
import pandas as pd
from biom.util import biom_open

from shared.paths import (
    FEATURES_DIR,
    FEATURES_RAW_DIR,
    REFERENCE_DIR,
    SHARED_RESULTS,
)

BLOOM_FASTA = REFERENCE_DIR / "bloom_seqs_amir2017_top10.fna"  # D3

# D8: McDonald et al. 2018 used samples with "at least 1,250 16S sequences/sample"
# after bloom removal ("removing exact sOTUs from the data set prior to rarefaction").
# This is only a minimum depth; whether to rarefy is a separate decision (D10).
MIN_READS = 1250

RAW_TABLE = FEATURES_RAW_DIR / "table.biom"
OUT_TABLE = FEATURES_DIR / "asv_counts.biom"
OUT_SAMPLES = FEATURES_DIR / "samples.tsv"
OUT_QC = SHARED_RESULTS / "feature_qc.tsv"


def read_sequences(path) -> list[str]:
    return [line.strip().upper() for line in path.read_text().splitlines()
            if line.strip() and not line.startswith(">")]


def qc_row(step: str, table: biom.Table, sample_names=None) -> dict:
    ids = pd.Series(table.ids())
    names = sample_names if sample_names is not None else ids.str.rsplit(".", n=1).str[0]
    return {
        "step": step,
        "sample_preps": table.shape[1],
        "samples": pd.Series(names).nunique(),
        "features": int((table.sum(axis="observation") > 0).sum()),
        "reads": int(table.sum()),
    }


if __name__ == "__main__":
    table = biom.load_table(str(RAW_TABLE))
    qc = [qc_row("raw", table)]

    # Per sample-prep bookkeeping
    preps = pd.DataFrame({"id": table.ids()})
    preps["sample_name"] = preps["id"].str.rsplit(".", n=1).str[0]
    preps["prep_id"] = preps["id"].str.rsplit(".", n=1).str[1]
    preps["reads_raw"] = table.sum(axis="sample")

    # 1. Bloom removal (exact sequence match; ASV IDs are the sequences)
    blooms = set(read_sequences(BLOOM_FASTA)) & set(table.ids(axis="observation"))
    table = table.filter(blooms, axis="observation", invert=True, inplace=False)
    preps["reads"] = table.sum(axis="sample")
    preps["bloom_fraction"] = 1 - preps["reads"] / preps["reads_raw"]
    qc.append(qc_row(f"bloom removal ({len(blooms)} ASVs)", table))

    # 2. One prep per sample: most reads after bloom removal; ties -> lowest prep_id
    preps["n_preps"] = preps.groupby("sample_name")["id"].transform("size")
    chosen = (
        preps.sort_values(["sample_name", "reads", "prep_id"], ascending=[True, False, True])
        .drop_duplicates("sample_name")
    )
    table = table.filter(set(chosen["id"]), axis="sample", inplace=False)
    qc.append(qc_row("one prep per sample (most reads)", table))

    # 3. Depth filter
    chosen["passed_depth"] = chosen["reads"] >= MIN_READS
    keep = chosen.loc[chosen["passed_depth"], "id"]
    table = table.filter(set(keep), axis="sample", inplace=False)
    qc.append(qc_row(f"depth >= {MIN_READS}", table))

    # 4. Drop ASVs without reads, relabel samples to plain sample_name
    table = table.remove_empty(axis="observation", inplace=False)
    table = table.update_ids(dict(zip(chosen["id"], chosen["sample_name"])),
                             axis="sample", inplace=False, strict=False)
    qc.append(qc_row("drop empty ASVs", table, sample_names=table.ids()))

    # Save
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    SHARED_RESULTS.mkdir(parents=True, exist_ok=True)
    with biom_open(str(OUT_TABLE), "w") as f:
        table.to_hdf5(f, generated_by="shared/features/process.py")

    chosen[["sample_name", "prep_id", "n_preps", "reads_raw", "reads",
            "bloom_fraction", "passed_depth"]].to_csv(OUT_SAMPLES, sep="\t", index=False)

    qc = pd.DataFrame(qc)
    qc.to_csv(OUT_QC, sep="\t", index=False)

    print(qc.to_string(index=False))
    print(f"\nSaved {OUT_TABLE}: {table.shape[0]:,} ASVs x {table.shape[1]:,} samples")
