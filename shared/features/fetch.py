"""Download the raw Deblur feature table for all AGP fecal samples via redbiom.

Step 1 of the feature pipeline (fetch -> process -> load). The table is stored
unchanged: no bloom removal, no prep resolution, no filtering.

Output: data/raw/features/<context>/
    samples.tsv        every fecal sample in the DB, whether it is in the
                       context, and its Qiita prep IDs
    batches/           one .biom per batch (lets an interrupted run resume)
    table.biom         merged table; sample IDs are "<sample_name>.<prep_id>",
                       so a sample with several preps has several columns
    fetch_info.json    context, date, redbiom version, counts

Why only fecal samples, not the cohorts: the raw table must not depend on
cohort definitions; cohorts select from it later.

Redbiom data is "not fixed" (github.com/knightlab-analyses/american-gut-analyses,
README), so a later fetch can differ.
"""

import json
from datetime import datetime, timezone
from importlib.metadata import version

import biom
import numpy as np
import pandas as pd
import redbiom.fetch
from biom.util import biom_open
from scipy import sparse

from shared.cohorts import FECAL, load_metadata, matches_criteria
from shared.paths import FEATURES_RAW_DIR, QIITA_STUDY_ID, REDBIOM_CONTEXT

BATCH_SIZE = 500  # ~2 min per batch in a test fetch (2026-11-17)

SAMPLES_PATH = FEATURES_RAW_DIR / "samples.tsv"
BATCH_DIR = FEATURES_RAW_DIR / "batches"
TABLE_PATH = FEATURES_RAW_DIR / "table.biom"
INFO_PATH = FEATURES_RAW_DIR / "fetch_info.json"


# ============================================================
# 1. Which fecal samples are in the context?
# ============================================================

def fecal_sample_names() -> pd.Series:
    metadata = load_metadata()
    fecal = metadata.loc[matches_criteria(metadata, FECAL), "sample_name"]
    return fecal.astype(str).drop_duplicates().sort_values().reset_index(drop=True)


def context_preps() -> pd.DataFrame:
    """All (prep_id, sample_name) pairs of this study in the context."""
    # "Unambiguous" IDs look like "<prep_id>_<sample_name>"
    ids = redbiom.fetch.samples_in_context(REDBIOM_CONTEXT, unambiguous=True)
    ids = pd.Series(sorted(ids), dtype=str)
    split = ids.str.split("_", n=1, expand=True)
    preps = pd.DataFrame({"prep_id": split[0], "sample_name": split[1]})
    return preps.loc[preps["sample_name"].str.startswith(f"{QIITA_STUDY_ID}.")]


def build_sample_overview() -> pd.DataFrame:
    fecal = fecal_sample_names()
    preps = context_preps()
    preps = preps.loc[preps["sample_name"].isin(set(fecal))]

    prep_lists = preps.groupby("sample_name")["prep_id"].agg(
        lambda p: ",".join(sorted(p))
    )
    overview = pd.DataFrame({"sample_name": fecal})
    overview["prep_ids"] = overview["sample_name"].map(prep_lists)
    overview["in_context"] = overview["prep_ids"].notna()
    overview["n_preps"] = overview["prep_ids"].str.count(",").add(1).fillna(0).astype(int)
    return overview


# ============================================================
# 2. Fetch in batches
# ============================================================

def fetch_batches(samples: list[str]) -> list:
    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    batches = [samples[i:i + BATCH_SIZE] for i in range(0, len(samples), BATCH_SIZE)]
    paths = []

    for i, batch in enumerate(batches):
        path = BATCH_DIR / f"batch_{i:03d}.biom"
        paths.append(path)
        if path.exists():
            continue

        print(f"  batch {i + 1}/{len(batches)} ({len(batch)} samples)", flush=True)
        # Taxonomy is assigned separately (decisions D5), so skip it here
        table, _ = redbiom.fetch.data_from_samples(
            REDBIOM_CONTEXT, batch, skip_taxonomy=True
        )
        tmp = path.with_suffix(".tmp")
        with biom_open(str(tmp), "w") as f:
            table.to_hdf5(f, generated_by=f"redbiom {REDBIOM_CONTEXT}")
        tmp.rename(path)  # only complete batches count as done

    return paths


def merge_batches(paths: list) -> biom.Table:
    """Merge batch tables into one sparse table.

    biom.Table.concat was killed for running out of memory (WSL, 7.8 GB) on the full fetch
    (~900k features x 26k sample-preps, 2026-11-18). Instead, collect all
    non-zero entries as (feature, sample, count) triplets with global indices
    and build the sparse matrix once.
    """
    feature_index: dict[str, int] = {}
    sample_ids: list[str] = []
    rows, cols, values = [], [], []

    for path in paths:
        table = biom.load_table(str(path))
        coo = table.matrix_data.tocoo()  # features x samples

        batch_features = table.ids(axis="observation")
        global_rows = np.array(
            [feature_index.setdefault(f, len(feature_index)) for f in batch_features],
            dtype=np.int64,
        )
        rows.append(global_rows[coo.row])
        cols.append(coo.col + len(sample_ids))
        values.append(coo.data)
        sample_ids.extend(table.ids(axis="sample"))

    matrix = sparse.csr_matrix(
        (np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
        shape=(len(feature_index), len(sample_ids)),
    )
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Duplicate sample IDs across batches")

    return biom.Table(matrix, list(feature_index), sample_ids)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    if TABLE_PATH.exists():
        print(f"Already fetched: {TABLE_PATH}  (delete it to fetch again)")
        raise SystemExit

    FEATURES_RAW_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Context: {REDBIOM_CONTEXT}")
    overview = build_sample_overview()
    overview.to_csv(SAMPLES_PATH, sep="\t", index=False)

    in_context = overview.loc[overview["in_context"], "sample_name"].tolist()
    print(f"Fecal samples in DB:  {len(overview):,}")
    print(f"  in context:         {len(in_context):,}")
    print(f"  not in context:     {(~overview['in_context']).sum():,}")
    print(f"  with >1 prep:       {(overview['n_preps'] > 1).sum():,}")

    paths = fetch_batches(in_context)
    table = merge_batches(paths)

    with biom_open(str(TABLE_PATH), "w") as f:
        table.to_hdf5(f, generated_by=f"redbiom {REDBIOM_CONTEXT}")

    n_samples, n_features = table.shape[1], table.shape[0]
    batch_times = [datetime.fromtimestamp(p.stat().st_mtime, timezone.utc) for p in paths]
    info = {
        "context": REDBIOM_CONTEXT,
        "qiita_study_id": QIITA_STUDY_ID,
        # Batches can be downloaded over several runs; record their time span
        "downloaded_first_utc": min(batch_times).isoformat(timespec="seconds"),
        "downloaded_last_utc": max(batch_times).isoformat(timespec="seconds"),
        "merged_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "redbiom_version": version("redbiom"),
        "biom_format_version": version("biom-format"),
        "fecal_samples_in_db": len(overview),
        "samples_requested": len(in_context),
        "table_columns_sample_preps": n_samples,
        "table_features": n_features,
    }
    INFO_PATH.write_text(json.dumps(info, indent=2))

    print(f"\nSaved {TABLE_PATH}: {n_features:,} features x {n_samples:,} sample-preps")
    print("Batches can be deleted once table.biom is checked:", BATCH_DIR)
