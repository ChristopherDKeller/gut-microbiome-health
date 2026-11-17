"""Reusable building blocks for cohort definitions (used by ICW and FP).

A cohort is described by criteria: a list of (column, allowed_values) tuples.
All criteria must match (AND logic). Values are compared case-insensitively;
values in MISSING_VALUES are treated as missing (None).
"""

import duckdb
import pandas as pd

from shared.paths import COHORT_DIR, DB_PATH

TABLE_NAME = "agp_metadata"

# Common representations of missing values in metadata
MISSING_VALUES = [
    None,
    "",
    " ",
    "None",
    "none",
    "NULL",
    "null",
    "NaN",
    "nan",
    "N/A",
    "n/a",
    "not provided",
    "not collected",
    "not applicable",
]

NO_CONDITION = "I do not have this condition"


# ============================================================
# Reusable criteria
# ============================================================

FECAL = [
    ("host_body_site", ["UBERON:feces"]),
]

# Conditions answered with "Diagnosed by ..." / "I do not have this condition"
CONDITION_COLUMNS = [
    "lung_disease",
    "skin_condition",
    "alzheimers",
    "diabetes",
    "pku",
    "fungal_overgrowth",
    "kidney_disease",
    "migraine",
    "cardiovascular_disease",
    "ibd",
    "clinical_condition",
    "liver_disease",
    "autoimmune",
    "ibs",
    "cancer",
    "epilepsy_or_seizure_disorder",
    "add_adhd",
    "asd",
]

# Mental illness types answered with "true" / "false"
MENTAL_ILLNESS_COLUMNS = [
    "mental_illness_type_depression",
    "mental_illness_type_anorexia_nervosa",
    "mental_illness_type_schizophrenia",
    "mental_illness_type_bipolar_disorder",
    "mental_illness_type_substance_abuse",
    "mental_illness_type_ptsd_post_traumatic_stress_disorder",
    "mental_illness_type_unspecified",
    "mental_illness_type_bulimia_nervosa",
]

NO_RECENT_ANTIBIOTICS = [
    ("antibiotic_history",
     ["I have not taken antibiotics in the past year.", "Year"] + MISSING_VALUES),
]


def no_other_conditions(exclude: list[str] | tuple[str, ...] = ()) -> list:
    """Criteria: no diagnosed condition / mental illness (missing counts as absent).

    `exclude` lists the target columns, which are defined by the cohort itself.
    """
    criteria = [
        (column, [NO_CONDITION] + MISSING_VALUES)
        for column in CONDITION_COLUMNS
        if column not in exclude
    ]
    criteria += [
        (column, ["false"] + MISSING_VALUES)
        for column in MENTAL_ILLNESS_COLUMNS
        if column not in exclude
    ]
    return criteria


# ============================================================
# Matching
# ============================================================

_MISSING_NORMALIZED = {
    str(v).strip().casefold() for v in MISSING_VALUES if v is not None
}


def normalize_value(value):
    """Normalize values; configured missing values become None."""
    if value is None or pd.isna(value):
        return None

    normalized = str(value).strip().casefold()
    return None if normalized in _MISSING_NORMALIZED else normalized


def matches_criteria(data: pd.DataFrame, criteria: list) -> pd.Series:
    """Boolean mask of rows matching all criteria (AND logic)."""
    mask = pd.Series(True, index=data.index)

    for column, allowed_values in criteria:
        if column not in data.columns:
            raise KeyError(f"Metadata column missing: {column}")

        actual = data[column].map(normalize_value)
        allowed = {normalize_value(v) for v in allowed_values}
        mask &= actual.isin(allowed)

    return mask


def assign_groups(
    data: pd.DataFrame,
    groups: dict[str, list],
    base_criteria: list = (),
) -> pd.Series:
    """Assign each row to one group.

    Rows matching several groups are labelled 'conflict',
    rows matching none 'not_selected'.
    """
    base = matches_criteria(data, list(base_criteria))
    masks = {name: base & matches_criteria(data, crit) for name, crit in groups.items()}

    n_matches = sum(m.astype(int) for m in masks.values())
    result = pd.Series("not_selected", index=data.index, dtype="string")

    for name, mask in masks.items():
        result[mask & (n_matches == 1)] = name
    result[n_matches > 1] = "conflict"

    return result


# ============================================================
# Loading and saving
# ============================================================

def load_metadata(where: str | None = None) -> pd.DataFrame:
    """Load the metadata table from DuckDB (all columns as text)."""
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Database not found: {DB_PATH}. Run shared/db/build_db.py first."
        )

    sql = f"SELECT * FROM {TABLE_NAME}"
    if where:
        sql += f" WHERE {where}"

    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        return con.execute(sql).df()
    finally:
        con.close()


def one_sample_per_subject(
    data: pd.DataFrame,
    subject_col: str = "host_subject_id",
    random_state: int = 42,
) -> pd.DataFrame:
    """Keep one random sample per subject; drop rows without subject ID."""
    return (
        data.loc[data[subject_col].notna()]
        .sample(frac=1, random_state=random_state)
        .drop_duplicates(subject_col)
    )


def save_cohort(cohort: pd.DataFrame, project: str, name: str):
    """Save sample_name, host_subject_id and group to data/cohorts/<project>/<name>.tsv."""
    out_dir = COHORT_DIR / project
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.tsv"

    cohort[["sample_name", "host_subject_id", "group"]].to_csv(
        path, sep="\t", index=False
    )
    return path


def load_cohort(project: str, name: str) -> pd.DataFrame:
    return pd.read_csv(COHORT_DIR / project / f"{name}.tsv", sep="\t", dtype=str)
