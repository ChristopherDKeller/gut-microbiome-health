"""ICW cohorts: disease cases vs. controls for network comparison.

Output: data/cohorts/icw/<name>.tsv  (sample_name, host_subject_id, group)
        results/icw/cohort_summary.tsv
"""

import pandas as pd

from shared.cohorts import (
    FECAL,
    MISSING_VALUES,
    NO_RECENT_ANTIBIOTICS,
    assign_groups,
    load_metadata,
    no_other_conditions,
    save_cohort,
)
from shared.paths import ICW_RESULTS


# ============================================================
# Cohort definitions
# ============================================================

COHORTS = {
    "depression": {
        "base": FECAL
        + no_other_conditions(exclude=["mental_illness_type_depression"])
        + NO_RECENT_ANTIBIOTICS,
        "groups": {
            "case": [("mental_illness_type_depression", ["true"])],
            "control": [("mental_illness_type_depression", ["false"] + MISSING_VALUES)],
        },
    },
}


# ============================================================
# Build cohorts
# ============================================================

if __name__ == "__main__":
    metadata = load_metadata()
    print(f"Metadata loaded: {len(metadata):,} samples")

    summary_rows = []

    for name, definition in COHORTS.items():
        groups = assign_groups(metadata, definition["groups"], definition["base"])
        cohort = metadata.assign(group=groups)
        cohort = cohort.loc[cohort["group"].isin(definition["groups"].keys())]

        path = save_cohort(cohort, "icw", name)
        print(f"\n{name}: {path}")

        counts = groups.value_counts()
        for group, n in counts.items():
            summary_rows.append({
                "cohort": name,
                "group": group,
                "n_samples": int(n),
                "n_subjects": int(metadata.loc[groups == group, "host_subject_id"].nunique()),
            })

    summary = pd.DataFrame(summary_rows)
    ICW_RESULTS.mkdir(parents=True, exist_ok=True)
    summary.to_csv(ICW_RESULTS / "cohort_summary.tsv", sep="\t", index=False)

    print("\nCohort summary:")
    print(summary.to_string(index=False))
