"""Benchmark: logistic regression penalty (L1 / L2 / elastic net) x class weighting.

Metadata-only IBD baseline (FP). Data preparation, preprocessing and CV are
the same as in fp/train.py; only the model configuration changes.

Design
------
- 6 configurations: penalty {L1, elastic net (l1_ratio=0.5), L2}
  x class_weight {None, "balanced"}.
- Solver: saga for L1 and elastic net (the only sklearn solver supporting
  them), lbfgs for L2.
- All configurations use the SAME folds (RepeatedStratifiedKFold, 5 x 3,
  random_state=42), so their results can be compared fold by fold.
- Metrics per fold: precision, recall, F1, accuracy (threshold 0.5),
  ROC-AUC, PR-AUC (threshold-free). Reported as mean +- SD over the 15 folds.
- The threshold is deliberately NOT tuned. Threshold-dependent metrics show
  the default 0.5 only; recall could later be raised by lowering the threshold.
- Extra columns: converged (did the solver reach tol within max_iter?),
  n_iter, n_nonzero (number of non-zero coefficients).

Known issue: saga with class_weight="balanced" does not converge on this data
(see docs/decisions.md). Those rows are kept and marked converged=False.

Output: results/fp/benchmarks/<YYYY-MM-DD>_penalty_x_weighting/
    folds.tsv      one row per configuration x fold
    summary.tsv    mean and SD per configuration
    run_info.json  date, git commit, package versions, n, prevalence, CV setup

Run:  python fp/benchmark.py
"""

import json
import subprocess
import time
import warnings
from datetime import date
from importlib.metadata import version

import duckdb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from fp.features import CATEGORICAL_FEATURES, GROUP_COL, NUMERIC_FEATURES, TARGET
from fp.preprocessing import clean_missing_values, clean_numeric_values
from shared.paths import DB_PATH, FP_RESULTS, ROOT

RANDOM_STATE = 42
N_SPLITS, N_REPEATS = 5, 3
MAX_ITER = 2000

# name -> LogisticRegression arguments
CONFIGS = {
    "L1 / None":               dict(l1_ratio=1.0, solver="saga",  class_weight=None),
    "L1 / balanced":           dict(l1_ratio=1.0, solver="saga",  class_weight="balanced"),
    "ElasticNet 0.5 / None":     dict(l1_ratio=0.5, solver="saga",  class_weight=None),
    "ElasticNet 0.5 / balanced": dict(l1_ratio=0.5, solver="saga",  class_weight="balanced"),
    "L2 / None":               dict(l1_ratio=0.0, solver="lbfgs", class_weight=None),
    "L2 / balanced":           dict(l1_ratio=0.0, solver="lbfgs", class_weight="balanced"),
}

METRICS = ["precision", "recall", "f1", "accuracy", "roc_auc", "pr_auc"]


# ============================================================
# 1. Data (same steps as fp/train.py)
# ============================================================

def load_data() -> tuple[pd.DataFrame, pd.Series]:
    # Only the needed columns: the full metadata table (~1,400 columns)
    cols = sorted(set(CATEGORICAL_FEATURES + NUMERIC_FEATURES + [TARGET, GROUP_COL]))
    con = duckdb.connect(str(DB_PATH), read_only=True)
    df = con.execute(
        f"SELECT {', '.join(cols)} FROM agp_metadata "
        "WHERE lower(trim(host_body_site)) = 'uberon:feces'"
    ).df()
    con.close()

    # Only these two answers are valid labels
    y = df[TARGET].astype("string").str.strip().map({
        "Diagnosed by a medical professional (doctor, physician assistant)": 1,
        "I do not have this condition": 0,
    })
    df = df.loc[y.notna() & df[GROUP_COL].notna()].copy()
    df["y"] = y.loc[df.index].astype(int)

    # One random sample per subject
    df = (
        df.sample(frac=1, random_state=RANDOM_STATE)
        .drop_duplicates(GROUP_COL)
        .reset_index(drop=True)
    )

    X = pd.concat(
        [clean_missing_values(df[CATEGORICAL_FEATURES]),
         clean_numeric_values(df[NUMERIC_FEATURES])],
        axis=1,
    )
    return X, df["y"]


# ============================================================
# 2. Model
# ============================================================

def create_model(config: dict):
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", make_pipeline(
                SimpleImputer(strategy="most_frequent"),
                OneHotEncoder(handle_unknown="ignore"),
            ), CATEGORICAL_FEATURES),
            ("numeric", make_pipeline(
                SimpleImputer(strategy="median"),
                StandardScaler(),
            ), NUMERIC_FEATURES),
        ],
        remainder="drop",
    )
    model = LogisticRegression(
        C=1.0, max_iter=MAX_ITER, random_state=RANDOM_STATE, **config
    )
    return make_pipeline(preprocessor, model)


# ============================================================
# 3. Evaluation
# ============================================================

def evaluate_fold(config: dict, X_train, y_train, X_test, y_test) -> dict:
    pipeline = create_model(config)

    start = time.time()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        pipeline.fit(X_train, y_train)
    seconds = time.time() - start

    model = pipeline[-1]
    y_pred = pipeline.predict(X_test)          # threshold 0.5
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    return {
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "pr_auc": average_precision_score(y_test, y_prob),
        "converged": not any(issubclass(w.category, ConvergenceWarning) for w in caught),
        "n_iter": int(np.max(model.n_iter_)),
        "n_nonzero": int((model.coef_ != 0).sum()),
        "fit_seconds": round(seconds, 1),
        "n_positive_test": int(y_test.sum()),
    }


def summarize(folds: pd.DataFrame) -> pd.DataFrame:
    grouped = folds.groupby("config", sort=False)
    summary = grouped[METRICS].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary["converged_folds"] = grouped["converged"].sum().astype(int).astype(str) + f"/{N_SPLITS * N_REPEATS}"
    summary["n_nonzero_mean"] = grouped["n_nonzero"].mean().round(0).astype(int)
    summary["fit_seconds_mean"] = grouped["fit_seconds"].mean().round(1)
    return summary.reset_index()


def git_commit() -> str:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                               capture_output=True, text=True, check=True).stdout.strip()
        return commit + ("-dirty" if dirty else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    X, y = load_data()
    print(f"Samples: {len(y):,}  positives: {y.sum():,} ({y.mean():.2%})  features: {X.shape[1]}")

    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS,
                                 random_state=RANDOM_STATE)
    splits = list(cv.split(X, y))  # identical folds for every configuration

    rows = []
    for name, config in CONFIGS.items():
        print(f"\n{name}", flush=True)
        for fold, (train_idx, test_idx) in enumerate(splits, start=1):
            result = evaluate_fold(config, X.iloc[train_idx], y.iloc[train_idx],
                                   X.iloc[test_idx], y.iloc[test_idx])
            rows.append({"config": name, "fold": fold, **result})
            print(f"  fold {fold:2d}: PR-AUC {result['pr_auc']:.3f}  "
                  f"converged={result['converged']}  ({result['fit_seconds']} s)", flush=True)

    folds = pd.DataFrame(rows)
    summary = summarize(folds)

    out_dir = FP_RESULTS / "benchmarks" / f"{date.today()}_penalty_x_weighting"
    out_dir.mkdir(parents=True, exist_ok=True)
    folds.to_csv(out_dir / "folds.tsv", sep="\t", index=False)
    summary.to_csv(out_dir / "summary.tsv", sep="\t", index=False, float_format="%.4f")
    (out_dir / "run_info.json").write_text(json.dumps({
        "date": str(date.today()),
        "git_commit": git_commit(),
        "sklearn": version("scikit-learn"),
        "target": TARGET,
        "n_samples": int(len(y)),
        "n_positive": int(y.sum()),
        "prevalence": round(float(y.mean()), 4),
        "pr_auc_random_baseline": round(float(y.mean()), 4),
        "n_input_features": int(X.shape[1]),
        "cv": f"RepeatedStratifiedKFold({N_SPLITS} x {N_REPEATS}), random_state={RANDOM_STATE}",
        "C": 1.0,
        "max_iter": MAX_ITER,
        "configs": {k: {**v, "class_weight": str(v["class_weight"])} for k, v in CONFIGS.items()},
    }, indent=2))

    print("\n" + summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nPR-AUC random baseline = prevalence = {y.mean():.3f}")
    print(f"Saved to {out_dir}")
