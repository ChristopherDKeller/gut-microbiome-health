import duckdb

from shared.paths import DB_PATH
import pandas as pd

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

from fp.features import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, TARGET, GROUP_COL
from fp.preprocessing import clean_missing_values, clean_numeric_values


# --------------------------------------------------
# 1. Load data
# --------------------------------------------------

con = duckdb.connect(str(DB_PATH), read_only=True)

df = con.execute("""
    SELECT *
    FROM agp_metadata
    WHERE lower(trim(host_body_site)) = 'uberon:feces'
""").df()

con.close()


# --------------------------------------------------
# 2. Select features and target
# --------------------------------------------------

feature_cols = [col for col in FEATURES if col in df.columns]

# DECISION: Only these two target responses are considered valid labels.
y_all = (
    df[TARGET]
    .astype("string")
    .str.strip()
    .map({
        "Diagnosed by a medical professional (doctor, physician assistant)": 1,
        "I do not have this condition": 0,
    })
)

# Keep only labeled rows with a subject ID.
df = df.loc[y_all.notna() & df[GROUP_COL].notna()].copy()
df["y"] = y_all.loc[df.index].astype(int)

# DECISION: Use one randomly selected sample per subject.
# This prevents the same subject from appearing multiple times.
df = (
    df.sample(frac=1, random_state=42)
    .drop_duplicates(GROUP_COL)
    .reset_index(drop=True)
)

# Preprocess categorical and numeric features separately, then combine them.
X_cat = clean_missing_values(df[CATEGORICAL_FEATURES])
X_num = clean_numeric_values(df[NUMERIC_FEATURES])

X = pd.concat([X_cat, X_num], axis=1)
y = df["y"]

print(f"Number of samples: {X.shape[0]}")
print(f"Number of features: {X.shape[1]}")


# --------------------------------------------------
# 3. Repeated stratified cross-validation
# --------------------------------------------------

# DECISION: 5 folds, repeated 3 times (15 evaluation runs).
# DECISION: Stratification preserves the class proportions in each fold.
# DECISION: Fixed random state for reproducibility.
cv = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=3,
    random_state=42,
)


# --------------------------------------------------
# 4. Preprocessing
# --------------------------------------------------

# DECISION: Replace missing categorical values with the most frequent value.
# DECISION: One-hot encode categorical features.
# DECISION: Ignore categories that only occur in the test fold.
# DECISION: Replace missing numerical values with the median.
# DECISION: Standardize numerical features.
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            make_pipeline(
                SimpleImputer(strategy="most_frequent"),
                OneHotEncoder(handle_unknown="ignore"),
            ),
            CATEGORICAL_FEATURES,
        ),
        (
            "numeric",
            make_pipeline(
                SimpleImputer(strategy="median"),
                StandardScaler(),
            ),
            NUMERIC_FEATURES,
        ),
    ],
    remainder="drop",
)


# --------------------------------------------------
# 5. Model and evaluation
# --------------------------------------------------

# DECISION: L1 regularization.
# DECISION: C = 1.0 (default regularization strength).
# DECISION: class_weight=None (no class weighting).
# DECISION: max_iter=1000 (maximum optimization iterations).
# DECISION: solver='saga'.
def create_model():
    return make_pipeline(
        preprocessor,
        LogisticRegression(
            C=1.0,
            #l1_ratio=1.0,
            class_weight="balanced",
            max_iter=1000,
            solver="lbfgs",
            random_state=42,
        ),
    )


results = []

for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model = create_model()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    results.append({
        "fold": fold,
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "pr_auc": average_precision_score(y_test, y_prob),
        "n_positive": int(y_test.sum()),
    })


# --------------------------------------------------
# 6. Results
# --------------------------------------------------

results_df = pd.DataFrame(results)

metrics = [
    "precision",
    "recall",
    "f1",
    "accuracy",
    "roc_auc",
    "pr_auc",
]

summary = pd.DataFrame({
    "mean": results_df[metrics].mean(),
    "std": results_df[metrics].std(),
})

print("\nResults across 15 evaluation runs:")
print(summary.to_string(float_format=lambda x: f"{x:.4f}"))

print("\nPositive cases per test fold:")
print(
    f"Min: {results_df['n_positive'].min()}, "
    f"Max: {results_df['n_positive'].max()}, "
    f"Mean: {results_df['n_positive'].mean():.1f}"
)

print("\nIndividual fold results:")
print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
