"""Central project paths, independent of the current working directory."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Data
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
METADATA_DIR = RAW_DIR / "AGP-sample-information"
DB_PATH = DATA_DIR / "AGP.duckdb"
COHORT_DIR = DATA_DIR / "cohorts"

# Results
RESULTS_DIR = ROOT / "results"
SHARED_RESULTS = RESULTS_DIR / "shared"
ICW_RESULTS = RESULTS_DIR / "icw"
FP_RESULTS = RESULTS_DIR / "fp"

# Qiita study 10317 = American Gut Project
QIITA_STUDY_ID = 10317

# Microbiome features (see docs/decisions.md, D2)
REDBIOM_CONTEXT = "Deblur_2021.09-Illumina-16S-V4-150nt-ac8c0b"
FEATURES_RAW_DIR = RAW_DIR / "features" / REDBIOM_CONTEXT
REFERENCE_DIR = ROOT / "shared" / "reference"
