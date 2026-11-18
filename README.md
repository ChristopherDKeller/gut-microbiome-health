# Microbial Networks

American Gut Project (Qiita study 10317) data for two related projects:

- `icw/`: **ICW**, Comparative Analysis of Human Gut Microbial Networks
- `fp/`: **FP**, Predicting Human Health Outcomes from the Gut Microbiome
- `shared/`: database, cohort building blocks, metadata pre-analysis

Data lives in `data/`, outputs in `results/{shared,icw,fp}/` (both git-ignored).
All paths are defined in `shared/paths.py`.

## Setup

```bash
conda activate agp
pip install -e .
```

## How to use

1. `python shared/db/build_db.py` downloads the metadata (if missing) and builds `data/AGP.duckdb`
2. Pre-analysis: `python shared/preanalysis/metadata_counts.py`, `python shared/preanalysis/metadata_missingness.py`
3. Feature-table: `python shared/features/fetch.py` downloads feature table for all stool samples
4. ICW cohorts: `python icw/cohorts.py` writes `data/cohorts/icw/*.tsv`
5. FP baseline model: `python fp/train.py`
