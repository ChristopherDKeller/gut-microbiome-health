"""Download the AGP metadata from Qiita (if missing) and load it into DuckDB.

This is the first script to run. All other scripts read from the database.
"""

import subprocess
import zipfile
from pathlib import Path

import duckdb

from shared.paths import DB_PATH, METADATA_DIR, QIITA_STUDY_ID, RAW_DIR

TABLE_NAME = "agp_metadata"

DOWNLOAD_URL = (
    "https://qiita.ucsd.edu/public_download/"
    f"?data=sample_information&study_id={QIITA_STUDY_ID}"
)
ZIP_PATH = RAW_DIR / "AGP-sample-information.zip"


# ============================================================
# 1. Download and unpack metadata
# ============================================================

def find_metadata_file() -> Path | None:
    """Return the (alphabetically first) non-empty metadata file, if present."""
    files = sorted(
        path for path in METADATA_DIR.glob(f"{QIITA_STUDY_ID}*.txt")
        if path.is_file() and path.stat().st_size > 0
    )
    return files[0] if files else None


def target_path(member_name: str) -> Path | None:
    """Map a ZIP member to its extraction path; strip the 'templates/' prefix."""
    relative = Path(member_name)
    if relative.parts and relative.parts[0] == "templates":
        relative = Path(*relative.parts[1:])
    if not relative.parts:
        return None

    target = (METADATA_DIR / relative).resolve()
    if not target.is_relative_to(METADATA_DIR.resolve()):
        raise RuntimeError(f"Unsafe ZIP path: {member_name}")
    return target


def download_metadata() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print("Downloading AGP metadata ...")
    subprocess.run(["wget", "-O", str(ZIP_PATH), DOWNLOAD_URL], check=True)

    if not ZIP_PATH.exists() or ZIP_PATH.stat().st_size == 0:
        raise RuntimeError("Download failed: ZIP missing or empty.")


def unpack_metadata() -> None:
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    print("Unpacking ZIP ...")

    with zipfile.ZipFile(ZIP_PATH) as archive:
        members = [m for m in archive.infolist() if not m.is_dir()]

        # Validate all paths before writing anything.
        targets = [(m, target_path(m.filename)) for m in members]

        for member, target in targets:
            if target is None:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as src, target.open("wb") as dst:
                dst.write(src.read())

    # Delete ZIP only after successful extraction.
    ZIP_PATH.unlink()


def ensure_metadata() -> Path:
    path = find_metadata_file()
    if path is not None:
        print(f"Metadata already present: {path}")
        return path

    if not ZIP_PATH.exists():
        download_metadata()
    unpack_metadata()

    path = find_metadata_file()
    if path is None:
        raise FileNotFoundError(f"No metadata file found in {METADATA_DIR}")
    return path


# ============================================================
# 2. Build database
# ============================================================

def build_db(metadata_path: Path) -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB_PATH))

    try:
        # Import all columns as text to preserve metadata values.
        con.execute(
            f"""
            CREATE OR REPLACE TABLE {TABLE_NAME} AS
            SELECT *
            FROM read_csv(
                ?,
                delim = '\\t',
                header = true,
                all_varchar = true,
                nullstr = ''
            )
            """,
            [str(metadata_path.resolve())],
        )

        n_rows = con.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}").fetchone()[0]
        n_cols = len(con.execute(f"DESCRIBE {TABLE_NAME}").fetchall())
        print(f"Database created: {DB_PATH}")
        print(f"Table {TABLE_NAME}: {n_rows:,} rows, {n_cols:,} columns")
    finally:
        con.close()


if __name__ == "__main__":
    build_db(ensure_metadata())
