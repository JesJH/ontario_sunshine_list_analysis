"""
Download SSA baby names data and NLTK names corpus.

SSA data: one txt per birth year, format: Name,M_or_F,Count
We keep birth years 1950-2003 (covers ages ~22-75 for the 2017-2025 dataset).

Run once:
    python scripts/download_gender_data.py
"""

import io
import zipfile
from pathlib import Path

import requests

SSA_URL = "https://www.ssa.gov/oact/babynames/names.zip"
BIRTH_YEAR_MIN = 1950
BIRTH_YEAR_MAX = 2003

OUT_DIR = Path(__file__).parent.parent / "data" / "gender" / "ssa"


def download_ssa():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    already = list(OUT_DIR.glob("yob*.txt"))
    if already:
        print(f"SSA data already present ({len(already)} files). Delete {OUT_DIR} to re-download.")
        return

    print("Downloading SSA baby names zip (~8 MB)...")
    resp = requests.get(SSA_URL, timeout=60)
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        for name in zf.namelist():
            match = name.startswith("yob") and name.endswith(".txt")
            if not match:
                continue
            try:
                year = int(name[3:7])
            except ValueError:
                continue
            if BIRTH_YEAR_MIN <= year <= BIRTH_YEAR_MAX:
                zf.extract(name, OUT_DIR)

    extracted = list(OUT_DIR.glob("yob*.txt"))
    print(f"Done. Extracted {len(extracted)} year files to {OUT_DIR}")


def download_nltk():
    import nltk
    nltk.download("names", quiet=True)
    print("NLTK names corpus ready.")


if __name__ == "__main__":
    download_ssa()
    download_nltk()
