"""
Download gender name data from two sources:

  1. SSA (US Social Security Administration) — one txt per birth year, format: Name,M_or_F,Count
     Birth years 1950-2003 (covers ages ~22-75 for the 2016-2025 dataset).

  2. Ontario baby names — separate CSVs for female and male births registered in Ontario
     (1913-2024), format: Year/Année, Name/Nom, Frequency/Fréquence

Run once:
    python scripts/download_gender_data.py
"""

import io
import zipfile
from pathlib import Path

import requests

# --- SSA ---
SSA_URL       = "https://www.ssa.gov/oact/babynames/names.zip"
SSA_DIR       = Path(__file__).parent.parent / "data" / "gender" / "ssa"
BIRTH_YEAR_MIN = 1950
BIRTH_YEAR_MAX = 2003

# --- Ontario ---
ONTARIO_FEMALE_URL = (
    "https://data.ontario.ca/dataset/4d339626-98f9-49fe-aede-d64f03fa914f"
    "/resource/acc72e92-3100-4a04-8f5f-4fad9cd77cc5/download/baby_names_-_female_.csv"
)
ONTARIO_MALE_URL = (
    "https://data.ontario.ca/dataset/eb4c585c-6ada-4de7-8ff1-e876fb1a6b0b"
    "/resource/5d8b8ece-fa01-43c5-955b-4b642b28c559/download/baby_names_-_male.csv"
)
ONTARIO_DIR = Path(__file__).parent.parent / "data" / "gender" / "ontario"

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; research script)"}


def download_ssa():
    SSA_DIR.mkdir(parents=True, exist_ok=True)

    already = list(SSA_DIR.glob("yob*.txt"))
    if already:
        print(f"SSA data already present ({len(already)} files). Delete {SSA_DIR} to re-download.")
        return

    print("Downloading SSA baby names zip (~8 MB)...")
    resp = requests.get(SSA_URL, headers=HEADERS, timeout=60)
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        for name in zf.namelist():
            if not (name.startswith("yob") and name.endswith(".txt")):
                continue
            try:
                year = int(name[3:7])
            except ValueError:
                continue
            if BIRTH_YEAR_MIN <= year <= BIRTH_YEAR_MAX:
                zf.extract(name, SSA_DIR)

    extracted = list(SSA_DIR.glob("yob*.txt"))
    print(f"Done. Extracted {len(extracted)} year files to {SSA_DIR}")


def download_ontario():
    ONTARIO_DIR.mkdir(parents=True, exist_ok=True)

    female_path = ONTARIO_DIR / "baby_names_female.csv"
    male_path   = ONTARIO_DIR / "baby_names_male.csv"

    if female_path.exists() and male_path.exists():
        print(f"Ontario data already present. Delete {ONTARIO_DIR} to re-download.")
        return

    print("Downloading Ontario baby names (female, 1913-2024)...")
    resp = requests.get(ONTARIO_FEMALE_URL, headers=HEADERS, timeout=60)
    resp.raise_for_status()
    female_path.write_bytes(resp.content)

    print("Downloading Ontario baby names (male, 1917-2024)...")
    resp = requests.get(ONTARIO_MALE_URL, headers=HEADERS, timeout=60)
    resp.raise_for_status()
    male_path.write_bytes(resp.content)

    print(f"Done. Ontario data saved to {ONTARIO_DIR}")


if __name__ == "__main__":
    download_ssa()
    download_ontario()
