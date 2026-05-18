import re
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"

# All schema variants across 2017-2025
_COLUMN_MAP = {
    "last name": "last_name",
    "first name": "first_name",
    "salary paid": "salary",
    "salary": "salary",
    "taxable benefits": "benefits",
    "benefits": "benefits",
    "calendar year": "year",
    "year": "year",
    "sector": "sector",
    "employer": "employer",
    "job title": "job_title",
}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [_COLUMN_MAP.get(c.strip().lower(), c.strip().lower()) for c in df.columns]
    return df


def _clean_salary(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.replace(r"[\$,]", "", regex=True)
        .str.strip()
        .pipe(pd.to_numeric, errors="coerce")
    )


def _extract_first_name(raw: str) -> str:
    """Return the first usable given name, skipping leading initials like 'L.'."""
    if not isinstance(raw, str):
        return ""
    tokens = raw.strip().split()
    for token in tokens:
        # skip single-letter initials with or without a trailing dot
        if re.fullmatch(r"[A-Za-z]\.?", token):
            continue
        return token.title()
    return tokens[0].title() if tokens else ""


def load_all() -> pd.DataFrame:
    """Load and normalize every CSV in data/, return a single combined DataFrame."""
    frames = []
    for path in sorted(DATA_DIR.glob("*.csv")):
        df = pd.read_csv(path, encoding="utf-8-sig", dtype=str, low_memory=False)
        df = _normalize_columns(df)

        required = {"last_name", "first_name", "salary", "year", "sector", "employer", "job_title"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{path.name} is missing columns after normalization: {missing}")

        df["salary"] = _clean_salary(df["salary"])
        df["benefits"] = _clean_salary(df.get("benefits", pd.Series(dtype=str)))
        df["year"] = pd.to_numeric(df["year"].str.strip(), errors="coerce").astype("Int64")

        for col in ("last_name", "first_name", "sector", "employer", "job_title"):
            df[col] = df[col].astype(str).str.strip()

        df["first_name_clean"] = df["first_name"].apply(_extract_first_name)

        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna(subset=["salary", "year"])
    combined = combined[combined["salary"] > 0]
    return combined
