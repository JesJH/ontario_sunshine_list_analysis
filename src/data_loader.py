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
    "jobtitle": "job_title",
}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [_COLUMN_MAP.get(c.strip().lower(), c.strip().lower()) for c in df.columns]
    return df


def _normalize_sector(series: pd.Series) -> pd.Series:
    """Collapse sector name variants that differ only in punctuation or spelling across years."""
    s = (
        series
        .str.strip()
        .str.rstrip("*")
        .str.strip()
        .str.replace("–", "-", regex=False)
        .str.replace(" & ", " and ", regex=False)
        .str.replace(r"\s+", " ", regex=True)
        .str.title()                            # consistent case across years
    )
    # Collapse all "Seconded (...)" variants into a single category
    s = s.where(~s.str.startswith("Seconded"), other="Seconded")
    return s


_TITLE_CANONICAL = {
    # Word-order variants without comma (comma variants handled by inversion regex below)
    "teacher elementary":  "elementary teacher",
    "teacher secondary":   "secondary teacher",
    "teacher primary":     "primary teacher",
    "nurse registered":    "registered nurse",
    "constable police":    "police constable",
}


def _normalize_job_title(series: pd.Series) -> pd.Series:
    """
    Produce a normalized job title for clustering and grouping.

    Steps applied in order:
    1. Lowercase, strip, collapse whitespace, normalize spaces around slashes
    2. Strip bilingual French suffixes after `/` (identified by accented characters)
    3. Invert HR comma-convention: "Nurse, Registered" -> "registered nurse"
    4. Apply canonical mapping for remaining word-order variants
    """
    s = (
        series.astype(str)
        .str.lower()
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.replace(r"\s*/\s*", "/", regex=True)
    )
    # Strip French bilingual suffix: "english/french" -> "english"
    # Triggered when the post-slash segment contains an accented character
    s = s.str.replace(r"/[^/]*[àâäéèêëîïôùûüçœæ][^/]*$", "", regex=True).str.strip()

    # Invert HR comma-convention: "Profession, Modifier" -> "modifier profession"
    mask = s.str.match(r"^.+,\s+\w+(\s+\w+){0,2}$")
    s = s.copy()
    s[mask] = s[mask].str.replace(r"^(.+),\s+(.+)$", r"\2 \1", regex=True)

    # Apply canonical mapping for known word-order variants
    s = s.map(lambda t: _TITLE_CANONICAL.get(t, t))
    return s


def _normalize_employer(series: pd.Series) -> pd.Series:
    """Collapse employer name variants that differ in case or punctuation across years."""
    return (
        series
        .str.strip()
        .str.replace("–", "-", regex=False)   # em-dash → hyphen
        .str.replace(r"\s+", " ", regex=True) # collapse double spaces
        .str.title()                           # consistent title case: "of" == "Of"
    )


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
        df["sector"]        = _normalize_sector(df["sector"])
        df["employer"]      = _normalize_employer(df["employer"])
        df["job_title_norm"] = _normalize_job_title(df["job_title"])

        df["first_name_clean"] = df["first_name"].apply(_extract_first_name)

        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna(subset=["salary", "year"])
    combined = combined[combined["salary"] > 0]
    return combined
