"""
Ensemble gender classifier — three complementary sources:

  1. SSA lookup       (60% weight) — count-based P(female) for names in the 1950-2003 SSA data
  2. Char-gram model  (30% weight) — logistic regression on character n-grams trained on SSA data;
                                     generalises to names not in SSA (e.g. South Asian, East Asian names)
  3. HuggingFace model(10% weight) — optional; a text-classification model for name→gender;
                                     configure via hf_model_name= on GenderClassifier()

Weights are renormalised at inference time if a source returns no signal for a name.

Final labels:
  P(female) >= 0.70  → "Female"
  P(female) <= 0.30  → "Male"
  otherwise          → "Uncertain"
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

SSA_DIR     = Path(__file__).parent.parent / "data" / "gender" / "ssa"
ONTARIO_DIR = Path(__file__).parent.parent / "data" / "gender" / "ontario"
MODEL_CACHE = Path(__file__).parent.parent / "data" / "gender" / "char_model.pkl"

BIRTH_YEAR_MIN = 1950
BIRTH_YEAR_MAX = 2003

FEMALE_THRESHOLD = 0.70
MALE_THRESHOLD = 0.30

_BASE_WEIGHTS = {"ssa": 0.50, "ontario": 0.20, "char_model": 0.20, "hf": 0.10}


# ---------------------------------------------------------------------------
# Source 1: SSA lookup
# ---------------------------------------------------------------------------

def _build_ssa_lookup() -> dict[str, float]:
    """Return name → P(female) from SSA birth-year data."""
    if not SSA_DIR.exists():
        return {}

    counts: dict[str, dict] = {}
    for path in SSA_DIR.glob("yob*.txt"):
        try:
            year = int(path.stem[3:])
        except ValueError:
            continue
        if not (BIRTH_YEAR_MIN <= year <= BIRTH_YEAR_MAX):
            continue
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                name, gender, n = line.strip().split(",")
                name = name.title()
                if name not in counts:
                    counts[name] = {"F": 0, "M": 0}
                counts[name][gender] += int(n)

    return {
        name: c["F"] / (c["F"] + c["M"])
        for name, c in counts.items()
        if c["F"] + c["M"] > 0
    }


# ---------------------------------------------------------------------------
# Source 2: Ontario baby names lookup
# ---------------------------------------------------------------------------

def _build_ontario_lookup() -> dict[str, float]:
    """Return name → P(female) from Ontario birth registration data (1950-2003)."""
    female_path = ONTARIO_DIR / "baby_names_female.csv"
    male_path   = ONTARIO_DIR / "baby_names_male.csv"

    if not female_path.exists() or not male_path.exists():
        return {}

    def _load(path: Path) -> pd.Series:
        raw = pd.read_csv(path, dtype=str)
        raw.columns = ["year", "name", "freq"]
        raw["year"] = pd.to_numeric(raw["year"], errors="coerce")
        raw["freq"] = pd.to_numeric(raw["freq"], errors="coerce").fillna(0)
        raw = raw[(raw["year"] >= BIRTH_YEAR_MIN) & (raw["year"] <= BIRTH_YEAR_MAX)]
        raw["name"] = raw["name"].str.title()
        return raw.groupby("name")["freq"].sum()

    female_counts = _load(female_path)
    male_counts   = _load(male_path)

    all_names = female_counts.index.union(male_counts.index)
    f = female_counts.reindex(all_names, fill_value=0)
    m = male_counts.reindex(all_names, fill_value=0)
    total = f + m
    return (f / total)[total > 0].to_dict()


# ---------------------------------------------------------------------------
# Source 3: Character n-gram logistic regression trained on SSA data
# ---------------------------------------------------------------------------

def _spaced(name: str) -> str:
    """Insert spaces between characters so TfidfVectorizer char n-grams work cleanly."""
    return " ".join(name.lower())


def _train_char_model(ssa_lookup: dict[str, float]) -> Pipeline:
    """Train a char-gram logistic regression using SSA names as labelled examples."""
    names, labels, weights = [], [], []
    for name, p_female in ssa_lookup.items():
        names.append(_spaced(name))
        labels.append(1 if p_female >= 0.5 else 0)
        # weight by sqrt of total implied count; we only have p so use confidence as proxy
        weights.append(abs(p_female - 0.5) * 2 + 0.01)  # range (0.01, 1.01]

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2)),
        ("clf", LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")),
    ])
    pipe.fit(names, labels, clf__sample_weight=weights)
    return pipe


def _load_or_train_char_model(ssa_lookup: dict[str, float]) -> Optional[Pipeline]:
    if not ssa_lookup:
        return None
    if MODEL_CACHE.exists():
        return joblib.load(MODEL_CACHE)
    print("  Training character n-gram model on SSA data...")
    model = _train_char_model(ssa_lookup)
    MODEL_CACHE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_CACHE)
    print(f"  Model cached to {MODEL_CACHE}")
    return model


# ---------------------------------------------------------------------------
# Source 3: HuggingFace text-classification pipeline (optional)
# ---------------------------------------------------------------------------

def _build_hf_classifier(model_name: str):
    """
    Load a HuggingFace text-classification pipeline for name→gender.

    Search https://huggingface.co/models?pipeline_tag=text-classification&q=gender+name
    for suitable models. The pipeline must return labels containing 'male'/'female'
    (case-insensitive) in the label field.
    """
    try:
        from transformers import pipeline
        clf = pipeline("text-classification", model=model_name)
        return clf
    except Exception as e:
        print(f"  HuggingFace model '{model_name}' could not be loaded: {e}")
        return None


def _hf_prob(clf, name: str) -> Optional[float]:
    """Return P(female) from a HuggingFace text-classification pipeline."""
    try:
        result = clf(name, top_k=None)
        for item in result:
            label = item["label"].lower()
            score = item["score"]
            if "female" in label:
                return score
            if "male" in label:
                return 1.0 - score
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------

class GenderClassifier:
    def __init__(self, hf_model_name: Optional[str] = None):
        """
        Args:
            hf_model_name: Optional HuggingFace model ID for name→gender classification.
                           If None, the HF source is skipped and weights are renormalised.
        """
        print("Loading SSA lookup...")
        self._ssa = _build_ssa_lookup()
        print(f"  {len(self._ssa):,} unique names loaded from SSA data.")

        print("Loading Ontario lookup...")
        self._ontario = _build_ontario_lookup()
        print(f"  {len(self._ontario):,} unique names loaded from Ontario data.")

        print("Loading / training char-gram model...")
        self._char_model = _load_or_train_char_model(self._ssa)
        print(f"  Char model: {'OK' if self._char_model else 'unavailable (no SSA data)'}")

        self._hf = None
        if hf_model_name:
            print(f"Loading HuggingFace model '{hf_model_name}'...")
            self._hf = _build_hf_classifier(hf_model_name)
            print(f"  HF model: {'OK' if self._hf else 'unavailable'}")

    # ------------------------------------------------------------------
    # Per-source probabilities
    # ------------------------------------------------------------------

    def _ssa_prob(self, name: str) -> Optional[float]:
        return self._ssa.get(name.title())

    def _ontario_prob(self, name: str) -> Optional[float]:
        return self._ontario.get(name.title())

    def _char_prob(self, name: str) -> Optional[float]:
        if self._char_model is None:
            return None
        p = self._char_model.predict_proba([_spaced(name)])[0][1]
        return float(p)

    def _hf_prob(self, name: str) -> Optional[float]:
        if self._hf is None:
            return None
        return _hf_prob(self._hf, name)

    # ------------------------------------------------------------------
    # Ensemble
    # ------------------------------------------------------------------

    def predict_proba(self, name: str) -> Optional[float]:
        """Weighted P(female) for one first name, or None if no source has any signal."""
        signals = {
            "ssa":        self._ssa_prob(name),
            "ontario":    self._ontario_prob(name),
            "char_model": self._char_prob(name),
            "hf":         self._hf_prob(name),
        }
        available = {k: v for k, v in signals.items() if v is not None}
        if not available:
            return None

        total_w = sum(_BASE_WEIGHTS[k] for k in available)
        return sum(_BASE_WEIGHTS[k] * v for k, v in available.items()) / total_w

    def classify(self, name: str) -> tuple[str, Optional[float]]:
        """Return (label, confidence). Confidence = max(p, 1-p) ∈ [0.5, 1.0]."""
        p = self.predict_proba(name)
        if p is None:
            return "Uncertain", None
        confidence = float(max(p, 1 - p))
        if p >= FEMALE_THRESHOLD:
            return "Female", confidence
        if p <= MALE_THRESHOLD:
            return "Male", confidence
        return "Uncertain", confidence

    def classify_series(self, names: pd.Series) -> pd.DataFrame:
        """
        Classify a Series of first names.
        Returns a DataFrame with columns ['gender', 'gender_confidence', 'gender_p_female'].
        """
        unique_names = [n for n in names.dropna().unique() if isinstance(n, str) and n]

        cache: dict[str, tuple] = {}
        for name in unique_names:
            label, conf = self.classify(name)
            p = self.predict_proba(name)
            cache[name] = (label, conf, p)

        def _lookup(n):
            return cache.get(n, ("Uncertain", None, None))

        rows = names.map(_lookup).tolist()
        return pd.DataFrame(rows, index=names.index,
                            columns=["gender", "gender_confidence", "gender_p_female"])
