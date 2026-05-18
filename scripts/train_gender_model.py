"""
Train and evaluate the character n-gram gender model on SSA data.

Run after download_gender_data.py:
    python scripts/train_gender_model.py

Outputs:
    data/gender/char_model.pkl   — saved model (also used automatically by GenderClassifier)
    Prints accuracy, classification report, and a few example predictions.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.model_selection import cross_val_score

from src.gender_inference import (
    MODEL_CACHE,
    _build_ssa_lookup,
    _spaced,
    _train_char_model,
)

import joblib


def main():
    print("Loading SSA data...")
    ssa = _build_ssa_lookup()
    if not ssa:
        print("No SSA data found. Run scripts/download_gender_data.py first.")
        return

    print(f"  {len(ssa):,} unique names loaded.")

    names = [_spaced(n) for n in ssa]
    labels = [1 if p >= 0.5 else 0 for p in ssa.values()]
    weights = [abs(p - 0.5) * 2 + 0.01 for p in ssa.values()]

    print("Training char-gram model...")
    model = _train_char_model(ssa)

    print("\n5-fold cross-validation accuracy:")
    scores = cross_val_score(model, names, labels, cv=5, scoring="accuracy",
                             fit_params={"clf__sample_weight": weights})
    print(f"  {scores.round(4)} → mean {scores.mean():.4f} ± {scores.std():.4f}")

    print("\nExample predictions:")
    test_names = ["Emma", "James", "Alex", "Priya", "Wei", "Jordan", "Fatima", "Michael"]
    for name in test_names:
        p = model.predict_proba([_spaced(name)])[0][1]
        label = "Female" if p >= 0.80 else ("Male" if p <= 0.20 else "Uncertain")
        print(f"  {name:<12} P(female)={p:.3f}  → {label}")

    MODEL_CACHE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_CACHE)
    print(f"\nModel saved to {MODEL_CACHE}")


if __name__ == "__main__":
    main()
