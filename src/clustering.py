"""
NLP job-title clustering pipeline.

  1. normalize_titles        — lowercase, collapse whitespace
  2. get_embeddings          — sentence-transformer embeddings, cached to disk
  3. k_selection_metrics     — WCSS, silhouette, BCSS/WCSS over a K range
  4. bootstrap_ari           — stability score for a given K
  5. cluster_kmeans          — K-Means (primary)
  6. cluster_hdbscan         — HDBSCAN (comparison; noise = -1)
  7. cluster_summary         — top titles + salary stats per cluster for human review
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

EMBED_CACHE_DIR = Path(__file__).parent.parent / "data" / "embeddings"
DEFAULT_MODEL = "all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# 1. Normalization
# ---------------------------------------------------------------------------

def normalize_titles(titles: pd.Series) -> pd.Series:
    """
    Lowercase, strip, collapse whitespace, and invert comma-separated HR title formats.

    Many Ontario Sunshine List titles use the convention "Profession, Modifier"
    (e.g. "Nurse, Registered", "Teacher, Secondary", "Officer, Administrative").
    This inverts them to "Modifier Profession" so they match the more common form
    and deduplicate with other years that used the reversed convention.
    Only inverts when the suffix is 1–3 words to avoid mangling complex titles
    with commas (e.g. "Government of Ontario, Ministry of Finance").
    """
    s = (
        titles.astype(str)
        .str.lower()
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )
    # Invert "prefix, short-suffix" → "short-suffix prefix"
    mask = s.str.match(r"^.+,\s+\w+(\s+\w+){0,2}$")
    inverted = s[mask].str.replace(r"^(.+),\s+(.+)$", r"\2 \1", regex=True)
    s = s.copy()
    s[mask] = inverted
    return s


# ---------------------------------------------------------------------------
# 2. Embeddings (cached)
# ---------------------------------------------------------------------------

def get_embeddings(
    unique_titles: list[str],
    model_name: str = DEFAULT_MODEL,
    cache_path: Optional[Path] = None,
    batch_size: int = 256,
    show_progress: bool = True,
) -> np.ndarray:
    """
    Return L2-normalized embeddings for each title in unique_titles (same order).
    Loads from cache if the cached title list matches exactly; otherwise encodes.
    """
    if cache_path is None:
        slug = model_name.replace("/", "_")
        cache_path = EMBED_CACHE_DIR / f"embeddings_{slug}.pkl"

    if cache_path.exists():
        cached = joblib.load(cache_path)
        if cached.get("titles") == unique_titles:
            print(f"  Loaded {len(unique_titles):,} embeddings from cache.")
            return cached["embeddings"]

    from sentence_transformers import SentenceTransformer

    print(f"  Encoding {len(unique_titles):,} unique titles with '{model_name}'...")
    model = SentenceTransformer(model_name)
    embs = model.encode(
        unique_titles,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"titles": unique_titles, "embeddings": embs}, cache_path)
    print(f"  Embeddings cached to {cache_path}")
    return embs


# ---------------------------------------------------------------------------
# 3. K selection metrics
# ---------------------------------------------------------------------------

def k_selection_metrics(
    embeddings: np.ndarray,
    k_range: range,
    random_state: int = 42,
    n_init: int = 3,
    silhouette_sample: int = 5_000,
    sweep_sample: int = 30_000,
) -> pd.DataFrame:
    """
    Compute WCSS (inertia), silhouette score, and BCSS/WCSS ratio for each K.

    Uses MiniBatchKMeans on a random subsample (sweep_sample) for speed.
    The relative shape of the curves is what matters for K selection, not the absolute values.

    Returns a DataFrame indexed by K.
    """
    rng = np.random.default_rng(random_state)

    # Subsample for the sweep — curves are stable at 30K; no need for all 165K points
    n = len(embeddings)
    if n > sweep_sample:
        idx = rng.choice(n, sweep_sample, replace=False)
        emb = embeddings[idx]
        print(f"  Subsampled to {sweep_sample:,} points for K-selection sweep.")
    else:
        emb = embeddings

    total_ss = float(np.sum((emb - emb.mean(axis=0)) ** 2))

    records = []
    for k in k_range:
        km = MiniBatchKMeans(
            n_clusters=k, init="k-means++", n_init=n_init,
            random_state=random_state, batch_size=4096,
        )
        labels = km.fit_predict(emb)
        wcss = float(km.inertia_)
        bcss = total_ss - wcss
        wcss_norm = wcss / total_ss   # unexplained variance fraction (lower = better)
        r_squared = bcss / total_ss   # explained variance fraction / R² (higher = better)

        n_sil = min(len(emb), silhouette_sample)
        sil_idx = rng.choice(len(emb), n_sil, replace=False)
        sil = silhouette_score(emb[sil_idx], labels[sil_idx])

        records.append({
            "k": k,
            "wcss_norm": wcss_norm,
            "r_squared": r_squared,
            "bcss_wcss_ratio": bcss / wcss,
            "silhouette": sil,
        })
        print(f"  K={k:3d}  WCSS%={wcss_norm:.3f}  R²={r_squared:.3f}  Silhouette={sil:.4f}")

    return pd.DataFrame(records).set_index("k")


# ---------------------------------------------------------------------------
# 4. Bootstrap ARI stability
# ---------------------------------------------------------------------------

def bootstrap_ari(
    embeddings: np.ndarray,
    k: int,
    n_iter: int = 20,
    subsample_frac: float = 0.80,
    random_state: int = 42,
    n_init: int = 5,
) -> float:
    """
    Estimate clustering stability for K via bootstrap ARI.

    Each iteration: cluster all points (reference), then cluster an 80% subsample,
    assign held-out points to the nearest subsampled centroid, compute ARI vs reference.
    Returns the mean ARI over n_iter iterations.

    Interpretation: ARI >= 0.90 = very stable; 0.70–0.90 = acceptable; < 0.70 = unstable.
    """
    rng = np.random.default_rng(random_state)
    n = len(embeddings)

    ref_km = MiniBatchKMeans(
        n_clusters=k, init="k-means++", n_init=n_init,
        random_state=random_state, batch_size=4096,
    )
    ref_labels = ref_km.fit_predict(embeddings)

    aris = []
    for _ in range(n_iter):
        idx = rng.choice(n, size=int(n * subsample_frac), replace=False)
        mask = np.zeros(n, dtype=bool)
        mask[idx] = True

        sub_km = MiniBatchKMeans(
            n_clusters=k, init="k-means++", n_init=n_init,
            random_state=int(rng.integers(0, 10_000)), batch_size=4096,
        )
        sub_labels_all = np.empty(n, dtype=int)
        sub_labels_all[mask] = sub_km.fit_predict(embeddings[mask])
        sub_labels_all[~mask] = sub_km.predict(embeddings[~mask])

        aris.append(adjusted_rand_score(ref_labels, sub_labels_all))

    return float(np.mean(aris))


# ---------------------------------------------------------------------------
# 5. Clustering
# ---------------------------------------------------------------------------

def cluster_kmeans(
    embeddings: np.ndarray,
    k: int,
    random_state: int = 42,
    n_init: int = 10,
) -> np.ndarray:
    """Fit K-Means and return cluster labels (0-indexed)."""
    km = KMeans(n_clusters=k, init="k-means++", n_init=n_init, random_state=random_state)
    return km.fit_predict(embeddings)


def cluster_hdbscan(
    embeddings: np.ndarray,
    min_cluster_size: int = 10,
    min_samples: int = 5,
    max_points: int = 10_000,
    random_state: int = 42,
) -> np.ndarray:
    """
    Fit HDBSCAN on a random subsample and return labels for all points.

    HDBSCAN is O(n²) in memory/time — running it on >20K high-dimensional points
    is not practical. We subsample to max_points, cluster, then assign the remaining
    points to their nearest sampled centroid via the fitted exemplars.

    Returns labels array of length len(embeddings); non-sampled points get label -2
    if no cluster can be assigned.
    """
    import hdbscan

    n = len(embeddings)
    rng = np.random.default_rng(random_state)

    if n > max_points:
        idx = rng.choice(n, max_points, replace=False)
        sub_emb = embeddings[idx]
        print(f"  HDBSCAN: subsampled to {max_points:,} of {n:,} points.")
    else:
        idx = np.arange(n)
        sub_emb = embeddings

    clf = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        prediction_data=True,
    )
    sub_labels = clf.fit_predict(sub_emb)

    if n <= max_points:
        return sub_labels

    # Assign non-sampled points to the nearest cluster exemplar
    import hdbscan as _hdbscan
    all_labels = np.full(n, -1, dtype=int)
    all_labels[idx] = sub_labels

    unsampled_mask = np.ones(n, dtype=bool)
    unsampled_mask[idx] = False
    soft_labels, strengths = _hdbscan.approximate_predict(clf, embeddings[unsampled_mask])
    all_labels[unsampled_mask] = soft_labels

    return all_labels


# ---------------------------------------------------------------------------
# 6. Cluster summary for human review
# ---------------------------------------------------------------------------

def cluster_summary(
    df: pd.DataFrame,
    cluster_col: str = "cluster_km",
    title_col: str = "job_title_norm",
    top_n: int = 8,
) -> pd.DataFrame:
    """
    For each cluster: top-N most frequent job titles, record count, median salary,
    and % female (if gender column present). Sorted by cluster ID.
    """
    rows = []
    for cluster_id, grp in df.groupby(cluster_col):
        top_titles = grp[title_col].value_counts().head(top_n).index.tolist()
        row = {
            "cluster": cluster_id,
            "n_records": len(grp),
            "top_titles": " | ".join(top_titles),
            "median_salary": grp["salary"].median(),
        }
        if "gender" in grp.columns:
            confirmed = grp[grp["gender"] != "Uncertain"]
            row["pct_female"] = (confirmed["gender"] == "Female").mean() if len(confirmed) else float("nan")
        rows.append(row)

    return pd.DataFrame(rows).sort_values("cluster").reset_index(drop=True)
