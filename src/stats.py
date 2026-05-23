"""
Statistical testing for gender salary bias.

  1. raw_gap_by_cluster      — per-cluster Mann-Whitney U + gap stats (confirmed gender only)
  2. apply_bh_fdr            — Benjamini-Hochberg FDR correction at 5%
  3. regression_adjusted_gap — pooled OLS: log(salary) ~ gender + year + sector + cluster FE
  4. resolve_identities      — link records across years via name + employer → person_id
  5. salary_growth_rates     — year-over-year salary growth per person (requires person_id)
  6. cluster_transitions     — classify upward/lateral/downward moves between clusters
  7. gap_by_cluster_year     — median gap % per cluster × year (for trend analysis)
  8. classify_gap_trend      — label a cluster's gap time series: closing/widening/stable
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as smf

MIN_GROUP_SIZE = 10  # minimum per-gender records to include a cluster in tests


# ---------------------------------------------------------------------------
# 1. Raw gap + Mann-Whitney U per cluster
# ---------------------------------------------------------------------------

def raw_gap_by_cluster(
    df: pd.DataFrame,
    cluster_col: str = "cluster_km",
    gender_col: str = "gender",
    salary_col: str = "salary",
    group_a: str = "Female",
    group_b: str = "Male",
) -> pd.DataFrame:
    """
    For each cluster: compute median salaries, raw gap, and Mann-Whitney U test
    comparing group_a vs group_b.

    gap_pct = (group_a_median − group_b_median) / group_b_median
      positive → group_a earns more; negative → group_b earns more
    cles = P(group_a salary > group_b salary)
      > 0.5 → group_a tends higher; < 0.5 → group_b tends higher

    Returns one row per cluster with columns:
      cluster, n_a, n_b, a_median, b_median,
      gap_abs, gap_pct, cles, mw_stat, mw_pvalue, sparse
    Sorted by gap_pct ascending (largest group_b-favoring gaps first).
    """
    subset = df[df[gender_col].isin([group_a, group_b])].copy()

    rows = []
    for cluster_id, grp in subset.groupby(cluster_col):
        a_vals = grp.loc[grp[gender_col] == group_a, salary_col].dropna().values
        b_vals = grp.loc[grp[gender_col] == group_b, salary_col].dropna().values

        sparse = len(a_vals) < MIN_GROUP_SIZE or len(b_vals) < MIN_GROUP_SIZE

        row = {
            "cluster":  cluster_id,
            "n_a":      len(a_vals),
            "n_b":      len(b_vals),
            "a_median": np.median(a_vals) if len(a_vals) else np.nan,
            "b_median": np.median(b_vals) if len(b_vals) else np.nan,
            "sparse":   sparse,
            "mw_stat":  np.nan,
            "mw_pvalue": np.nan,
            "cles":     np.nan,
        }

        if not sparse:
            U, p = stats.mannwhitneyu(a_vals, b_vals, alternative="two-sided")
            row["mw_stat"]   = float(U)
            row["mw_pvalue"] = float(p)
            row["cles"]      = float(U) / (len(a_vals) * len(b_vals))

        rows.append(row)

    result = pd.DataFrame(rows)
    result["gap_abs"] = result["a_median"] - result["b_median"]
    result["gap_pct"] = result["gap_abs"] / result["b_median"]
    return result.sort_values("gap_pct").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Benjamini-Hochberg FDR correction
# ---------------------------------------------------------------------------

def apply_bh_fdr(
    gap_df: pd.DataFrame,
    alpha: float = 0.05,
    pvalue_col: str = "mw_pvalue",
) -> pd.DataFrame:
    """
    Apply Benjamini-Hochberg FDR correction across all non-sparse clusters.
    Adds columns bh_pvalue (adjusted) and bh_significant (bool).
    Sparse / untested clusters keep NaN.
    """
    out = gap_df.copy()
    out["bh_pvalue"]      = np.nan
    out["bh_significant"] = False

    testable = out.index[~out["sparse"] & out[pvalue_col].notna()]
    if len(testable) == 0:
        return out

    pvals = out.loc[testable, pvalue_col].values
    reject, pvals_adj, _, _ = multipletests(pvals, alpha=alpha, method="fdr_bh")

    out.loc[testable, "bh_pvalue"]      = pvals_adj
    out.loc[testable, "bh_significant"] = reject

    return out


# ---------------------------------------------------------------------------
# 3. Regression-adjusted gap (pooled OLS)
# ---------------------------------------------------------------------------

def regression_adjusted_gap(
    df: pd.DataFrame,
    gender_col: str = "gender",
    salary_col: str = "salary",
    year_col:   str = "year",
    sector_col: str = "sector",
    cluster_col: str = "cluster_km",
):
    """
    Pooled OLS: log(salary) ~ gender + year + C(sector) + C(cluster)

    Only confirmed-gender records used. Returns a fitted statsmodels RegressionResults.

    The coefficient on 'gender_fac[T.Male]' is the log-scale male premium
    controlling for year, sector, and job cluster (multiply by ~100 for approx %).

    Employer fixed effects are omitted — thousands of levels make OLS impractical;
    sector + cluster together capture most occupational and organizational variation.
    """
    confirmed = df[df[gender_col].isin(["Female", "Male", "Uncertain"])].copy()
    confirmed = confirmed.dropna(subset=[salary_col, year_col, sector_col, cluster_col])
    confirmed = confirmed[confirmed[salary_col] > 0]

    confirmed["log_salary"] = np.log(confirmed[salary_col])
    # Female is the baseline; coefficients for Male and Uncertain are premiums vs Female
    confirmed["gender_fac"] = pd.Categorical(
        confirmed[gender_col], categories=["Female", "Male", "Uncertain"]
    )

    formula = (
        f"log_salary ~ gender_fac + {year_col}"
        f" + C({sector_col}) + C({cluster_col})"
    )
    return smf.ols(formula, data=confirmed).fit()


# ---------------------------------------------------------------------------
# 4. Identity resolution
# ---------------------------------------------------------------------------

def resolve_identities(df: pd.DataFrame) -> pd.DataFrame:
    """
    Link records across years using first_name_clean + last_name + employer as a key.
    Assigns an integer person_id to each unique (name, employer) combination.

    People who change employers are NOT linked across the employer boundary — this
    is conservative but avoids false merges between different people with the same name.

    Returns df with a new 'person_id' column (integer).
    """
    key = (
        df["first_name_clean"].fillna("").str.strip().str.lower()
        + "|"
        + df["last_name"].fillna("").str.strip().str.lower()
        + "|"
        + df["employer"].fillna("").str.strip().str.lower()
    )
    unique_keys = key.unique()
    key_to_id = {k: i for i, k in enumerate(unique_keys)}

    out = df.copy()
    out["person_id"] = key.map(key_to_id)
    return out


# ---------------------------------------------------------------------------
# 5. Salary growth rates (requires person_id from resolve_identities)
# ---------------------------------------------------------------------------

def salary_growth_rates(
    df: pd.DataFrame,
    cluster_col: str = "cluster_km",
) -> pd.DataFrame:
    """
    Compute year-over-year salary growth rate for each person with 2+ consecutive years.

    Returns one row per (person, year) transition with columns:
      person_id, year, gender, cluster_km, salary, prev_salary, growth_rate

    Only consecutive year pairs (gap == 1) are kept — non-consecutive gaps are skipped
    to avoid confounding salary growth with years away from the list.
    """
    if "person_id" not in df.columns:
        raise ValueError("Run resolve_identities(df) first to create person_id.")

    cols = ["person_id", "year", "gender", cluster_col, "salary"]
    d = (
        df[cols]
        .dropna(subset=["salary", "year", cluster_col])
        .sort_values(["person_id", "year"])
        .copy()
    )

    d["prev_salary"] = d.groupby("person_id")["salary"].shift(1)
    d["prev_year"]   = d.groupby("person_id")["year"].shift(1)

    # Drop non-consecutive pairs (person left list and returned)
    d = d[(d["year"] - d["prev_year"]) == 1].copy()
    d["growth_rate"] = (d["salary"] - d["prev_salary"]) / d["prev_salary"]

    return d.dropna(subset=["growth_rate"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 6. Cluster transitions (upward / lateral / downward moves)
# ---------------------------------------------------------------------------

def cluster_transitions(
    df: pd.DataFrame,
    cluster_col: str = "cluster_km",
    threshold: float = 0.05,
) -> pd.DataFrame:
    """
    For each consecutive year pair per person, record whether they moved to a
    higher-paying cluster (upward), lower-paying (downward), or stayed equivalent
    (lateral — same cluster or <5% median salary difference by default).

    Cluster direction is defined by comparing the median salary of the source and
    destination clusters across all records (not just that person's salary).

    Returns one row per transition with columns:
      person_id, year_from, year_to, gender,
      cluster_from, cluster_to, direction

    threshold: fraction by which destination cluster median must exceed source
               for a move to be classified as upward (default 5%).
    """
    if "person_id" not in df.columns:
        raise ValueError("Run resolve_identities(df) first to create person_id.")

    cluster_medians = df.groupby(cluster_col)["salary"].median().to_dict()

    cols = ["person_id", "year", "gender", cluster_col]
    d = (
        df[cols]
        .dropna(subset=["year", cluster_col])
        .sort_values(["person_id", "year"])
        .copy()
    )

    d["cluster_next"] = d.groupby("person_id")[cluster_col].shift(-1)
    d["year_next"]    = d.groupby("person_id")["year"].shift(-1)

    # Only consecutive year pairs
    d = d[(d["year_next"] - d["year"]) == 1].dropna(subset=["cluster_next"]).copy()
    d["cluster_next"] = d["cluster_next"].astype(int)

    med_from = d[cluster_col].map(cluster_medians)
    med_to   = d["cluster_next"].map(cluster_medians)

    conditions = [
        d[cluster_col] == d["cluster_next"],                 # same cluster
        med_to > med_from * (1 + threshold),                 # meaningfully higher
        med_to < med_from * (1 - threshold),                 # meaningfully lower
    ]
    choices = ["lateral", "upward", "downward"]
    d["direction"] = np.select(conditions, choices, default="lateral")

    return (
        d.rename(columns={"year": "year_from", "year_next": "year_to",
                          cluster_col: "cluster_from"})
        [["person_id", "year_from", "year_to", "gender",
          "cluster_from", "cluster_next", "direction"]]
        .rename(columns={"cluster_next": "cluster_to"})
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# 7. Per-cluster × year gap (for longitudinal trend plots)
# ---------------------------------------------------------------------------

def gap_by_cluster_year(
    df: pd.DataFrame,
    cluster_col: str = "cluster_km",
    min_per_gender: int = 10,
) -> pd.DataFrame:
    """
    Compute median salary gap % (Female − Male) / Male for each cluster × year.

    Returns a long-form DataFrame with columns:
      cluster, year, gap_pct, n_female, n_male

    Cluster-year cells with fewer than min_per_gender in either group are dropped.
    """
    confirmed = df[df["gender"].isin(["Female", "Male"])].copy()

    rows = []
    for (cluster, year), grp in confirmed.groupby([cluster_col, "year"]):
        females = grp[grp["gender"] == "Female"]["salary"].dropna().values
        males   = grp[grp["gender"] == "Male"]["salary"].dropna().values
        if len(females) < min_per_gender or len(males) < min_per_gender:
            continue
        gap = (np.median(females) - np.median(males)) / np.median(males)
        rows.append({
            "cluster":  cluster,
            "year":     int(year),
            "gap_pct":  gap,
            "n_female": len(females),
            "n_male":   len(males),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 8. Classify a cluster's gap trend over time
# ---------------------------------------------------------------------------

def classify_gap_trend(gap_series: pd.Series, min_years: int = 5) -> str:
    """
    Classify the direction of a cluster's gender gap trend over time.

    gap_series: a Series of gap_pct values indexed by year (negative = male-favoring).
    Returns one of: 'closing', 'widening', 'stable', 'insufficient_data'.

    gap_pct convention: negative = male-favoring, positive = female-favoring.
    A rising slope means gap_pct is increasing → gap is closing (becoming less male-favoring).
    A falling slope means gap_pct is decreasing → gap is widening (more male-favoring).

    Uses linear regression slope; threshold of ±0.001 per year separates stable from trending.
    """
    s = gap_series.dropna()
    if len(s) < min_years:
        return "insufficient_data"

    x = np.arange(len(s))
    slope, _, _, _, _ = stats.linregress(x, s.values)

    if slope > 0.001:
        return "closing"
    if slope < -0.001:
        return "widening"
    return "stable"
