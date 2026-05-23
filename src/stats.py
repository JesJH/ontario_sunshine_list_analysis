"""
Statistical testing for gender salary bias.

  1. raw_gap_by_cluster      — per-cluster Mann-Whitney U + gap stats (confirmed gender only)
  2. apply_bh_fdr            — Benjamini-Hochberg FDR correction at 5%
  3. regression_adjusted_gap — pooled OLS: log(salary) ~ gender + year + sector + cluster FE
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
) -> pd.DataFrame:
    """
    For each cluster: compute median salaries, raw gap, and Mann-Whitney U test.
    Only confirmed-gender records (Female / Male) are used.
    Clusters with < MIN_GROUP_SIZE in either gender are flagged sparse and skipped.

    Returns one row per cluster with columns:
      cluster, n_female, n_male, female_median, male_median,
      gap_abs (female − male), gap_pct (gap_abs / male_median),
      cles (P(female > male); < 0.5 means males tend higher),
      mw_stat, mw_pvalue, sparse
    Sorted by gap_pct ascending (largest male-favoring gaps first).
    """
    confirmed = df[df[gender_col].isin(["Female", "Male"])].copy()

    rows = []
    for cluster_id, grp in confirmed.groupby(cluster_col):
        females = grp.loc[grp[gender_col] == "Female", salary_col].dropna().values
        males   = grp.loc[grp[gender_col] == "Male",   salary_col].dropna().values

        sparse = len(females) < MIN_GROUP_SIZE or len(males) < MIN_GROUP_SIZE

        row = {
            "cluster":       cluster_id,
            "n_female":      len(females),
            "n_male":        len(males),
            "female_median": np.median(females) if len(females) else np.nan,
            "male_median":   np.median(males)   if len(males)   else np.nan,
            "sparse":        sparse,
            "mw_stat":       np.nan,
            "mw_pvalue":     np.nan,
            "cles":          np.nan,
        }

        if not sparse:
            U, p = stats.mannwhitneyu(females, males, alternative="two-sided")
            row["mw_stat"]   = float(U)
            row["mw_pvalue"] = float(p)
            # Common Language Effect Size: P(female salary > male salary)
            # > 0.5 → females tend higher; < 0.5 → males tend higher
            row["cles"] = float(U) / (len(females) * len(males))

        rows.append(row)

    result = pd.DataFrame(rows)
    result["gap_abs"] = result["female_median"] - result["male_median"]
    result["gap_pct"] = result["gap_abs"] / result["male_median"]
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
    confirmed = df[df[gender_col].isin(["Female", "Male"])].copy()
    confirmed = confirmed.dropna(subset=[salary_col, year_col, sector_col, cluster_col])
    confirmed = confirmed[confirmed[salary_col] > 0]

    confirmed["log_salary"] = np.log(confirmed[salary_col])
    confirmed["gender_fac"] = pd.Categorical(
        confirmed[gender_col], categories=["Female", "Male"]
    )

    formula = (
        f"log_salary ~ gender_fac + {year_col}"
        f" + C({sector_col}) + C({cluster_col})"
    )
    return smf.ols(formula, data=confirmed).fit()
