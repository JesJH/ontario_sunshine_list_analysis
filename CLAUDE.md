# Ontario Sunshine List — Gender Bias Analysis

## Project Goal
Analyze Ontario Sunshine List salary data (2016–2025) to detect and characterize gender-based salary bias across job categories. Aim: rigorous enough to publish, presented accessibly.

## What Has Been Built (all sections complete)

### Source modules (`src/`)
- `src/data_loader.py` — loads and normalizes all CSVs (handles schema differences across years), cleans $ formatting, normalizes sectors/employers/job titles, extracts first given name, creates `job_title_norm`
- `src/gender_inference.py` — ensemble gender classifier: SSA lookup (50%) + Ontario baby names (20%) + char n-gram logistic regression (20%) + optional HuggingFace model (10%). Threshold 0.70/0.30; Uncertain is a first-class label.
- `src/clustering.py` — sentence embeddings (`all-MiniLM-L6-v2`), K-selection metrics (normalized WCSS, silhouette, R²), bootstrap ARI, K-Means (primary), HDBSCAN (comparison, subsampled to 10K), cluster_summary
- `src/stats.py` — raw_gap_by_cluster (generic group_a/group_b, Mann-Whitney U + CLES), apply_bh_fdr (Benjamini-Hochberg), regression_adjusted_gap (OLS log(salary) ~ gender + year + sector FE + cluster FE, all three gender categories), resolve_identities, salary_growth_rates, cluster_transitions, gap_by_cluster_year, classify_gap_trend

### Scripts
- `scripts/download_gender_data.py` — downloads SSA baby names (1950–2003) + Ontario data
- `scripts/train_gender_model.py` — trains and caches the char-gram model

### Notebook (`analysis.ipynb`) — all sections wired up
- **Section 1**: Data load, salary distribution, record count by year; sector median trend lines + total growth bar chart; YoY % change by sector; Judiciary deep-dive (audit table of individual 2019–2021 salaries, scatter plot coloured by job title, sustained vs one-time raise classifier)
- **Section 2**: Gender inference, threshold sensitivity, Uncertain breakdown, manual override cell
- **Section 2b**: Gender EDA — KDE salary distributions, median trend, salary decile chart, sector chart, employer imbalance table, takeaways commentary
- **Section 3**: NLP clustering — embeddings (cached), K-selection sweep, bootstrap ARI, K-Means + HDBSCAN, cluster human-review table
- **Section 4**: Statistical testing — three pairwise gaps (Female vs Male, Uncertain vs Male, Female vs Uncertain), BH FDR, per-cluster pairwise summary table, three-panel waterfall chart, female-favoring cluster detail, OLS regression with all three gender coefficients + derived Uncertain vs Male, interpretation commentary
- **Section 5**: Longitudinal analysis — identity resolution, salary growth rates (all three genders + Mann-Whitney), cluster transitions (upward/lateral/downward proxy for promotions), gap trajectories per cluster (closing/widening/stable), Uncertain salary benchmark

## Key Decisions Made (do not change without discussion)

| Decision | Choice | Rationale |
|---|---|---|
| Gender threshold | 0.70/0.30 | 94% coverage; chosen from sensitivity table |
| Uncertain treatment | First-class category, not imputed | Workforce diversification signal; included in all analyses as third group |
| Pairwise comparisons | Female vs Male, Uncertain vs Male, Female vs Uncertain | Covers intersectional picture |
| Regression baseline | Female | Male and Uncertain coefficients = premiums vs Female |
| K for clustering | K=100 | Silhouette peaks 90–100; K=100 ARI=0.475 > K=90 ARI=0.438 |
| Multiple comparisons | Benjamini-Hochberg FDR at α=0.05 | Exploratory research; controls false discovery rate not family-wise error |
| Sparse cluster threshold | n < 10 per group | Excluded from tests; flagged in output |
| Identity resolution | first_name_clean + last_name + employer | Conservative; no cross-employer linking |
| Cluster transition threshold | 5% median salary difference | Separates meaningful moves from noise |
| Gap trend threshold | slope ±0.001/year | Linear regression on annual gap_pct series |
| Bias definition | Raw gap (Mann-Whitney) + regression-adjusted gap | Separates occupational segregation from within-job discrimination |
| Data scope | 2016–2025 CSVs from Ontario government open data | |
| Tools | Free, local only — no rate-limited APIs | |

## Setup (run once on a new machine)
```bash
pip install -r requirements.txt
python scripts/download_gender_data.py
python scripts/train_gender_model.py
```

**Data files not in git** — copy the `data/` folder from the other machine or re-download CSVs from https://www.ontario.ca/page/public-sector-salary-disclosure and place in `data/`. The embedding cache (`data/embeddings/`) is also not in git — Section 3 will re-encode on first run (~1–2 min on CPU) and cache automatically.

## Notebook Run Order
Run top to bottom. Slow cells (first run only):
- Section 2 gender inference: ~1–2 min
- Section 3 embeddings: ~1–2 min (cached after first run)
- Section 5 identity resolution + transitions: a few minutes on 2.4M records

`gap_df` (Female vs Male, used by waterfall/female-favoring cells) is set in the Section 4 gap cell. `gap_fm`, `gap_um`, `gap_fu` are all available after that cell runs. `review` (cluster summary) must be computed in Section 3 before Section 4 cells run.

## Known Issues / Pending
- `applymap` → `map` rename: pandas 2.1+ removed `DataFrame.applymap`. Affects 3 cells in the notebook (sect1_sector_trend, sect1_sector_yoy, and one Section 5 cell). Fix by replacing `.applymap(` with `.map(` in those cells.

## Key Caveats
- No seniority/experience field — gap conflates discrimination with tenure differences
- Left-censoring: $100K threshold means we only observe people after they cross it; "years on list" understates true seniority
- Gender is inferred, not observed — coverage and method flagged per record
- Uncertain is a heterogeneous residual category (South Asian, East Asian, French Canadian, genuinely ambiguous English names) — treat as a subgroup signal, not a single demographic
- With N=2.4M, all p-values are near zero — report CLES and gap_pct as effect sizes; a gap < 3% or CLES between 0.45–0.55 is statistically significant but practically negligible

## User Preferences
- Explain statistical concepts in plain language before implementing
- Free, local tools only — no rate-limited APIs
- Heavy logic in `src/` modules; notebook cells are thin wrappers
- Include Uncertain gender in all analyses alongside Female and Male
- Commentary cells after every chart section
