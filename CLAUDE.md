# Ontario Sunshine List — Gender Bias Analysis

## Project Goal
Analyze Ontario Sunshine List salary data (2017–2025) to detect and characterize gender-based salary bias across job categories. Aim: rigorous enough to publish, presented accessibly.

## What Has Been Built
- `src/data_loader.py` — loads and normalizes all 9 CSVs (handles schema differences across years), cleans $ formatting, extracts first given name
- `src/gender_inference.py` — ensemble gender classifier: SSA lookup (60%) + char n-gram logistic regression trained on SSA data (30%) + optional HuggingFace model (10%)
- `scripts/download_gender_data.py` — downloads SSA baby names (birth years 1950–2003) + NLTK corpus
- `scripts/train_gender_model.py` — trains and caches the char-gram model, prints cross-val accuracy
- `analysis.ipynb` — runner notebook; Sections 1 (data load) and 2 (gender inference) are wired up; Sections 3–5 are stubs

## What To Build Next
- **Section 3:** NLP job title clustering — sentence embeddings + K-Means + HDBSCAN + bootstrap stability (ARI)
- **Section 4:** Statistical testing — Mann-Whitney U per cluster + Benjamini-Hochberg FDR
- **Section 5:** Longitudinal trend analysis — salary growth rates, cluster transitions, trajectory shapes

## Setup (run once on a new machine)
```bash
pip install -r requirements.txt
python scripts/download_gender_data.py
python scripts/train_gender_model.py
```
Data CSVs are not in git — copy the `data/` folder manually or re-download from Ontario open data.

## Methodology Decisions (fixed — do not change without discussion)

| Decision | Choice |
|---|---|
| Bias definition | Both raw gap AND regression-adjusted gap (year, sector, employer controls) |
| Sparse clusters | Show separately; if widespread, split overlap vs. segregated analysis |
| Time trends | Flag all three: closing, widening, sudden appearance |
| Multiple comparisons | Benjamini-Hochberg FDR at 5% |
| Gender inference | SSA lookup + char model + optional HF model. Uncertain = excluded from gender tests, kept in aggregate |
| Gender data | SSA birth years 1950–2003. Free/local only, no APIs |
| NLP clustering | K-Means (primary) + HDBSCAN (comparison). K via elbow + silhouette + WCSS/BCSS + bootstrap stability (ARI). ~100 clusters target. Human review layer. |
| Seniority proxy | Years on list + salary growth rate + cluster transitions |
| Identity resolution | Normalized name + employer as primary key, fuzzy fallback, flag ambiguous |
| Data scope | 2017–2025 CSVs from Ontario government open data |

## Key Caveats to Document
- No seniority/experience field — only raw gaps and partially-controlled gaps are measurable
- Left-censoring: list only shows people once they cross $100K, so "years on list" underestimates true seniority
- Gender is inferred from first names — flag inference method per record, report coverage %

## User Preferences
- Walk through statistical concepts in plain language with a recommendation before asking for a decision
- Free, local tools only — no rate-limited APIs
- Runner notebook (`analysis.ipynb`) calls `src/` modules; no heavy logic in the notebook itself
