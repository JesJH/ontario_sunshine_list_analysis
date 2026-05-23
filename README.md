# Ontario Sunshine List — Gender Bias Analysis

The Government of Ontario annually publishes the Public Sector Salary Disclosure ("Sunshine List") — every public sector employee earning over $100,000. This project uses that data (2016–2025) to detect and characterize gender-based salary bias across job categories and sectors.

All tools are free and run locally — no external APIs.

---

## Key Findings

- **Glass ceiling effect:** Women make up ~62% of confirmed earners in the lowest salary decile ($100K–$104K) and drop to ~38% in the top decile ($167K+).
- **Stable aggregate split:** Women are ~52–53% of confirmed earners in every year from 2016–2025. The top-line gap has not closed or widened — change is happening *within* sectors and job types.
- **91 of 100 job clusters** show a statistically significant gender pay gap after FDR correction. 86 favor men, 5 favor women.
- **+3.5% adjusted male premium** — after controlling for year, sector, and job cluster, men earn 3.5% more on average (95% CI: [3.45%, 3.56%]).
- **Occupational segregation is real:** hospitals and school boards (~67–70% female) pay at the lower end of the $100K+ range; Ontario Power Generation and municipalities (~20–27% female) pay substantially more.
- **Uncertain-gender share growing:** 5.0% (2016) → 6.9% (2025), reflecting more workers with South Asian, East Asian, and French Canadian names crossing the threshold — a workforce diversification signal, not a data quality issue.

See [METHODOLOGY.md](METHODOLOGY.md) for full details on data processing, gender inference, clustering, statistical tests, and longitudinal analysis.

---

## Setup

### 1. Get the data

Download the salary CSVs from the [Ontario open data portal](https://www.ontario.ca/page/public-sector-salary-disclosure) and place them in `data/`. The loader handles all schema differences across years automatically — no renaming needed.

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download gender reference data

```bash
python scripts/download_gender_data.py
```

Downloads SSA baby name files (birth years 1950–2003) and Ontario provincial name data to `data/gender/`. Run once.

### 4. Train the gender model

```bash
python scripts/train_gender_model.py
```

Trains a character n-gram logistic regression on SSA names and caches it to `data/gender/char_model.pkl`. Takes about a minute. Run once.

### 5. Run the notebook

```bash
jupyter notebook analysis.ipynb
```

Run cells top to bottom. Two cells are slow on first run only:

| Cell | Why | Time |
|---|---|---|
| Section 2 — gender inference | Classifies every unique first name | ~1–2 min |
| Section 3 — embeddings | Encodes all unique job titles | ~1–2 min on CPU |

Embeddings are cached to `data/embeddings/` after the first run — subsequent runs are instant.

---

## Project Structure

```
analysis.ipynb          # Main notebook — run this
src/
  data_loader.py        # Load + normalize all CSVs
  gender_inference.py   # Ensemble gender classifier
  clustering.py         # Embeddings, K-selection, K-Means, HDBSCAN
  stats.py              # Statistical tests + longitudinal analysis
scripts/
  download_gender_data.py
  train_gender_model.py
data/                   # Not in git — add CSVs here
METHODOLOGY.md          # Full methodology, statistical justifications, caveats
```

---

## Caveats

- **No seniority data** — gaps conflate discrimination with tenure differences
- **Left-censoring** — the $100K threshold means we only observe people after they cross it; "years on list" understates true seniority
- **Gender is inferred, not observed** — 94% coverage at the chosen threshold; Uncertain records are kept as a third category throughout
- **Ontario public sector only** — not representative of the broader Ontario workforce
