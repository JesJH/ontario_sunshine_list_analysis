# Ontario Sunshine List — Gender Bias Analysis

The Government of Ontario publishes the Public Sector Salary Disclosure ("Sunshine List") annually — every public sector employee earning over $100,000. This project uses that data (2016–2025) to measure gender-based salary bias across job categories and sectors, separating the effect of *which jobs women hold* from the effect of *what women are paid within those jobs*.

All tools are free and run locally — no external APIs.

---

## What This Project Does

The core challenge is that raw salary comparisons across genders are confounded by occupational segregation — women and men don't hold the same jobs in equal proportions. This project builds a full pipeline to untangle that:

1. **Gender inference** — Gender is not in the raw data. An ensemble model infers it from first names using US and Ontario birth registration data, combined with a character n-gram classifier trained to generalize to non-Western names. Names that can't be confidently assigned are kept as a third "Uncertain" category rather than dropped or forced into Female/Male.

2. **NLP job title clustering** — The Sunshine List contains ~147,000 distinct job title variants. Sentence embeddings group semantically similar titles into 100 comparable role clusters (e.g. "registered nurse", "charge nurse", and "staff nurse" land in the same cluster), enabling apples-to-apples salary comparisons.

3. **Statistical testing** — Mann-Whitney U tests measure salary gaps within each cluster, with Benjamini-Hochberg correction for multiple comparisons. OLS regression further controls for year and sector to isolate the within-job premium. All tests run across three gender pairings: Female vs Male, Uncertain vs Male, and Female vs Uncertain.

4. **Longitudinal analysis** — Individual workers are tracked across years to measure salary growth rates, cluster transitions (a promotion proxy), and whether the gap is closing or widening over time.

See [docs/methodology.md](docs/methodology.md) for full detail on each step, statistical justifications, and assumptions.

---

## Key Findings

- **Glass ceiling:** Women are ~52–53% of earners on the list every year, but make up ~62% in the lowest salary decile ($100K–$104K) and fall to ~38% in the top decile ($167K+). The higher the salary band, the fewer women there are.
- **3.5% within-job male premium:** After controlling for role, sector, and year via OLS regression, men earn ~3.5% more than women in the same position (95% CI: [3.45%, 3.56%]). This is not explained by occupational sorting.
- **91 of 100 job clusters show a significant gap; 86 favor men** — after FDR correction.
- **Occupational segregation compounds the gap:** Female-dominated sectors (hospitals, school boards, ~67–70% female) pay at the lower end of the $100K+ range. Male-dominated sectors (Ontario Power Generation, municipalities, ~20–27% female) pay substantially more.
- **Uncertain-gender workers are a growing share:** The Uncertain share grew from 5.0% (2016) to 6.9% (2025), reflecting more workers with South Asian, East Asian, and French Canadian names crossing the $100K threshold. In most job clusters the salary ordering is Male > Uncertain > Female — name-based disadvantage is real, but smaller than the gender gap.
- **The gap is not closing:** Tracking the same individuals year over year, most clusters show a stable or widening Female–Male gap. Salary growth rates and upward cluster transitions also differ by gender, meaning the gap compounds over a career.

---

## Setup

### 1. Get the data

Download the salary CSVs from the [Ontario open data portal](https://www.ontario.ca/page/public-sector-salary-disclosure) and place them in `data/`. The loader handles all schema differences across years — no renaming needed.

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download gender reference data

```bash
python scripts/download_gender_data.py
```

Downloads SSA baby name files (1950–2003) and Ontario provincial name data to `data/gender/`. Run once.

### 4. Train the gender model

```bash
python scripts/train_gender_model.py
```

Trains the character n-gram classifier and caches it to `data/gender/char_model.pkl`. Takes about a minute. Run once.

### 5. Run the notebook

```bash
jupyter notebook analysis.ipynb
```

Run cells top to bottom. Two cells are slow on first run only:

| Cell | Why | First-run time |
|---|---|---|
| Section 2 — gender inference | Classifies every unique first name | ~1–2 min |
| Section 3 — embeddings | Encodes all unique job titles | ~1–2 min on CPU |

Embeddings are cached to `data/embeddings/` — subsequent runs are instant.

---

## Project Structure

```
analysis.ipynb          # Main notebook — run this
src/
  data_loader.py        # Load and normalize all CSVs
  gender_inference.py   # Ensemble gender classifier
  clustering.py         # Embeddings, K-selection, K-Means, HDBSCAN
  stats.py              # Statistical tests and longitudinal analysis
scripts/
  download_gender_data.py
  train_gender_model.py
docs/
  methodology.md        # Full methodology and statistical justifications
  ai_usage.md           # How Claude Code was used in this project
data/                   # Not in git — add CSVs here
```

---

## Caveats

- **No seniority data** — gaps conflate pay discrimination with tenure differences; a male-dominated cluster may pay more simply because men have been in the role longer
- **Left-censoring** — the $100K threshold means early-career salaries are invisible; "years on list" understates true seniority and does so unequally across groups
- **Gender is inferred, not observed** — 94% of records are assigned Female or Male at the chosen threshold; Uncertain records are kept as a third category throughout, not dropped
- **Ontario public sector only** — unionization rates, pay equity legislation, and public scrutiny all differ from the private sector; results are not generalizable beyond this population
