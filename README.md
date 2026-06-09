# Ontario Sunshine List — Gender Bias Analysis

The Government of Ontario annually publishes the Public Sector Salary Disclosure ("Sunshine List") — every public sector employee earning over $100,000. This project uses that data (2016–2025) to detect and characterize gender-based salary bias across job categories and sectors.

All tools are free and run locally — no external APIs.

---

## Key Findings

**Gender distribution and the glass ceiling**

- **The top-line split looks balanced, but hides a ceiling.** Women are ~52–53% of confirmed earners in every year from 2016–2025 — roughly representative of the workforce. But that aggregate is misleading: women make up ~62% of earners in the lowest salary decile ($100K–$104K) and fall to ~38% in the top decile ($167K+). The higher you go, the fewer women there are. This pattern is consistent across every year in the dataset.
- **The aggregate gap has not closed in ten years.** The female share of the $100K+ workforce has been essentially flat since 2016. Progress is not showing up at the headline level — if it is happening, it is happening within specific sectors and job types, not in the overall composition of who makes the list.

**Pay gap within comparable jobs**

- **A 3.5% male salary premium persists after controlling for job, sector, and year.** Running OLS regression on 2.4M records with job cluster, sector, and year as fixed effects, men earn approximately 3.5% more than women in the same role, same sector, same year (95% CI: [3.45%, 3.56%]). This is not explained by women choosing lower-paying fields — it is a within-job gap.
- **91 of 100 job clusters show a statistically significant gap; 86 favor men.** After Benjamini-Hochberg FDR correction (which guards against false discoveries when running 100 simultaneous tests), the gap is significant in the vast majority of clusters. Five clusters favor women, concentrated in roles where women have substantially higher seniority. The remaining 9 clusters are statistically indistinguishable.

**Occupational segregation**

- **Where women work and where the money is do not overlap.** Hospitals and school boards — the two most female-dominated sectors (~67–70% female) — pay at the lower end of the $100K+ range. Ontario Power Generation and municipalities — among the most male-dominated (~20–27% female) — pay substantially more. This occupational segregation accounts for a large share of the aggregate gap: women are not just underpaid within their roles, they are concentrated in lower-paying sectors.
- **Sector salary growth has been uneven.** Median salaries across most sectors grew 15–30% in nominal terms from 2016 to 2025, roughly tracking public sector wage patterns. The Judiciary sector is a notable outlier: a sharp spike in 2020 followed by a reversal in 2021, consistent with a retroactive arrears payment tied to the 2018 Order in Council (OC 1273/2018) that set judge salaries as a scheduled percentage of Superior Court rates — rather than a structural raise.

**Uncertain gender as an intersectional signal**

- **Workers with non-Western names are a growing share of the $100K+ workforce.** Gender is inferred from first names; names that are ambiguous or underrepresented in Western name databases are classified as "Uncertain" rather than forced into Female or Male. The Uncertain share has grown from 5.0% in 2016 to 6.9% in 2025 — driven by more workers with South Asian, East Asian, and French Canadian names crossing the $100K threshold. This is a workforce diversification signal.
- **Uncertain-gender workers are included in all statistical analyses as a third group.** Running three pairwise comparisons (Female vs Male, Uncertain vs Male, Female vs Uncertain) across all 100 job clusters reveals the intersectional picture: in most clusters, the ordering is Male > Uncertain > Female, suggesting that name-based disadvantage is real but smaller than the gender gap. Where Uncertain falls below Female (Male > Female > Uncertain), it points to a compounding disadvantage for workers whose names signal non-Western origin. The Uncertain category is heterogeneous — Punjabi men, East Asian women, and genuinely gender-neutral English names all land here — so it should be read as a subgroup signal, not a single demographic.

**Longitudinal trends**

- **The gap is not closing over time in most job clusters.** Tracking the same individuals year over year (linked via name and employer), most clusters show a stable or widening Female–Male salary gap across 2016–2025. The flat aggregate female share combined with a persistent within-job premium that has not narrowed suggests the structural drivers of the gap are not self-correcting under current conditions.
- **Salary growth rates and promotion proxies reveal where the gap compounds.** Even conditional on being on the Sunshine List, year-over-year salary growth rates and upward cluster transitions (a proxy for promotions) differ by gender group. Groups with lower starting salaries who also receive smaller raises and fewer promotions face compounding disadvantage over time — the gap in any given year understates the cumulative career effect.

See [methodology.md](docs/methodology.md) for full details on data processing, gender inference, clustering, statistical tests, and longitudinal analysis.

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
