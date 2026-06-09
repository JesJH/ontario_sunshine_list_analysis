# Analysis Notes — Working Document

Quick-reference companion to METHODOLOGY.md. Update this as the analysis evolves.

---

## Coverage — what's been built

| Area | Status | Where |
|---|---|---|
| Data load + normalization | Done | `src/data_loader.py`, Section 1 |
| Salary distribution EDA | Done | Section 1 |
| Sector salary trend charts | Done | Section 1 |
| Judiciary sector deep-dive | Done | Section 1 (cells after sector trends) |
| Gender inference (ensemble) | Done | `src/gender_inference.py`, Section 2 |
| Gender EDA (KDE, decile, sector, employer imbalance) | Done | Section 2b |
| NLP job title clustering (K=100) | Done | `src/clustering.py`, Section 3 |
| Statistical testing — 3 pairwise gaps + BH FDR | Done | `src/stats.py`, Section 4 |
| OLS regression (adjusted gap) | Done | Section 4 |
| Longitudinal: salary growth rates | Done | Section 5 |
| Longitudinal: cluster transitions (promotion proxy) | Done | Section 5 |
| Longitudinal: gap trajectories per cluster | Done | Section 5 |

---

## Key decisions (short form — full rationale in METHODOLOGY.md)

| Decision | Choice | Why |
|---|---|---|
| Gender threshold | 0.70 / 0.30 | 94% coverage; loosening adds noise not signal |
| Uncertain treatment | First-class category, not imputed | Workforce diversification signal |
| Pairwise comparisons | F vs M, U vs M, F vs U | Covers intersectional picture |
| OLS baseline | Female | Male + Uncertain coefficients read as premiums |
| K for clustering | 100 | Best ARI + silhouette plateau in 90–100 range |
| Multiple comparisons | BH FDR α=0.05 | Exploratory; controls false discovery rate |
| Sparse cluster cutoff | n < 10 per group | Excluded from tests |
| Identity resolution key | first_name + last_name + employer | Conservative; no cross-employer linking |
| Cluster transition threshold | 5% median salary difference | Separates signal from noise |
| Gap trend threshold | ±0.001 slope/year | Linear regression on annual gap_pct series |

---

## Assumptions (short form — full discussion in METHODOLOGY.md)

- SSA birth years 1950–2003 are representative of $100K+ Ontario public sector workers in 2016–2025
- Char-gram model generalizes imperfectly to non-Western names — Uncertain is the honest fallback
- Growing Uncertain share (5.0% → 6.9%) reflects diversification, not data degradation
- Sentence similarity is a reasonable proxy for "comparable job"
- OLS omits employer fixed effects — sector + cluster capture most but not all employer-level variation
- Left-censoring ($100K threshold) biases "years on list" as a seniority proxy, more so for lower-starting groups

---

## Parking lot — noticed but not investigated

These are observations or questions that came up during the analysis and were set aside. Not commitments — just things worth returning to.

### Benefits gap
The raw data includes a `Benefits` column (taxable benefits paid). All analysis uses salary only. It's plausible that total compensation gaps differ from salary gaps — benefits packages may vary by role type in ways that correlate with gender. Quick check: compute median benefits by gender and test whether the benefits gap runs the same direction as the salary gap.

### Employer-level breakdown
Sector-level analysis is done. Employer level is not. Some employers (e.g. a single hospital or university) may be outliers driving sector-wide patterns. An employer Gini or "worst offender" table would let you say "the sector gap is largely explained by three employers."

### D9–D10 salary decile drop
Section 2b flags this: women drop from ~50% of D8 earners to ~38% of D10 earners. The top-of-range drop is steeper than any other decile transition. This was described as "worth investigating in the cluster-level analysis" but the cluster analysis doesn't specifically revisit it. Which clusters are driving D10 male dominance — executives, legal, medicine?

### Uncertain name-origin decomposition
Uncertain is known to be heterogeneous: South Asian names, East Asian names, French Canadian ambiguous names, genuinely gender-neutral English names. These sub-groups likely have different salary patterns. A rough decomposition using phonetic heuristics (names ending in -singh, -patel, -chen, etc.) could split Uncertain into sub-clusters for a more targeted intersectional signal.

### COVID-era salary effects (2020–2022) across all sectors
The judiciary 2020 spike was investigated in detail. The broader COVID-era effect on salaries and gender composition across other sectors was not. Healthcare in particular — were women (overrepresented in frontline nursing) more or less affected by pandemic-era pay changes vs. men?

### New entrant vs. long-tenure gap
The longitudinal analysis tracks people over time but doesn't distinguish new entrants from long-tenure earners. People appearing for the first time post-2020 may show a different gender gap than those who've been on the list since 2016 — relevant to whether hiring practices or promotion pipelines are improving.

### Exit patterns (who leaves the list)
Currently the analysis tracks who's on the list but not who drops off. If women are more likely to exit (cross back below $100K, change employers, retire earlier) that could explain part of the stable Female share — a treadmill effect where women enter at the same rate they exit, rather than genuine stability.

### Sector × cluster interaction
The sector-level box plots showed no single direction of bias across sectors. But the cluster-level analysis doesn't disaggregate by sector. A nurse cluster might show bias running opposite to an administrator cluster within the same Healthcare sector, cancelling out. Cross-tabulating cluster gap by sector would reveal this.

### Union vs. non-union salary compression
Heavily unionized roles (teachers, nurses, most municipal workers) have compressed salary grids — the gap is structurally smaller there. Non-unionized roles (executives, legal, some university admin) have more discretionary pay. The analysis doesn't flag whether a cluster is likely unionized, which affects how to interpret a small gap (compressed grid) vs. large gap (discretionary pay).

### Geographic variation
Employer names imply geography (Toronto, Ottawa, rural health units). Urban/rural or GTA vs. rest-of-Ontario comparisons were never attempted. Pay scales differ enough that geography could explain some employer-level variation.
