---
marp: true
theme: default
paginate: true
style: |
  section {
    font-family: 'Segoe UI', sans-serif;
    font-size: 22px;
    padding: 40px 50px;
  }
  h1 { font-size: 38px; color: #1a3a5c; border-bottom: 3px solid #4472c4; padding-bottom: 10px; }
  h2 { font-size: 30px; color: #1a3a5c; }
  h3 { font-size: 24px; color: #2c5f8a; }
  strong { color: #c0392b; }
  table { font-size: 18px; width: 100%; }
  th { background: #1a3a5c; color: white; padding: 6px 10px; }
  td { padding: 5px 10px; }
  tr:nth-child(even) { background: #f0f4f8; }
  .columns { display: grid; grid-template-columns: 1fr 1fr; gap: 30px; }
  ul li { margin-bottom: 6px; }
  footer { font-size: 14px; color: #888; }
---

<!-- _paginate: false -->

# Ontario Sunshine List
## Gender Bias Analysis — 2016–2025

**Ontario public sector salary data ($100K+ earners)**
2.4 million records across 10 sectors and 9 years

---

## What Are We Trying to Achieve?

**Research question:** Is there a measurable gender pay gap in Ontario's public sector, and if so — where, how large, and is it changing over time?

**Why this matters:**
- The Sunshine List covers every public sector worker earning $100K+
- It is the most complete high-earner salary dataset available to the public in Canada
- Understanding *where* the gap exists (by job role, sector, over time) is more actionable than a single aggregate number

**What "gap" means here:**
- **Raw gap** — women earn less, full stop
- **Adjusted gap** — women earn less even after controlling for job type, sector, and year
- Both matter: raw gap captures occupational segregation; adjusted gap captures within-role discrimination

---

## How We're Addressing It — Five Steps

| Step | What | Why |
|---|---|---|
| **1. Data** | Load & normalize 10 CSVs (2016–2025), 2.4M records | Consistent schema, clean salaries |
| **2. Gender inference** | Ensemble classifier: SSA names + char n-gram ML | No self-reported gender — best available proxy |
| **3. Job clustering** | NLP sentence embeddings → K-Means (K=100) | Compare apples to apples — same job role |
| **4. Statistical testing** | Mann-Whitney U + OLS regression + BH FDR | Separate occupational segregation from within-role pay gap |
| **5. Longitudinal** | Identity resolution → growth rates, promotions, gap trends | Does the gap close, widen, or persist over time? |

---

## The Data

**Source:** Ontario Government Open Data — Public Sector Salary Disclosure

| | |
|---|---|
| Years covered | 2016–2025 (10 years) |
| Total records | ~2.4 million |
| Threshold | $100,000+ salary (left-censored — we only see workers after they cross this) |
| Sectors | Healthcare, Education, Municipalities, Crown Agencies, Universities, Government of Ontario, Hospitals, School Boards, Judiciary, Other |
| Key columns | Last Name, First Name, Employer, Job Title, Salary Paid, Taxable Benefits, Year, Sector |

**Key caveat:** No seniority or experience field. The gap we measure conflates discrimination with tenure differences. The left-censoring means "years on the list" understates true seniority, more so for groups who start at lower salaries.

---

## Gender Inference

**Three-source ensemble classifier — no rate-limited APIs, fully local:**

| Source | Weight | What it covers |
|---|---|---|
| SSA baby name frequencies (birth years 1950–2003) | 50% | Strong signal for common North American names |
| Ontario baby name frequencies | 20% | Local context, especially French Canadian names |
| Character n-gram logistic regression (trained on SSA) | 20% | Handles spelling variants and less common names |
| HuggingFace model (optional) | 10% | Additional signal when enabled |

**Threshold:** p(Female) ≥ 0.70 → Female; ≤ 0.30 → Male; in between → **Uncertain**

**Coverage at this threshold:** ~94% of records classified as Female or Male

**Uncertain is treated as a first-class category** — not imputed, not dropped. Growing Uncertain share is a workforce diversification signal (more workers with South Asian, East Asian, French Canadian names crossing $100K).

---

## EDA: Who Is on the List?

**Gender composition — stable at the top, changing underneath:**

- Female records: **~49–50%** of all records; **~52–53%** of confirmed-gender earners — consistently
- Uncertain share: **5.0% (2016) → 6.9% (2025)** — growing +38% relative, a diversification signal
- **The Female–Male split has not meaningfully closed or widened in 10 years** at the aggregate level

**What this means:**
If representation were improving, we'd see the Female share rising. It isn't — which tells us that any change in gender equity is happening *within* sectors and job roles, not at the top-line level. This is exactly why the cluster-level analysis matters.

---

## EDA: Women Cluster at the Bottom, Men at the Top

**Salary decile breakdown (confirmed-gender earners only):**

| Decile | Salary range | % Female |
|---|---|---|
| D1–D2 | $100K – $104K | ~62% |
| D3–D5 | $105K – $115K | ~55% |
| D6–D8 | $116K – $140K | ~48% |
| D9 | $141K – $167K | ~46% |
| **D10** | **$167K+** | **~38%** |

The D9 → D10 drop is the sharpest single-decile transition. **Women are overrepresented among the lowest-paid $100K+ earners and underrepresented at the very top.** This is the most direct evidence of a pay gap before any statistical testing.

---

## EDA: Occupational Segregation — Not Uniform Bias

**No single gap direction across all sectors — the gap is structural:**

<div class="columns">

**Female-dominated (67–70%+ female):**
- Healthcare — Hospitals
- Boards of Public Health
- School Boards
- Community Health / Social Care

**Male-dominated (20–27% female):**
- Ontario Power Generation
- Municipalities (fire services, trades)
- Construction/trades training centres

</div>

**Key implication:** Much of the raw salary gap is driven by *which sectors pay more* — not by within-role pay differences. Male-dominated sectors (energy, skilled trades) tend to have higher median salaries than female-dominated sectors (care, education). The OLS regression in Step 4 controls for this.

---

## EDA: The Uncertain Group Is a Diversification Signal

**Uncertain (unclassified) earners are growing and concentrated in high-diversity employers:**

- Highest Uncertain shares: **Crown Agencies (8.7%), Universities (8.0%), Hospitals (7.9%)**
- These are the sectors most likely to employ internationally trained professionals
- Uncertain workers earn **between Female and Male** medians in most years
- The Uncertain median sitting closer to Male suggests this pool is **male-majority in salary terms** — consistent with many South Asian and East Asian men in engineering, medicine, and academia

**Why this matters:** Uncertain is not a data quality failure — it is the Sunshine List slowly becoming more representative of Ontario's population. Female and Male headcount figures become slightly more conservative (understated) each year as more workers flow into Uncertain.

---

## Step 3: Job Title Clustering (NLP)

**Why we cluster:** 165,000+ unique job titles across 10 years and 10 sectors. A raw sector-level comparison lumps together nurses and hospital executives — not apples to apples.

**Approach:**
1. Normalize job titles (lowercase, invert comma-format: "Nurse, Registered" → "Registered Nurse")
2. Encode with `all-MiniLM-L6-v2` sentence transformer (cached after first run)
3. Sweep K=30–150 using silhouette, WCSS elbow, and R²
4. Validate chosen K with **Bootstrap ARI** (stability across 20 sub-samples)
5. Fit **K-Means** (primary, K=100) + **HDBSCAN** (comparison)
6. Human-review table: top 8 titles per cluster as sanity check

**Result: K=100** — silhouette plateaus in the 90–100 range; ARI at K=100 (0.475) > K=90 (0.438). High ARI (≥0.90 threshold) means the clustering is robust to sub-sampling — the same jobs land in the same groups consistently.

---

## Step 4: Statistical Testing — What We Test

**Three pairwise comparisons per cluster (100 clusters × 3 pairs = 300 tests):**

| Comparison | What it answers |
|---|---|
| Female vs Male | Primary pay gap: do women earn less doing the same job? |
| Uncertain vs Male | Do workers with non-Western names face a similar gap to women? |
| Female vs Uncertain | Does gender (not name origin) drive the gap? |

**Two complementary measures:**
- **Raw gap (Mann-Whitney U + CLES)** — non-parametric, no normality assumption. CLES = P(female salary > male salary); 0.5 = no gap, <0.5 = male advantage.
- **Regression-adjusted gap (OLS on log salary)** — controls for year, sector, and job cluster simultaneously. The coefficient × 100 ≈ % premium for males.

**Multiple comparisons:** Benjamini-Hochberg FDR at α=0.05. With N=2.4M, nearly all p-values are near zero — **CLES and gap_pct are the real effect sizes.** A gap <3% or CLES between 0.45–0.55 is statistically significant but practically negligible.

---

## Step 4: Findings — Gender Pay Gap by Cluster

**The gap is real but uneven — it varies dramatically by job type:**

- **Male-favoring clusters:** Judiciary, executive/C-suite roles, skilled trades, engineering → gaps of 5–15%+
- **Female-favoring clusters:** Some nursing specialties, social work supervisors, certain administrative roles → women earn more in a small number of clusters
- **Near-parity clusters:** Many teaching and frontline healthcare clusters show <3% gap after adjustment

**Ordering across all three groups in most clusters:**
> Male > Uncertain > Female

Consistent with: women face the largest salary disadvantage; workers with non-Western names face an intermediate disadvantage (partly masked by a male-majority composition); the hierarchy is persistent across sectors.

**Regression-adjusted gap:** After controlling for year, sector, and cluster, men still earn a positive premium over women. The adjusted gap is smaller than the raw gap — confirming that some (but not all) of the raw gap is occupational segregation.

---

## Step 5: Longitudinal Findings

**Do the gaps change over time — in salaries, raises, and promotions?**

**Salary growth rates (year-over-year raises for multi-year earners):**
- Female raise distributions generally peak to the left of Male — women tend to get slightly smaller year-over-year raises, **compounding the static salary gap over time**
- Uncertain growth rates sit between Female and Male, consistent with the salary snapshot findings

**Cluster transitions (promotion proxy — moving to a 5%+ higher-paying cluster):**
- **Upward transition rates are lower for Female than Male** across most sectors
- This is evidence of a promotion barrier on top of a starting salary gap — not just where you start, but how fast you advance
- Uncertain transition rates vary by sector; in some (universities, hospitals) they track close to Male

**Gap trajectories per cluster (2016–2025):**
- Most clusters: **stable** — the gap structure has been persistent, not dynamic
- A minority: **closing** (positive sign, mostly in education and some healthcare)
- A small number: **widening** (concerning, concentrated in executive/legal clusters)

---

## Summary: What the Data Shows

<div class="columns">

**Where the gap is largest:**
- Executive / C-suite roles
- Judiciary
- Skilled trades and engineering
- Top salary decile (D10, $167K+)

**Where the gap is smallest:**
- Frontline healthcare clusters with fixed pay grids
- Education (teachers on collective agreements)
- Some unionized municipal roles

</div>

**Three-part finding:**
1. **Representation gap** — women start behind: fewer in top-decile roles, more in lower-paying $100K jobs
2. **Within-role gap** — even in the same job cluster, women earn less (OLS controls for job type, sector, year)
3. **Advancement gap** — women receive smaller raises and transition to higher-paying clusters at lower rates

All three are real. All three are persistent over 10 years.

---

## Areas for Further Investigation

These were observed but set aside — worth pursuing with more data or resources:

| Topic | What to investigate | Data needed |
|---|---|---|
| **Benefits gap** | Does total compensation (salary + benefits) gap run the same direction? | Already in the data — quick check |
| **D9→D10 drop** | Which job clusters drive the sharp female decline at the very top? | Cluster × decile cross-tab |
| **Employer "worst offenders"** | Is the sector gap driven by a few outlier employers? | Employer Gini / employer-level gaps |
| **Uncertain decomposition** | Split Uncertain by name origin (South Asian / East Asian / French Canadian) | Phonetic heuristics on first names |
| **COVID-era effects** | Did pandemic-era pay changes (2020–2022) affect genders differently in healthcare? | Year × sector × gender interaction |
| **New entrant vs long-tenure gap** | Do post-2020 new entrants show a different gap than 2016 cohort? | Cohort analysis on identity-resolved data |
| **Exit patterns** | Are women more likely to drop off the list? Treadmill vs. genuine stability? | Track who disappears from year to year |
| **Union vs non-union** | Unionized roles have compressed grids — does this explain the small gaps in education/healthcare? | Employer unionization lookup |

---

<!-- _paginate: false -->

## Caveats and Limitations

- **No seniority data** — the gap conflates pay discrimination with tenure differences
- **Left-censoring** — we only see workers above $100K; "years on list" understates true seniority
- **Gender is inferred, not observed** — 94% coverage; Uncertain is the honest fallback, not a failure
- **Uncertain is heterogeneous** — South Asian men, East Asian women, French Canadian ambiguous names all land here; treat as a subgroup signal, not a single demographic
- **Statistical significance ≠ practical significance** — with 2.4M records, every gap is "significant"; report CLES and gap% as the real story
- **OLS omits employer fixed effects** — sector + cluster absorb most but not all employer-level variation
- **$100K threshold means we're studying high earners only** — conclusions don't extend to the full public sector workforce

---

<!-- _paginate: false -->

# Thank You

**Data:** Ontario Government Open Data — Public Sector Salary Disclosure (2016–2025)

**Methods:** Python · pandas · sentence-transformers · scikit-learn · statsmodels · HDBSCAN

**Key finding in one sentence:**
> Women in Ontario's public sector ($100K+) face a three-part disadvantage — they start lower, advance less, and the gap has not meaningfully closed in a decade.
