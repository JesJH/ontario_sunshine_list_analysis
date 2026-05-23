# Ontario Sunshine List — Gender Bias Analysis

## Background

The Government of Ontario annually publishes the Public Sector Salary Disclosure — commonly called the "Sunshine List" — naming every public sector employee who earned more than $100,000 in a given year. This project uses that data (2016–2025) to detect and characterize gender-based salary bias across job categories and sectors.

The core questions:
- Is there a measurable salary gap between men and women at comparable job levels?
- Which sectors and job clusters show the strongest bias?
- Has the gap been closing, widening, or staying flat over time?

The goal is analysis rigorous enough to publish, presented accessibly. All tools are free and run locally — no external APIs.

---

## Key Findings

### 1. Women are overrepresented at the bottom of the high-salary range; men dominate the top

The salary decile chart shows a consistent gradient: women make up ~62% of confirmed earners in the lowest decile ($100K–$104K) and fall to ~38% in the top decile ($167K+). This is the clearest headline finding — women are more likely to be just barely on the Sunshine List and less likely to be at the very top.

### 2. The overall gender split has been stable for a decade

Women are ~52–53% of confirmed-gender earners in every year from 2016 to 2025. The aggregate gap has not meaningfully closed or widened. This means change is happening *within* sectors and job categories — which is why the cluster-level analysis matters.

### 3. The gap is partly structural (occupational segregation) and partly within-job

- **Raw gap**: 91 of 100 job clusters show a statistically significant salary gap after FDR correction. 86 clusters favor men; 5 favor women.
- **Regression-adjusted gap**: After controlling for year, sector, and job cluster, men still earn a **+3.5% premium** on average (95% CI: [3.45%, 3.56%]).

The fact that the gap survives adjustment is stronger evidence than the raw gap alone — it means men earn more even compared to women *in the same job category and sector*, not just because they're in different types of work.

### 4. The sectors with the largest gaps are healthcare and public safety

- School boards (~70% female) and hospitals (~67% female) are female-dominated but pay at the lower end of the $100K+ range.
- Ontario Power Generation (~20% female) and municipalities (~27% female) are male-dominated and pay substantially higher.
- Occupational segregation — not within-job discrimination alone — accounts for a meaningful portion of the aggregate gap.

### 5. The Uncertain share is growing as a diversification signal

Records that couldn't be gender-classified grew from 5.0% (2016) to 6.9% (2025). This reflects more workers with South Asian, East Asian, and French Canadian names crossing the $100K threshold — names that fall outside North American gender reference data or are genuinely ambiguous. It is not a data quality problem; it is the Sunshine List slowly reflecting Ontario's population.

---

## Data Source

Ontario Government Open Data: https://www.ontario.ca/page/public-sector-salary-disclosure

CSVs are not included in this repo — download them from the link above and place them in `data/`. The loader handles all schema differences across years automatically.

---

## Setup & Running the Analysis

### Step 1 — Get the data

Download the salary CSVs from the Ontario open data portal and place them in `data/`. The data loader globs `data/*.csv` and handles all schema differences across years. No renaming needed.

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3 — Download gender name data

```bash
python scripts/download_gender_data.py
```

Downloads ~8 MB of SSA baby name files (birth years 1950–2003) and Ontario baby name data to `data/gender/`. Only needs to run once.

### Step 4 — Train the gender model

```bash
python scripts/train_gender_model.py
```

Trains a character n-gram logistic regression on the SSA names and caches it to `data/gender/char_model.pkl`. Takes about a minute. Only needs to run once.

### Step 5 — Open the notebook

```bash
jupyter notebook analysis.ipynb
```

Run cells top to bottom. Two cells will be slow on first run only:

| Cell | Why | Time |
|---|---|---|
| Section 2 — gender inference | Classifies every unique first name | ~1–2 min |
| Section 3 — embeddings | Encodes all unique job titles with sentence-transformer | ~1–2 min on CPU |

All embeddings are cached to `data/embeddings/` after the first run, so re-running is instant.

### Step 6 — Review clusters before running tests

Before running statistical tests, do a quick sanity check:

1. Look at the **K-selection plots** (Section 3) — pick a K where the elbow and silhouette agree
2. Check the **bootstrap ARI** for your candidate K — aim for ≥ 0.90 (lower is acceptable for text clustering)
3. Scan the **cluster review table** — if two clearly different roles share a cluster, increase K; if one role splits across two clusters, decrease it or post-merge
4. Set `K = <your value>` in the clustering cell and re-run from there

---

## Methodology

### Step 1: Data Loading and Normalization

The raw CSVs have changed schema several times across years (e.g. column renamed from "Salary Paid" to "Salary"). The loader normalizes all variants, cleans dollar formatting, and applies several normalization passes:

- **Sector normalization**: collapses variants that differ only in punctuation (em-dash vs. hyphen, `&` vs. `and`, inconsistent casing). "Seconded (Hospitals)" and "Seconded (Universities)" are all collapsed to "Seconded" since the seconded arrangement is what's analytically meaningful. Result: 12 clean sectors from ~20 raw variants.
- **Employer normalization**: strips trailing punctuation, normalizes casing, collapses duplicate name variants across years.
- **Job title normalization**: several passes in sequence:
  1. Lowercase, strip, collapse whitespace
  2. Normalize spaces around slashes
  3. Strip French bilingual suffixes (e.g. "Registered Nurse / Infirmière Autorisée" → "registered nurse") — identified by accented characters after a slash
  4. Invert HR comma-convention: "Nurse, Registered" → "registered nurse"
  5. Apply a canonical mapping for known word-order variants ("teacher elementary" → "elementary teacher")

**Why normalize job titles at load time?** The alternative — normalizing only inside the clustering module — means the raw variants show up in every earlier table (uncertain gender breakdown, sector analysis). Normalizing at load time means `job_title_norm` is available everywhere in the notebook from Section 2 onwards.

---

### Step 2: Gender Inference

Gender is not in the raw data. We infer it from first names using a weighted ensemble of four sources:

| Source | Weight | What it does |
|---|---|---|
| SSA lookup | 50% | US Social Security Administration baby name counts (birth years 1950–2003). Computes P(female) = female_births / total_births per name. Very accurate for common Western names. |
| Ontario baby names | 20% | Ontario provincial birth registration data for the same birth year range. Provides local-specific signal for names that differ in regional prevalence. |
| Char n-gram model | 20% | Logistic regression on 2–4 character n-grams, trained on SSA names. Generalizes to names not in SSA — South/East Asian names and others with predictable phonetic patterns. |
| HuggingFace model | 10% | Optional text-classification transformer. Off by default; enable by passing `hf_model_name=` to `GenderClassifier()`. |

Weights are renormalized at inference time when a source has no signal for a name (e.g. a name not in SSA simply doesn't count toward the denominator).

**Classification thresholds (chosen = 0.70/0.30):**
- P(female) ≥ 0.70 → **Female**
- P(female) ≤ 0.30 → **Male**
- Otherwise → **Uncertain**

The notebook includes a threshold sensitivity table showing coverage and gender split across six threshold pairs (0.55/0.45 to 0.80/0.20). The 0.70/0.30 threshold was chosen because it gives 94% coverage while keeping the gender split stable — widening to 0.55/0.45 only adds 4.5 percentage points of coverage but introduces names that are nearly coin-flip ambiguous.

**Uncertain records** are excluded from all gender tests but kept in aggregate statistics. Coverage and the Uncertain % are reported by year.

**Assumptions:**
- Names from SSA birth years 1950–2003 are representative of the workforce that appears on the Sunshine List. People who earned $100K+ between 2016 and 2025 were typically born in roughly that window.
- The phonetic patterns the char-gram model learns from SSA names generalize partially to non-Western names. This is imperfect — South Asian and East Asian names have different phonetic structures — which is why they disproportionately fall into Uncertain.

---

### Step 3: NLP Job Title Clustering

The Sunshine List has ~147,000 distinct normalized job titles. Most of these are long-tail variations of a smaller number of real roles. Clustering groups semantically similar titles so that salary comparisons are apples-to-apples.

**What does a cluster represent?**

A cluster is a group of job titles that the sentence embedding model judged to be semantically similar. For example, one cluster might contain "registered nurse", "charge nurse", "nurse practitioner", and "staff nurse" — all nursing roles that sit close together in the 384-dimensional embedding space. Another cluster might contain "chief financial officer", "vp finance", and "director of finance". The cluster ID itself is just a number; the meaning comes from scanning the top titles in the cluster review table (Section 3). This human review step is important — it catches cases where two unrelated roles ended up in the same cluster (a sign K is too small) or one role is split across two clusters (K too large).

**Why sentence embeddings instead of keyword matching?**

Keyword matching would correctly group "teacher" with "elementary teacher" but would fail on abbreviations ("RN" vs. "registered nurse"), word order differences ("director of operations" vs. "operations director"), and role equivalents with no shared keywords. Sentence embeddings capture meaning: a model trained on billions of sentences has learned that these phrases are used in the same contexts.

**Pipeline:**
1. Normalize titles (see Step 1)
2. Embed unique titles with `all-MiniLM-L6-v2` via `sentence-transformers` — a fast, local 384-dimensional model, cached after first run
3. Select K using three signals run on a 30,000-point subsample (for speed):
   - **Normalized WCSS (elbow)**: unexplained variance fraction — look for the bend where adding clusters stops helping
   - **Silhouette score**: measures how well each point fits its cluster vs. the next-closest cluster — higher is better
   - **R² (BCSS/Total SS)**: explained variance fraction — look for where gains flatten
4. Validate the chosen K with **bootstrap ARI**: subsample 80% of titles 20 times, re-cluster, assign held-out titles to nearest centroid, measure agreement with reference clustering. ARI ≥ 0.90 = very stable; the text embedding domain typically yields 0.40–0.70 because job title clusters have soft boundaries, not hard ones.
5. Run K-Means (primary, K=100) and HDBSCAN (comparison — density-based, no fixed K, subsampled to 10K points due to O(n²) complexity)
6. Human review of top titles per cluster

**Why K=100?**

K=100 had a better bootstrap ARI (0.475) than K=90 (0.438), and the silhouette score peaks in the 90–100 range. The WCSS curve is still declining at K=100 but the gains have diminished. One hundred clusters gives enough granularity to separate teachers from professors from administrators from nurses, while keeping each cluster large enough for reliable statistical tests.

**Assumptions:**
- Sentence-level semantic similarity is a reasonable proxy for "comparable job". This is imperfect — two roles with similar titles can have different responsibilities, unionization, or pay scales. The human review layer is the check on this.
- The 30K-point subsample used for K-selection is representative of the full 147K title space. This holds in practice because the distribution of titles is not heavily clustered at the subsample scale.

---

### Step 4: Statistical Testing

Two complementary measures of gender bias are computed for each job cluster.

#### Mann-Whitney U test (raw gap)

For each cluster, we compare the salary distributions of female vs. male earners using the Mann-Whitney U test. This is a non-parametric test — it makes no assumption about whether salaries are normally distributed (they are not; salary distributions are right-skewed). It answers the question: "If I picked one woman and one man at random from this cluster, what is the probability the woman earns more?"

**Why Mann-Whitney instead of a t-test?**

A t-test assumes salaries within each group are approximately normally distributed. Salary data is right-skewed (a few very high earners pull the mean up). The Mann-Whitney test works on ranks rather than raw values, so it is not distorted by outliers or skew. It is also equivalent to computing the Common Language Effect Size (CLES) directly, which makes interpretation intuitive.

**Effect size: CLES**

CLES = P(female salary > male salary). A value of:
- **0.50** = no difference — a randomly picked woman is equally likely to earn more or less than a randomly picked man
- **< 0.50** = males tend higher (e.g. 0.45 means there is a 45% chance the woman earns more, 55% the man does)
- **> 0.50** = females tend higher

CLES is the right effect size to report here because with N=2.2M records, every non-zero gap is "statistically significant" — p-values are essentially zero. CLES tells you whether the gap is *meaningful*, not just detectable. A CLES of 0.48 vs. a CLES of 0.35 are both "significant" but one is a rounding error and the other is a 15 percentage point effect.

#### What is the Benjamini-Hochberg FDR correction?

We run 100 separate Mann-Whitney tests, one per cluster. Running 100 tests at α = 0.05 means that even if there were *no real effect anywhere*, we would expect 5 false positives just by chance. The Benjamini-Hochberg (BH) procedure controls this.

**In plain language:** BH ranks all 100 p-values from smallest to largest, then asks for each one: "Given that I'm looking at test #k out of 100, and I'm willing to accept that up to 5% of my 'significant' findings are false, does this p-value still qualify?" Tests that survive this stricter standard are called BH-significant. The result is that among all the clusters you flag as significant, you expect no more than 5% to be false alarms.

**Why BH instead of Bonferroni?**

Bonferroni divides α by the number of tests (0.05 / 100 = 0.0005). This controls the probability of even *one* false positive across all 100 tests — extremely strict, appropriate for safety-critical settings. BH instead controls the *rate* of false discoveries: among the clusters you flag, 5% are allowed to be wrong. For exploratory research — where the goal is to identify candidates for further investigation, not to make a single definitive claim — BH is the standard choice. It finds more true effects while keeping false discoveries bounded.

**Assumptions of BH:**
- Tests are independent or positively correlated (clusters are not fully independent since they come from the same dataset, but the correlation structure is mild)
- P-values are computed correctly under the null hypothesis of no gender difference

#### OLS Regression (adjusted gap)

A pooled Ordinary Least Squares regression on all confirmed-gender records:

```
log(salary) ~ gender + year + C(sector) + C(job_cluster)
```

The gender coefficient (×100) gives the approximate % salary premium for males *after* controlling for time (salary inflation), sector composition, and job type.

**Why log-transform salary?**

Salary data is right-skewed. Taking the log makes the distribution closer to normal, which is an OLS assumption. It also means the coefficient is directly interpretable as a percentage: a coefficient of 0.035 means males earn approximately 3.5% more, not $3.50 more (which would be a fixed-dollar interpretation and harder to compare across salary levels).

**Why include sector and cluster fixed effects?**

Without them, the gender coefficient would capture both within-job gaps AND the fact that men and women are in different sectors and jobs. By including sector and cluster dummies, we partial out those differences. The remaining coefficient isolates how much men earn versus women *doing the same type of job in the same sector*, after adjusting for year.

**Interpreting the two measures together:**

| Raw gap | Adjusted gap | Likely explanation |
|---|---|---|
| Large, significant | Near zero | Occupational segregation — women are in lower-paying clusters, not underpaid *within* them |
| Both significant | Both significant | Within-job pay discrimination — the most concerning finding |
| Near zero | Significant | Women in higher-paying clusters but still underpaid vs. male peers |

**Assumptions:**
- The OLS model is correctly specified — key omitted variables are seniority and experience, which are not in the data. The 3.5% adjusted gap could partly reflect men having more years in role.
- Employer fixed effects are omitted (thousands of levels make OLS impractical). Sector + cluster capture most of the variation; a mixed-effects model with employer random effects would be the rigorous extension.
- Clusters with fewer than 10 confirmed-gender records in either group are excluded from tests (flagged "sparse") to avoid underpowered results.

---

### Step 5: Longitudinal Trend Analysis *(in progress)*

Longitudinal analysis means tracking the *same people over time* rather than just comparing snapshots year by year.

**What it is:**

The Sunshine List gives us a record for each person in each year they appear. By linking records with the same name and employer across years, we can build a salary trajectory for each person — a path from their first year on the list through to the most recent.

**What questions it answers:**

- **Are women getting smaller raises?** Compare year-over-year salary growth rates within each cluster by gender. Even if women and men start at the same salary in a cluster, different raise rates compound into a gap over time.
- **Are women being promoted at lower rates?** A "cluster transition" happens when someone appears in a lower-paying cluster in year 1 and a higher-paying cluster in year 5 (their job title changed). Comparing transition rates by gender is a proxy for promotion rates, since the Sunshine List has no explicit promotion data.
- **Is the gap opening or closing over time?** Classify each cluster's gap trajectory: closing (gap shrinking each year), widening (gap growing), sudden appearance (new role type on the list), or stable.

**Left-censoring caveat:**

The $100K threshold creates a selection problem. Someone who has worked in a role for 20 years first appears on the list when their salary crosses $100K. So the "years on list" measure always understates true seniority — and it understates it more for people who started at lower salaries (who tend to be female). This biases seniority comparisons and means some apparent salary growth is really just selection: we're only seeing people once they're already earning at a certain level.

**Identity resolution:**

Linking records across years uses normalized first name + last name + employer as the primary key, with fuzzy matching as a fallback for employer name changes. Ambiguous matches (same name, multiple employers) are flagged and excluded from trajectory analysis.

---

## Key Caveats

**No seniority or experience data.** The gap we measure conflates true pay discrimination with tenure differences. A male-dominated cluster may pay more simply because men in that role have been there longer. We can partially proxy this with "years on list" but it is left-censored.

**Left-censoring.** The $100K threshold means we only observe people once they cross that level. New entrants appear to start at $100K regardless of prior career length. This biases seniority proxies and could inflate or deflate apparent gaps depending on which gender is crossing the threshold faster.

**Gender is inferred, not observed.** The ensemble is accurate for most Western names but coverage drops for names underrepresented in SSA data. All results should be read alongside the coverage %, and the inference method is flagged per record.

**Statistical significance ≠ practical significance.** With 2.2M records, any non-zero effect has a p-value near zero. We report CLES and gap_pct as effect sizes; a gap of 1% with CLES of 0.49 is "significant" but not meaningful. Focus on clusters with gap_pct > 3% or CLES below 0.45.

**Ontario public sector only.** The Sunshine List is not a representative sample of the Ontario workforce. Sector composition, unionization, pay equity legislation, and public scrutiny all differ from private sector employment.

---

## How to Interpret Results

1. **Start with the EDA (Section 2b)** — look at the salary decile chart to see if women are concentrated at lower pay bands overall, and check the sector split to understand occupational composition before any formal tests.

2. **Cluster review (Section 3)** — scan the human-review table before accepting the clustering. If two clearly different roles share a cluster, or one role splits across two, adjust K.

3. **Raw gap chart (Section 4)** — shows which job clusters have statistically significant salary gaps after FDR correction. Red bars = male-favoring. The size of the bar is the % gap in median salary.

4. **Regression coefficient (Section 4)** — the single number summarizing the average adjusted gap across all roles. Positive = males earn more on average, controlling for year, sector, and job type.

5. **Cross-reference the two measures** — use the interpretation table above. A large raw gap that shrinks to near-zero after adjustment points to occupational segregation. A gap that survives adjustment is stronger evidence of within-job pay differences.

6. **Effect sizes first, p-values last** — with 2.2M records, significance is guaranteed. The CLES and gap_pct columns tell you whether a cluster's gap is meaningful.
