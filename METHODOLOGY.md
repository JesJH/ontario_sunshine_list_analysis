# Methodology

Full documentation of the analysis pipeline — data processing, gender inference, NLP clustering, statistical tests, and longitudinal analysis.

---

## Step 1: Data Loading and Normalization

The raw CSVs have changed schema several times across years (e.g. column renamed from "Salary Paid" to "Salary"). The loader normalizes all variants, cleans dollar formatting, and applies several normalization passes before any analysis runs.

**Sector normalization** collapses variants that differ only in punctuation (em-dash vs. hyphen, `&` vs. `and`, inconsistent casing). All "Seconded (Hospitals)", "Seconded (Universities)" etc. variants are collapsed to "Seconded" since the seconded arrangement is what's analytically meaningful. Result: 12 clean sectors from ~20 raw variants.

**Employer normalization** strips trailing punctuation, normalizes casing, and collapses duplicate name variants across years.

**Job title normalization** runs several passes in sequence to create a `job_title_norm` column:
1. Lowercase, strip, collapse whitespace
2. Normalize spaces around slashes
3. Strip French bilingual suffixes — e.g. "Registered Nurse / Infirmière Autorisée" → "registered nurse" (identified by accented characters after a slash)
4. Invert HR comma-convention — "Nurse, Registered" → "registered nurse"
5. Apply a canonical mapping for known word-order variants — "teacher elementary" → "elementary teacher"

Normalizing at load time (rather than inside the clustering module) means `job_title_norm` is available in every section of the notebook, including the uncertain gender breakdown tables in Section 2.

---

## Step 2: Gender Inference

Gender is not in the raw data. We infer it from first names using a weighted ensemble of four independent sources, each producing P(female) — the probability a given name belongs to a woman.

### Sources and weights

| Source | Weight | How it works |
|---|---|---|
| SSA lookup | 50% | US Social Security Administration baby name counts, birth years 1950–2003. P(female) = female_births / total_births per name. Very accurate for common Western names. |
| Ontario baby names | 20% | Ontario provincial birth registration data for the same birth year range. Adds local signal for names more common in Canada. |
| Character n-gram model | 20% | Logistic regression trained on 2–4 character sequences from SSA names (e.g. names ending in "-a" or "-ine" tend to be female). Generalizes to names not in SSA — South/East Asian, Arabic, and other names with learnable phonetic patterns. |
| HuggingFace model | 10% | Optional text-classification transformer. Off by default; enable by passing `hf_model_name=` to `GenderClassifier()`. |

### How sources are combined

Each source that has a signal for a given name contributes its P(female) estimate, weighted by the values above. Sources with no entry for a name are dropped from the denominator and the remaining weights are renormalized. A name absent from SSA but present in Ontario data and recognized by the char-gram model gets those two sources reweighted to sum to 1.0.

### Classification thresholds

The combined P(female) score is the confidence metric:
- P(female) near **1.0** → sources agree the name is female
- P(female) near **0.0** → sources agree the name is male
- P(female) near **0.5** → genuinely ambiguous

**Chosen thresholds: 0.70 / 0.30**
- P(female) ≥ 0.70 → **Female**
- P(female) ≤ 0.30 → **Male**
- 0.30 < P(female) < 0.70 → **Uncertain**

These were chosen from a sensitivity table across six threshold pairs (0.55/0.45 through 0.80/0.20). At 0.70/0.30, coverage is 94% — loosening to 0.55/0.45 adds only 4.5 percentage points of coverage but pulls in names that are nearly coin-flip ambiguous, adding noise to gender tests.

**Uncertain is a first-class category**, not a fallback. It is kept in all aggregate statistics and included as a third group in all statistical comparisons alongside Female and Male.

### Assumptions

- Names from SSA birth years 1950–2003 are representative of workers earning $100K+ between 2016 and 2025.
- The phonetic patterns the char-gram model learns from SSA names generalize partially to non-Western names — this is imperfect, which is why South Asian and East Asian names disproportionately fall into Uncertain.
- The growing Uncertain share (5.0% in 2016 → 6.9% in 2025) reflects workforce diversification, not data quality degradation.

---

## Step 3: NLP Job Title Clustering

The Sunshine List has ~147,000 distinct normalized job titles — mostly long-tail variations of a smaller number of real roles. Clustering groups semantically similar titles so salary comparisons are apples-to-apples.

### What a cluster represents

A cluster is a group of job titles the embedding model judged to be semantically similar. For example, one cluster might contain "registered nurse", "charge nurse", "nurse practitioner", and "staff nurse"; another might contain "chief financial officer", "vp finance", and "director of finance". The cluster ID is just a number — meaning comes from scanning the top titles in the cluster review table. This human review step catches cases where two unrelated roles ended up in the same cluster (K too small) or one role is split across two clusters (K too large).

### Why sentence embeddings instead of keyword matching

Keyword matching correctly groups "teacher" with "elementary teacher" but fails on abbreviations ("RN" vs. "registered nurse"), word-order variants, and role equivalents with no shared keywords. Sentence embeddings capture meaning — a model trained on billions of sentences has learned that "staff nurse" and "charge nurse" appear in similar contexts.

### Pipeline

1. Normalize titles (see Step 1)
2. Embed each unique title with `all-MiniLM-L6-v2` via `sentence-transformers` — a fast, local 384-dimensional model. Embeddings are cached after the first run.
3. Select K using three signals computed on a 30,000-point random subsample for speed:
   - **Normalized WCSS (elbow)** — fraction of total variance *unexplained* by clustering; lower is better; normalized so it doesn't change with dataset size
   - **Silhouette score** — how well each point fits its own cluster vs. the next-closest; higher is better; peaks indicate natural groupings
   - **R²** — explained variance fraction; look for where gains flatten
4. Validate with **bootstrap ARI** — subsample 80% of titles 20 times, re-cluster, assign held-out titles to the nearest centroid, measure agreement with the reference clustering. ARI = 1.0 is perfect, 0 = random.
5. Run **K-Means** (primary) and **HDBSCAN** (comparison, subsampled to 10K points due to O(n²) complexity)
6. Human review of top titles per cluster

### Why K=100

K=100 had better bootstrap ARI (0.475) than K=90 (0.438), and silhouette peaks in the 90–100 range. ARI of ~0.47 is normal for text clustering — job title clusters have soft, overlapping semantic boundaries. It does not indicate a bad clustering. One hundred clusters gives enough granularity to separate teachers from professors from administrators from nurses, while keeping each cluster large enough for reliable statistical tests.

### Assumptions

- Sentence-level semantic similarity is a reasonable proxy for "comparable job". Two roles with similar titles can still have different pay scales or unionization — the human review layer is the check on this.
- The 30K-point subsample used for K-selection is representative of the full 147K title space.

---

## Step 4: Statistical Testing

Two complementary measures of gender bias are computed for each job cluster, across three pairwise comparisons: **Female vs Male**, **Uncertain vs Male**, and **Female vs Uncertain**.

### Handling group size imbalance

Any cluster with fewer than 10 records in either group is flagged **sparse** and excluded from tests. For clusters with sufficient data, unequal group sizes are not a problem — the Mann-Whitney test operates on ranks, not counts, so a cluster with 5,000 women and 200 men is handled correctly.

### Mann-Whitney U test (raw gap)

For each cluster and each pairwise comparison, we compare salary distributions using the **Mann-Whitney U test** — a non-parametric test that makes no assumption about the shape of the distribution. It answers: *if I pick one person from group A and one from group B at random, what is the probability A earns more?*

**Why not a t-test?** Salary data is right-skewed (a few very high earners distort the mean). Mann-Whitney works on ranks rather than raw values, so it is not distorted by outliers or skew.

### Effect size: CLES

The result of Mann-Whitney is the **Common Language Effect Size (CLES)** = P(group A salary > group B salary):
- **0.50** = no difference
- **< 0.50** = group B tends higher (e.g. 0.45 means group A earns more only 45% of the time)
- **> 0.50** = group A tends higher

CLES is the primary effect size reported because with N=2.4M records, p-values are essentially zero for any non-zero effect. CLES tells you whether the gap is *meaningful*, not just detectable. A gap_pct of <3% or CLES between 0.45–0.55 is statistically significant but practically negligible.

### Benjamini-Hochberg FDR correction

Running 100 tests at α = 0.05 means roughly 5 would appear significant by chance even if nothing is real. **Benjamini-Hochberg (BH)** ranks all p-values and asks for each: "Given I'm willing to accept that at most 5% of my flagged findings are false alarms, does this p-value qualify?" Tests that survive are called BH-significant, and among them you expect no more than 5% to be false discoveries.

**Why BH instead of Bonferroni?** Bonferroni divides α by the number of tests (0.05 / 100 = 0.0005), controlling the probability of even *one* false positive — appropriate for safety-critical settings. BH controls the *rate* of false discoveries, which is appropriate for exploratory research where the goal is to identify candidates for investigation, not make a single definitive claim.

**Assumptions:** Tests are independent or positively correlated (mild in practice); p-values are correctly computed under the null.

### OLS Regression (adjusted gap)

A pooled Ordinary Least Squares regression on all records (Female, Male, and Uncertain):

```
log(salary) ~ gender + year + C(sector) + C(job_cluster)
```

Female is the baseline. The coefficients for Male and Uncertain give the adjusted salary premium vs Female *after* controlling for year (salary growth over time), sector (hospitals vs municipalities), and job cluster (nurses vs executives). The derived Uncertain vs Male gap = Uncertain coefficient − Male coefficient.

**Why log-transform salary?** Salary data is right-skewed. Log-transforming makes the distribution closer to normal (an OLS assumption) and makes coefficients interpretable as percentages: a coefficient of 0.035 means approximately 3.5% more, not $3.50.

**Interpreting the three comparisons together:**

| Ordering | What it means |
|---|---|
| Male > Uncertain > Female | Classic gradient — name-based disadvantage is real but smaller than gender disadvantage |
| Male > Female > Uncertain | Workers with non-Western names fare worse than even women — strongest intersectional signal |
| Male > Female ≈ Uncertain | Gender is the primary driver; name origin adds little additional disadvantage |
| Uncertain ≈ Male > Female | Uncertain pool is male-majority in salary terms; gender is the primary axis of disadvantage |

**Interpreting raw vs adjusted gap:**

| Raw gap | Adjusted gap | Likely explanation |
|---|---|---|
| Large, significant | Near zero | Occupational segregation — women in lower-paying clusters, not underpaid *within* them |
| Both significant | Both significant | Within-job pay discrimination — most concerning |
| Near zero | Significant | Women in higher-paying clusters but still underpaid vs male peers |

**Assumptions:**
- Key omitted variable is seniority — the adjusted gap partly reflects men having more years in role
- Employer fixed effects are omitted (thousands of levels make OLS impractical); sector + cluster capture most variation; a mixed-effects model with employer random effects would be the rigorous extension
- Clusters with fewer than 10 records per group are excluded (flagged sparse)

---

## Step 5: Longitudinal Trend Analysis

Longitudinal analysis tracks the *same people across years* rather than comparing annual snapshots. Records are linked via `first_name_clean + last_name + employer` to create a `person_id`. People who change employers are not linked — conservative, but avoids false merges.

### Salary growth rates

For each person with 2+ consecutive years on the list, we compute year-over-year salary growth: `(salary_t − salary_{t−1}) / salary_{t−1}`. Growth rate distributions are compared across all three gender groups using Mann-Whitney + CLES, answering: *even if starting salaries were equal, are women getting smaller raises?*

### Cluster transitions (promotion proxy)

For each consecutive year pair, we check whether a person's job title moved to a cluster whose median salary is at least 5% higher (upward), 5% lower (downward), or equivalent (lateral). Upward transition rate by gender over time is the closest available proxy for promotion rates — the Sunshine List has no explicit promotion field.

### Gap trajectories

For each cluster, we compute the Female−Male median salary gap in every year from 2016 to 2025 and fit a linear trend. A positive slope (≥ +0.001/year) = closing; negative slope (≤ −0.001/year) = widening; otherwise = stable. This classifies each cluster's gap as closing, widening, stable, or insufficient data.

### Uncertain in longitudinal analysis

Uncertain-gender workers are included in all three longitudinal measures as a third reference group. Where their growth rates or transition rates fall relative to Female and Male reveals whether name-based disadvantage is static (captured in Section 4 snapshots) or compounds over time through smaller raises and fewer promotions.

### Left-censoring caveat

The $100K threshold means we only observe people after they cross it. Someone who crossed in 2022 looks like they have 3 years of seniority but may have been in the role for 15. This biases "years on list" as a seniority proxy, and biases it more for groups whose salaries started lower. All longitudinal measures should be interpreted with this limitation in mind.

---

## Key Caveats

**No seniority or experience data.** Gaps conflate discrimination with tenure differences. A male-dominated cluster may pay more simply because men have been in the role longer.

**Left-censoring.** The $100K threshold means early-career salaries and promotions are invisible. New entrants appear to start at $100K regardless of prior career length.

**Gender is inferred, not observed.** 94% coverage at the 0.70/0.30 threshold. Uncertain records are kept as a first-class category and included in all analyses.

**Uncertain is heterogeneous.** Punjabi men, East Asian women, French Canadian ambiguous names, and genuinely gender-neutral English names all land in Uncertain. Treat it as a subgroup signal, not a single demographic.

**Statistical significance ≠ practical significance.** With N=2.4M, every non-zero effect is significant. Focus on CLES and gap_pct as effect sizes — clusters with gap_pct < 3% or CLES between 0.45–0.55 are detectable but not meaningful.

**Ontario public sector only.** Sector composition, unionization, pay equity legislation, and public scrutiny all differ from the private sector. Results are not generalizable beyond Ontario's public sector $100K+ earners.
