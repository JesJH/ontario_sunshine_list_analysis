# How I Used Claude in This Project

This document describes how I used Claude Code (Anthropic's AI coding assistant) throughout this analysis. The goal is transparency about the human vs. AI contribution — what I directed, what I delegated, and where the back-and-forth shaped the final result.

---

## What I brought to the project

- The research question: detecting gender-based salary bias in Ontario's public sector using open data
- Decisions about methodology (described below)
- Domain judgment on outputs — reading charts, interpreting results, deciding what to investigate next
- Pushing back when something didn't look right

## What Claude built

- The full `src/` module suite: data loading, gender inference, clustering, statistical testing, longitudinal analysis
- The `analysis.ipynb` notebook structure and all cell code
- The README and METHODOLOGY documentation

---

## Key decisions I made (Claude explained options, I chose)

**Gender threshold at 0.70/0.30**
Claude showed a sensitivity table across six threshold pairs and walked through the coverage vs. accuracy tradeoff. I chose 0.70/0.30 — 94% coverage felt right; the 0.80/0.20 option dropped too many records.

**Treating Uncertain as a first-class label**
Early in the project Claude suggested backfilling Uncertain records based on similar confirmed-gender roles as a separate label. I decided against it — I wanted Uncertain to stay as its own category and be included in all analyses, not imputed away. This ended up shaping the entire analysis: every chart, every statistical test, and every commentary cell includes Uncertain as a third group alongside Female and Male.

**Choosing K=100 for clustering**
Claude ran bootstrap ARI for K=90 and K=100 and presented the tradeoffs (K=90 ARI=0.438, K=100 ARI=0.475). I chose K=100. Claude also explained what ARI means in plain language and why a score of ~0.47 is acceptable for text clustering even though it sounds low.

**Benjamini-Hochberg over Bonferroni**
Claude explained both options and recommended BH for exploratory research. I agreed — the goal is to flag clusters for investigation, not make a single definitive claim.

**Running three pairwise comparisons instead of just Female vs Male**
I asked to include Uncertain gender in all the statistical analysis so we could make commentary across all three pairs (Female vs Male, Uncertain vs Male, Female vs Uncertain). This required refactoring `raw_gap_by_cluster` to accept generic `group_a`/`group_b` parameters and updating the regression to include all three gender categories. The three-panel waterfall and the ordering interpretation table (Male > Uncertain > Female, etc.) came from this decision.

---

## Where I caught problems and redirected

**Charts were broken multiple times**
The % Female trend chart was visually wrong — the x-axis positions didn't align with the year values. I flagged it, Claude diagnosed the root cause (pandas `.plot()` placing data at index positions 2016–2025 while `set_xticks` expected 0–9), and fixed it by plotting directly against a years list.

**Job title duplicates weren't fixed after the first attempt**
After the first normalization pass, "Secondary Teacher" and "Teacher, Secondary" were still showing as separate entries in the tables. I flagged it. Claude traced the issue — normalization was only happening inside the clustering module, so it didn't affect earlier tables — and moved the normalization to load time in `data_loader.py` so `job_title_norm` was available everywhere.

**HDBSCAN ran for an estimated 2000+ hours**
I started the HDBSCAN cell and it was clearly not going to finish. I killed it and reported the problem. Claude explained the O(n²) complexity issue and rewrote the function to subsample to 10,000 points and use `approximate_predict` to assign the remaining points — bringing runtime down to seconds.

**K-selection was stuck at K=70 after 15 minutes**
Same class of problem — the original implementation used full K-Means on 165K points for every K value. I flagged it; Claude switched to MiniBatchKMeans on a 30K subsample, which completed the full sweep in under a minute.

**Sector and employer duplicates**
I noticed the sector chart was messy — "Hospitals & Boards" and "Hospitals and Boards" were separate rows, and there were a dozen "Seconded (X)" variants. Similarly for employers. I flagged both; Claude added normalization functions for each.

**Statistical significance with large N**
I raised the concern that with 2.2M records, everything would look significant. Claude explained the distinction between statistical and practical significance and introduced CLES (Common Language Effect Size) as the right metric to focus on. The analysis now leads with CLES and gap_pct rather than p-values.

---

## What Claude proactively caught

- Normalized WCSS / R² instead of raw WCSS — pointed out that raw inertia scales with dataset size and isn't comparable across subsample sizes. Switched to variance fractions before I asked.
- The left-censoring problem — explained unprompted that the $100K threshold means "years on list" understates seniority, and biases it differently by gender. This became a documented caveat throughout.
- The heterogeneity of the Uncertain category — flagged that Uncertain is not a single demographic (South Asian men, East Asian women, French Canadian ambiguous names, and genuinely gender-neutral English names all land there) and that interpreting it requires care.

---

## What the collaboration felt like in practice

Most sessions followed a pattern: I described what I wanted to see or understand, Claude implemented it or explained the options, I reviewed the output, and then I decided what to investigate next. The methodology decisions were mine; the implementation was Claude's.

The most useful moments were when I pushed back on something that didn't look right in the output — the broken charts, the duplicate job titles, the slow cells. Claude's explanations of *why* something was wrong were generally clear and helped me understand the underlying issue, not just accept the fix.

The statistical concepts (BH FDR, Mann-Whitney, CLES, bootstrap ARI) were all explained in plain language on request before any implementation happened. I asked for that explicitly — I wanted to understand what we were doing before agreeing to do it.

---

## Tools used

- **Claude Code** — Anthropic's AI coding assistant (CLI + VS Code extension)
- **Model** — Claude Sonnet 4.6
- All analysis tools are free and local: `sentence-transformers`, `scikit-learn`, `statsmodels`, `scipy`, `pandas`, `matplotlib`
