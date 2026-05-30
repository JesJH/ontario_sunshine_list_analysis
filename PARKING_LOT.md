# Assumptions & Parking Lot

## Assumptions Made (explicitly flagged in analysis)

- **Gender inference accuracy**: Ensemble method (SSA + char model + optional HF) is assumed to be accurate enough for statistical patterns. No ground-truth validation against known individuals.
- **Name-to-gender stability over time**: SSA birth year distribution (1950–2003) reflects the Ontario workforce in 2017–2025. No update to birth cohort assumptions as workforce ages.
- **First name extraction**: Assumes first token (after skipping initials) is the given name used for gender lookup. May miss hyphenated or multi-word first names entered inconsistently.
- **Identity resolution across years**: Using `first_name_clean + last_name + employer` as primary key. Does not capture name changes (marriage, legal name change) or employer changes within the same person's trajectory.
- **Salary as sole outcome**: Treats salary as the metric for bias. Does not measure: hiring rates, promotion speed, job title advancement, benefits equity, or non-monetary compensation.
- **Cluster homogeneity**: Assumes sentence embeddings (`all-MiniLM-L6-v2`) cluster job titles by function/level. Does not validate against domain expertise or occupational classification systems (e.g., NOC).
- **Statistical independence**: Assumes observations (people × year) are independent for Mann-Whitney U tests. Violates assumption for longitudinal observations of same person.
- **Regression controls are sufficient**: OLS with year + sector + cluster fixed effects is assumed to adequately control for structural differences. Omitted variables (e.g., job tenure, education, prior experience) may bias estimates.
- **Left-censoring is symmetric across genders**: Assumes women and men enter the list at similar career stages. If women enter later (due to career breaks), observed gap underestimates discrimination.

---

## Parking Lot — To Consider Later

### Data & Measurement
- [ ] Validate gender inference on a sample of known individuals (e.g., public figures, cross-reference with LinkedIn/news articles)
- [ ] Investigate hiring/recruitment patterns — are certain job clusters male/female dominated by design, or by promotion/retention dynamics?
- [ ] Benefits breakdown — are salary gaps mirrored in taxable benefits gaps?
- [ ] Compare against other Canadian provinces' sunshine lists (if available) to contextualize Ontario patterns
- [ ] Job title validation — sample and manually review clusters to assess semantic coherence

### Statistical Robustness
- [ ] Mixed-effects model (person-specific random intercept) to account for repeated observations
- [ ] Sensitivity analysis: drop sparse clusters, refit, compare results
- [ ] Bootstrap confidence intervals on effect sizes (not just p-values)
- [ ] Test for interaction effects: do salary gaps vary by sector, employer size, or time period?
- [ ] Quantile regression: are gaps larger at the bottom, middle, or top of salary distribution?

### Longitudinal Mechanics
- [ ] Cross-employer transitions — do people who switch employers show different salary growth patterns by gender?
- [ ] Promotion speed proxy: compare time in cluster before moving to higher cluster, by gender
- [ ] Survivorship bias — do women and men leave the list (drop below $100K, retire, move provinces) at different rates?
- [ ] Entry cohort effects — cohort of first appearance (which year they entered the list) as a control

### Interpretation & Context
- [ ] Compare Sunshine List salary gaps to broader Canadian earnings data (e.g., Statistics Canada Labour Force Survey)
- [ ] Occupational segregation vs. discrimination: decompose raw gap into between-cluster (segregation) vs. within-cluster (discrimination) components
- [ ] Qualitative interviews or case studies with people in high-bias clusters to understand mechanisms
- [ ] Temporal heterogeneity: were certain sectors/clusters more biased in specific years (e.g., policy/legal changes)?

### Presentation & Reproducibility
- [ ] Create data dictionary / codebook for all derived variables
- [ ] Unit tests for normalization functions (sector, job title, name extraction)
- [ ] Sensitivity tables: effect of K selection, gender threshold, sparse cluster cutoff on findings
- [ ] Interactive visualization (e.g., Plotly/Streamlit) for exploring cluster-level gaps
- [ ] Document limitations table for publication

---

## Decisions Deferred (intentionally, for later discussion)

- **Weighting by employer size**: Should we weight observations by employer? (e.g., 1000-person ministry vs. 50-person agency). Currently unweighted.
- **Sector-stratified analysis**: Run separate analyses by sector (healthcare, education, municipal, etc.) to look for heterogeneity?
- **Non-binary gender**: Current framework is binary (Male/Female/Uncertain). Could Uncertain category be subdivided or treated as a signal of workforce diversity?
- **Intersectionality with ethnicity**: No ethnicity data in Sunshine List. Could names be used as a proxy for ethnicity to study compound bias? (Methodological risks noted.)
- **Historical comparison**: Extend back to 1996 if historical data can be sourced; is the trend in wage gaps accelerating or decelerating?

---

## Known Limitations (document in final output)

- **No seniority field**: Gap confounds discrimination with tenure/rank differences
- **Left-censoring**: Only observe people above $100K; "years on list" ≠ true career seniority
- **Gender is inferred**: Not self-reported; method has ~90% coverage, with residual "Uncertain" category
- **No occupational classification**: Job titles are textual; clustering may miss hierarchical levels (e.g., level 1 vs. level 3 manager)
- **Salary is nominal**: Not adjusted for inflation (2016–2025 span spans period with significant inflation)
- **Cross-sectional snapshots**: Each year is a snapshot; people move in/out, roles change, career stage changes
