# NULLGARDEN

Executable negative controls for scientific machine learning. You load a labelled dataset with batches (or use the bundled bladderbatch-shaped designs), and NULLGARDEN reruns the same classifier under controls that remove the real signal but keep a suspected shortcut. If the score survives, the model was exploiting the shortcut, not learning the biology.

Built for bench scientists and ML reviewers who see a high cross-validated AUC on batch-structured data and need to know whether to believe it.

## Quick start

Requires Node 22 or newer. No npm dependencies.

```
export NEBIUS_API_KEY=...        # optional; enables the Nemotron features
node server.mjs                  # serves http://127.0.0.1:4412
node --test                      # 15 tests, none need the network
```

Open http://127.0.0.1:4412. Pick a dataset tab, press **Run controls**, read the verdict and the AUC ladder. With a key set, **Propose hypotheses** and **Explain results with Nemotron** are live. Without a key those two buttons show a plain message and everything else still works. `PORT` changes the port. A CSV can be loaded with the "Load CSV" button (columns: `sample_id,batch,label,feature...` or `sample_id,batch,label,subject_id,feature...`; label 0 or 1; at least 10 measurement rows). The optional `subject_id` is essential when multiple rows come from the same donor/patient/animal; it may not span batches or have contradictory labels.

## The controls

A fixed pipeline (top-20 feature selection by |t| inside each training fold, then diagonal LDA) is scored by cross-validation, then rerun under four controls: batch-only baseline, within-batch label permutation (with a shuffled-label null band), leave-one-batch-out, and batch-centered features. The rules in `src/core.mjs` turn the ladder of AUCs into a verdict. Nothing in that path calls a model.

## Independent units, repeated measurements, and blocked inference

A sample row is not necessarily an independent biological replicate. In a study with technical replicates, multiple imaging frames per person, longitudinal visits, or repeated assays, ordinary row-wise cross-validation leaks subject fingerprints between train and test; row-wise conditional permutations also generate label configurations that were *never physically randomized*. A small apparent p-value in that setting can be anti-conservative.

**Use the fourth CSV column `subject_id` whenever a donor contributes multiple measurements:**

```csv
sample_id,batch,label,subject_id,feature_1,feature_2
assay-001,runA,1,donor-P,2.5,1.3
assay-002,runA,1,donor-P,2.6,1.1
assay-003,runA,0,donor-N,0.4,2.0
assay-004,runA,0,donor-N,0.5,1.9
```

The native engine now treats subjects as **indivisible conditional-label allocation units**. Every row from one subject must belong to the same batch and carry the same binary label; contradictory subject metadata is rejected at CSV intake and direct library use. Stratified CV assigns entire subjects to folds, averages repeated held-out scores to one score per subject before within-fold AUROC, and reruns the **complete training-only feature selection + diagonal-LDA** under the same subject-heldout statistic for observed and randomized labels. The exact conditional orbit counts combinations of *subject labels* per batch, not combinations of individual rows, and the Monte Carlo path shuffles those same blocks. For example, six batches with two subject units (one of each label) and two measurements per unit have exactly **64 = 2^6** assignments, not 46,656 = 6^6 row-wise pseudo-permutations. The web app and Markdown export state the independent-unit count and grouping mode.

**Scope:** This is a one-level, subject-nested-in-batch design. It does not solve multi-level family/site hierarchies, subjects crossing batches, interventions changing labels within the same subject, time-series dependencies, or exchangeability under unmeasured common factors. A `subject_id` asserts an analysis grouping; it does not *prove* independent donors or justify exchanging donor-level labels. If `subject_id` is omitted, each row is **assumed** independent and the user is explicitly warned. This design controls the permutation *unit*, not the scientific causal interpretation. Exact inference requires a predeclared test statistic and an actually exchangeable subject-level null, while the fixed-budget sequential bound inherits its prior constraints.

The receiving suite includes independent 64-mask exhaustive p-value computation, finite-null type-I enumeration over all 64 legal observations, donor coassignment to folds, grouped Monte Carlo invariants, metadata contradiction failures, and end-to-end HTTP/CSV and exported-report checks. See [Winkler et al., *Multi-level block permutation*](https://pubmed.ncbi.nlm.nih.gov/26074200/) (NeuroImage 2015) for why exchangeability blocks matter and [Roberts et al., *Cross-validation strategies for hierarchical data*](https://doi.org/10.1111/ecog.02881) (Ecography 2017) for leakage from dependence in ordinary random splits.

## Falsification: an entire donor-level conditional null

Run `node research/subject-pseudoreplication-null.mjs` to reproduce a frozen **complete 64-assignment null**, using the production CV, feature selection, diagonal-LDA, permutation p-value and stopped/fixed-budget contract. The null observations are 12 subjects across six mixed batches, with one balanced binary subject assignment per batch and **no feature that predicts biology on a new donor**. Features are subject-identity indicators, representing technical fingerprints. All 64 donor-level relabelings are equiprobable under this declared synthetic null.

| Physical observations per subject | Legacy row-wise null rejects at 0.05 | Subject-blocked exact null rejects at 0.05 | Row-wise mean AUC | Subject-heldout mean AUC |
| --- | ---: | ---: | ---: | ---: |
| 1 (independent rows) | 0/64 | 0/64 | 0.500 | 0.500 |
| 3 (technical replicates) | **64/64** | **0/64** | **1.000** | **0.500** |

The invalid row test reports minimum p = 1/41 ≈ 0.0244 on all 64 dependent assignments: it mistakes donor-fingerprint memorization for learned biological signal. The blocked test finds no such evidence (p = 1 for every assignment). A **separate** positive synthetic control with genuine across-subject feature signal produces subject-heldout AUC = 1 and Monte Carlo p = 0.005 over 199 shuffles; blocking does not simply suppress every signal.

This is a deliberately adversarial, deterministic **one-design empirical counterexample** to row-wise analysis—not a universal estimate of type-I error, a human clinical cohort, an external generalization result, or a multi-hypothesis correction. The complete-null claims can be checked independently against the finite 64-point orbit. The workflow runs the benchmark after the standard scientific tests and frozen real-app scenario checks, publishing machine-readable result JSON and a SHA-256 digest in Actions logs without committing generated reports.

## Predeclared familywise discovery without post-hoc significance inflation

Register the family **at dataset intake**, e.g. `POST /api/dataset` with JSON `{"csv":"...","family":{"ks":[1,5,20],"seed":20261030,"perms":100}}`. The response includes an immutable family registration and SHA-256 fingerprint of the complete uploaded dataset plus its frozen plan. Subsequent `POST /api/family` accepts **only** `{"datasetId":"..."}`: it refuses modified ks, seeds or permutation budgets, even after results have been viewed. With no family specified on upload, an automatic outcome-independent grid is fixed before any app results. The exact digest demonstrates consistency of repeated calls within the running server, **not external timestamping or proof that a scientist did not preinspect the CSV outside this service**. This evaluates a **predeclared family** of 1–8 distinct feature-selection budgets using each pipeline's production subject-blocked or batch-conditional permutation null, then applies **Holm step-down** adjustment to the marginal p-values. Under valid marginal p-values, the adjusted results control familywise Type-I error for this family even when its member statistics and conditional permutations are arbitrarily dependent. By default, feature-count-only choices are `[1,min(5,d),min(20,d)]` with duplicates removed—no outcome-dependent grid search.

A subject-id CSV must use the existing `subject_id` column: repeated rows are held out and permuted as indivisible units and scored once per subject. When subject IDs are absent, the API explicitly assumes independent rows, an unverified scientific assumption. The exact conditional orbits and fixed-budget anytime decision bounds remain unchanged; if any member is unidentifiable, **all adjusted family decisions abstain** instead of pruning the family after seeing the result. The family endpoint is deliberately opt-in; regular `/api/run` remains a single-pipeline diagnostic.

**Holm-safe early stopping:** The original single-control decision stopping threshold of 0.05 is insufficient for a family: a preliminary conservative marginal upper p of 0.05 may adjust to 0.10 across two hypotheses even when the completed-budget p is 0.005. The familywise engine first applies Holm to the **pathwise lower and upper p vectors**. Only if all familywise decisions are already certain does it keep early conservative p bounds. If any decision is unresolved, it completes the *same seed and predeclared budget* for censored hypotheses before applying Holm. On the frozen positive transfer test, this changes two falsely retained adjusted p=0.10 decisions after 190/199 draws into two correctly rejected adjusted p=0.01 decisions after 199/199 draws. Separate null and changed-input suites ensure every resulting decision matches the full-budget oracle, while clearly disclosing the extra runtime.


This guarantees error control only for the fixed declared family and for the exchangeability null each member actually tests. It does **not** justify selecting a model family or endpoints after observing their p-values, trying alternative random seeds until favorable, changing the permutation budget, repeated uncorrected requests, or claiming causation. The independent finite 32-allocation grouped receiving checks marginal statistics, exact Holm adjustment, and the complete-null familywise false-rejection envelope. The cold subject-ID CSV HTTP journey demonstrates immutable intake registration, changed-input refusal, digest stability and idempotent repeat calls. A separate 24-subject synthetic positive signal verifies that two meaningful families can both survive Holm correction, not only abstain.

## Primary sample-lineage receiving: GEO GSE3167 falsifies independent-donor assurance

To qualify whether the **real** assay observations can legally be permuted as independent units, [`research/gse3167-origin-metadata.mjs`](research/gse3167-origin-metadata.mjs) streams and hashes the original **60-array** NCBI GEO [GSE3167 series matrix](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE3167) directly from NCBI FTP, parsing accession, array title, source and every public sample-level characteristic. It drains the complete compressed source without retaining or committing the gene-expression matrix. The 60 original GEO assays and the Bioconductor `bladderEset` 57 observations are **different denominators**; no unverified one-to-one mapping is asserted.

**Observed primary-source result:** the original GEO series narrative explicitly describes sampling normal and carcinoma-in-situ biopsies from **the same urinary bladder**. It lists 60 distinct assay accessions but provides **zero explicit patient, donor, or subject IDs** in its machine-readable sample characteristics. A GSM ID identifies a *measurement*, not an independent person. The source therefore proves that correlated specimens exist in the parent study but does **not** identify which, if any, pairs are among the processed `bladderEset` subset or what their cluster memberships are. Do not infer independence, sample labels, or donor-block assignments from array-title suffixes. None of the nominal real-data p-values in the following analyses is validated as a biological false-positive probability by these metadata; they remain **conditional scenario calculations** until original subject linkage is recovered.

The primary source compressed SHA-256 is locked as `1aca6001faa77d5dcb5c6aae454fded1f4f0e2300b7273604016ca97b6131bd9`. The standalone [GEO lineage receiving workflow](.github/workflows/geo-lineage-science.yml) verifies this digest, preserves every available public source field, tests for explicit donor keys, and refuses a changed source or lost paired-biopsy evidence. If the public source later exposes donor codes, that is a reason to reconstruct the sample-to-person map and rerun **donor-blocked** inference, not to silently assume the old row-level null was correct.

## Real measured bladderbatch experiment: identical model, different scientific contrasts

The main demo's expression measurements are synthetic. To independently exercise the **real original RMA-normalized microarray assays**, the separate GitHub Actions workflow [`real-bladderbatch-science.yml`](.github/workflows/real-bladderbatch-science.yml) downloads and SHA-256-verifies the public Bioconductor 3.23 `bladderbatch` 1.50.0 source package, installs Biobase and runs [`export-real-bladderbatch.R`](research/export-real-bladderbatch.R) and [`real-bladderbatch-receiving.mjs`](research/real-bladderbatch-receiving.mjs). It creates temporary CSVs on the isolated runner, then deletes them; no raw expression dataset is committed to git.

The **source revealed three phenotype groups** in `pData(bladderEset)$cancer`: 40 Cancer, 8 Normal, and 9 Biopsy. These are **not interchangeable**. We therefore freeze a 200-probe subset chosen with a label-blind R sampling seed `20261118` and run three **predeclared scientific contrasts**, each with the same fold-local top-20 feature selection, diagonal-LDA, fivefold/three-repeat statistic, and 100 batch-conditional shuffles (seed `20261030`, fixed stopping). The matrix has 22,283 original probe measurements; this is a **frozen illustrative subset**, not a genome-wide clinical test.

| Original measured-data contrast | Rows | Mixed batches | Naive within-fold AUC | Batch-only AUC | Fixed-budget conditional p | Batch transfer AUC | Batch-centered AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Cancer vs **Normal + Biopsy pooled** | 57 | 2/5 | 0.972 | 0.773 | 0.0099 | 1.000 (only 2 mixed heldouts) | 1.000 |
| Cancer vs **Normal only** | 48 | 1/4 | 0.933 | 0.910 | 0.0198 | **n/a** (1 mixed heldout) | **n/a** |
| Cancer vs **Biopsy only** | 49 | 1/4 | 0.996 | 0.908 | 0.0099 | **n/a** (1 mixed heldout) | **n/a** |

These are real computed outcomes from the pinned pipeline, not demonstration numbers. The contrast-specific shift is decisive: pooling two biologically distinct reference phenotypes permits two mixed-label processing batches, whereas choosing a single reference phenotype leaves **one** and thus cannot estimate a multi-batch transfer AUC. The default heuristic verdicts were respectively `signal`, `shortcut`, and `open`—these are **not biological truth labels**. In particular, the high same-cohort AUC does not verify sample independence, prospective diagnostic performance, genuine biology, or causal effects.

The `p` values here are **finite 100-shuffle Monte Carlo values** (not complete conditional-orbit enumeration) and are interpretable only if the unverified row-level exchangeability assumption holds. The package provides sample identifiers but no independently verified donor/subject grouping; different donors, biopsies, technical repeats and hidden sampling dependencies cannot be inferred from identifiers. Three inspected contrasts were evaluated on the **same public cohort** and are correlated; do not select the smallest p or use these numbers as a familywise-confirmed result. The held-out-batch AUC of 1.000 in the pooled contrast is an average of **only two** mixed heldouts; the three pure-label heldouts have undefined AUC and contribute no discrimination evidence.

### Six predeclared probe-panel seeds × three contrasts: all 18 outcomes received

The original one-panel result above is a **pilot**, not evidence that a different 200-probe subset would behave the same way. The current receiving workflow now checks that original pilot CSV's three SHA-256 hashes **without altering its inputs**, draws five new independent *probe-selection seeds* (`20261119..20261123`), and evaluates all three phenotype contrasts for all six seeds. Each panel uses 200 originally measured probes, with training-only top-20 discriminant selection under the same CV statistic.

A 100-shuffle family would have **no chance of a multiplicity-adjusted rejection**: the smallest possible first Holm p across 18 hypotheses would be `18/(100+1)=0.1782`, larger than 0.05 irrespective of any signal. The receiving code therefore enforces an attainable power floor, running a complete **499 conditional Monte Carlo shuffles per contrast** and applying the estate's shared `holmAdjust` from `src/familywise.mjs` once across all 18 p-values. Its minimum first possible Holm p is `18/500=0.036`; an insufficient budget fails with an actionable error instead of silently running an impossible test.

| Measured contrast, six 200-probe panels | Naive CV AUC range | Raw conditional p range | Holm-adjusted p range (18 correlated tests) | Mixed processing batches | Transfer AUC defined? |
| --- | ---: | ---: | ---: | ---: | --- |
| Cancer vs Normal + Biopsy | 0.972–0.985 | 0.002 | 0.036 | 2 | Yes, on just 2 mixed heldouts |
| Cancer vs Normal | 0.933–0.954 | 0.002–0.004 | 0.036 | 1 | **No** |
| Cancer vs Biopsy | 0.996–1.000 | 0.002 | 0.036 | 1 | **No** |

**Observed:** all 18 finite-budget comparisons have adjusted p `0.036` under the declared synthetic analysis assumptions; the observed AUC is not a quirk of the first sampled probe panel. **Not established:** that row samples are independent exchangeable donors, that a mixed Normal+Biopsy negative class is clinically coherent, that independent batches beyond the existing two replicate, or that the result predicts an external patient. Six *probe choices* on the **same cohort** do not create six biological replications. The Holm family was frozen **after viewing the one-panel pilot**, so this remains a retrospective exploratory robustness audit rather than prospective preregistration. Only under the unverified marginal null assumptions does Holm offer formal error control; the arithmetic correction cannot repair invalid exchangeability.

The workflow prints a machine-readable science-only digest (excludes non-reproducible elapsed milliseconds) of `2813d42ff984cdc2312ebf9fa55574532857edb4c06a3809cef1faf01754e821` on this execution. The source checksum and pilot CSV checksums are enforced on every run. A rerun yielding a different digest is treated as scientific drift, not glossed over.

Frozen source tar SHA-256: `431b6c2e2b751f706a897a2d6162e6735617975d8f3d783775418a35c6e9de74`. Temporary CSV hashes for the three contrasts (full, Normal-only, Biopsy-only) are `20e9f0f8778a747e7922287b0cd3c1e995384302967a7ef3e8caf5426e3dba8d`, `5ff767f3b5a842125f88f6b8362f48c999766c8ad8de889e73ef4dedc6234049`, and `ac93ba3ca8988370d5fff333250f68120547a0a966818eaf44ff840fc874093`; complete machine-readable result JSON is logged by the workflow. The underlying package is attributed to Jeffrey T. Leek ([Bioconductor `bladderbatch`](https://bioconductor.org/packages/release/data/experiment/html/bladderbatch.html), Artistic-2.0). The three-class processing design is independently documented by the [BatchQC package vignette](https://rdrr.io/bioc/BatchQC/f/vignettes/BatchQC_examples.Rmd).

## Nested family / site dependence: a second scientifically testable null

When different subjects share a household, pedigree, site, or other scientifically declared dependence cluster, merely adding `subject_id` is **not enough**: it stops technical-read replication leakage within a subject, but can still train on that subject's relatives and permute dependent families as if unrelated. An optional `cluster_id` after `subject_id` activates a separate two-level design: `sample_id,batch,label,subject_id,cluster_id,feature...`. A subject's recordings must share one binary subject-level outcome, one batch and one cluster; a cluster cannot cross batches. Empty or contradictory IDs fail at ingestion and before any inference.

**Changed estimand:** `nestedClusterFolds` holds out **complete related clusters** for every cross-validation repeat, then computes within-fold AUROC with each independent subject contributing one average held-out score. Exact and Monte Carlo randomization shuffle binary labels between **whole subjects within their declared cluster only**; no observation moves away from its subject and no label moves across potentially dependent clusters. The exact conditional space is the product of `choose(number of subjects in cluster, positive subjects in cluster)`, regardless of repeated-row count. If each cluster has only one label, the orbit contains only the observed allocation: the control explicitly returns **unavailable** instead of suggesting that a meaningless p=1 proves absence of a biological effect. A preregistered `/api/family` Holm family uses exactly these same cluster-conditional nulls. The intake SHA-256 fingerprint includes the cluster membership field.

**Frozen adversarial counterexample:** 20 households, 40 subjects and 80 recordings, with only household-fingerprint features, two positive/negative households per batch and **no transferable individual signal**. The accepted subject-only CV mixed members of all 20 households between train/test and claimed mean AUROC **0.9917**, uncorrected conditional p **0.005** (199 planned shuffles). Under the declared second-level clusters, heldout-family AUROC is **0.5000**, with exact conditional orbit size 1 and p **unavailable**, not a false discovery. A separate 20-family mixed-label positive-control has cluster-heldout AUROC **1.000** and two distinct pipeline models each retain raw p **0.005**, Holm-adjusted p **0.010**. An independently enumerated 32-allocation five-family null checks p-value ranking and finite-null Type-I control, including changed-input and cold HTTP/CSV/Markdown tests. Accepted one-level and three batch-science scenario outputs remain byte-stable.

**Scientific boundary:** This only models `subject` nested in `cluster` nested in `batch`. Within-cluster **subject exchangeability under the null** is an unverified design assumption; unrelated families must not be given the same cluster label, and family/site IDs alone do not establish biological independence or causal identification. Families spanning sites/batches, longitudinal outcome switching, arbitrary multilevel pedigrees, informative cluster-size selection and cross-site confounding require a separate scientifically specified randomization group. No assumed fallback silently guesses such a design. For the primary statistical justification see <https://pmc.ncbi.nlm.nih.gov/articles/PMC4644991/> (Winkler et al., *Multi-level block permutation*, NeuroImage 2015).


## Predeclared whole-cluster randomized trials (different from dependency blocks)

If **the entire cluster was assigned one treatment label by a known experimental randomization protocol**, the observational nested-cluster null above can be too restrictive: every subject in one family/site/cluster has the same label, so its within-cluster exchangeability orbit has size 1. NULLGARDEN now supports a separately registered treatment-*allocation* unit. Registration must be made when the dataset is first loaded, not after viewing results:

```json
{
  "csv": "sample_id,batch,label,subject_id,cluster_id,feature_1,...\n...",
  "randomization": {
    "unit": "cluster",
    "mechanism": "uniform_within_batch",
    "protocolId": "PREREGISTERED-TRIAL-ID"
  },
  "family": {"ks": [1, 2], "seed": 31, "perms": 127}
}
```

Pass this JSON to `POST /api/dataset`, then use the returned `datasetId` with `POST /api/run`, `POST /api/family` or `POST /api/report`. The web **Load CSV** control also exposes an opt-in **Registered cluster trial** checkbox and required protocol ID. The intake family digest covers both the original measurements and the asserted treatment-allocation protocol. Historical randomization is **not independently verified** by a text ID.

**What changes:** within each batch/randomization stratum, exchange labels between **entire clusters** rather than between subjects within clusters. Every subject and all its repeated measurements stay together; complete clusters remain disjoint across CV train/test; AUROC is still averaged per heldout subject and fold. The conditional exact assignment count is `product_strata choose(number_of_clusters, treated_clusters)`, independent of cluster size. Small orbits are exhaustively enumerated with ties counted; large orbits use the predeclared fixed-budget Monte Carlo p bound. The immutable Holm family reuses the same registered assignment scheme. The default (no `randomization` object) retains the observational within-cluster subject null **without change**.

**Independent locked oracle:** six strata with two clusters each provide **64** legal allocations even with multiple related subjects and repeated rows inside every cluster. A separate bitmask reference computes every legal label vector, CV statistic, exact p and the entire finite-null Type-I envelope; it never consults the candidate enumeration. Mixed treatment labels inside a cluster, missing cluster/subject IDs, unknown assignment mechanisms, and post-hoc family changes are rejected. This is consistent with the restricted/whole-block permutation distinction in Winkler et al. (2015), *Multi-level block permutation*, [doi:10.1016/j.neuroimage.2015.05.092](https://doi.org/10.1016/j.neuroimage.2015.05.092).

**Scope:** Validity depends on actual `uniform_within_batch` cluster assignment, conditional on the treated-cluster count in each stratum, under the sharp randomization null and no interference. A protocol ID cannot prove compliance. Arbitrary observational clusters, unequal allocation probabilities, nonuniform adaptive assignment, site-wide unmeasured interference, and restricted randomization beyond explicit strata are **not supported**. This pipeline tests whether a *predictive CV statistic* is unusual under a sharp null; it does not estimate an average treatment effect or justify a causal claim by itself.

## Exact batch-conditional randomization for tractable designs

NULLGARDEN now checks the size of the *complete* conditional label orbit before drawing random shuffles. With batch (b) containing (n_b) observations and (k_b) positive labels, there are exactly `product_b choose(n_b, k_b)` distinct allocations consistent with the observed within-batch positive counts. It uses bounded, overflow-safe integer combinatorics and a **lazy** joint assignment enumerator (no factorial-sized array, no duplicate arrangements of indistinguishable labels). If the orbit fits both a 512-assignment ceiling and a deterministic estimated-work budget of 25 million feature/sample/fold operations, it evaluates **every** allocation using the same three-repeat foldwise AUROC statistic and fixed CV seed, with the observed allocation included exactly once.

The exact conditional p-value is the proportion of all permitted allocations with a score **greater than or equal to** the observation—there is **no Monte Carlo +1 correction**, no shuffle-seed variability, no adaptive stopping ambiguity, and every tie counts against rejecting the null. The result is explicitly marked `nullSampling:'complete_batch_conditional_orbit'`, `stopping:'exact'`, `pExact:true`, `orbitSize`, and `pLower=pUpper=p`. If the orbit is too large, the real app uses the fixed-budget anytime-valid Monte Carlo mechanism below; the direct library uses full fixed-budget Monte Carlo by default. If some permutation has an undefined heldout-fold AUC, the exact diagnostic abstains rather than silently excluding that part of the null space.

**Independent receiving:** Six within-batch pairs each containing one positive and one negative provide (2^6=64) assignments. A separate 64-mask exhaustive reference computes the exact p without calling the production label enumerator, and the cold HTTP CSV journey must use the same exact result. Tied/constant features must yield p=1, and a large one-batch orbit must fall back to conditional Monte Carlo rather than allocating an exponential array. The guarantee remains **conditional on genuine within-batch label exchangeability** and a predeclared statistic/hypothesis; it is neither causal identification nor across-hypothesis error control. See [Hothorn et al., *Implementing a Class of Permutation Tests: The coin Package*](https://cran.rstudio.com/web/packages/coin/vignettes/Implementation.pdf) for the exact-vs-conditional-Monte-Carlo distinction.

## Certified anytime permutation decisions (the web app's default)

For designs whose complete exact orbit exceeds the work budget, the web/API run starts with a **predeclared** budget of 100 conditional label shuffles, a fixed model pipeline, a fixed random seed, and a fixed decision threshold α = 0.05. After `t` shuffles of which `Lₜ` meet or exceed the observed foldwise AUC, the unobserved full-budget Monte Carlo p-value is guaranteed, for **every possible remaining outcome**, to lie between

`p_lower = (Lₜ + 1) / (B + 1)` and `p_upper = (Lₜ + 1 + B - t) / (B + 1)`.

If `p_lower > 0.05`, no possible remaining permutation can make the declared fixed-budget test significant, so the app immediately stops. If `p_upper <= 0.05`, the test is already significant even under the least favorable possible remaining shuffles. Otherwise it continues. The reported `p` is always the **conservative upper bound**, an anytime-valid p-process for this fixed-B test; it is marked `pExact:false` unless all B shuffles were run. A stopped test does **not** publish an adaptively sampled null mean or histogram as if it were the full null, and a large bound is never evidence that the batch caused the model score. The JSON includes `permsBudget`, `permsRun`, `pLower`, `pUpper`, and `pExact`; the exported report says what was and wasn't observed.

The key guarantee is **pathwise**: every decision reached early is identical to the eventual fixed-budget decision, no matter what the unseen suffix contains. The tests exhaustively verify all 512 exceedance streams for B=9, compare all three scientific scenarios against full-budget runs, and check cold HTTP output. Choosing a different budget, rerunning with different seeds until favorable, or selecting different hypothesis controls afterward **does not inherit** this guarantee. Direct `runControls` calls still default to `stopping:'fixed'` for reproducible complete-null distribution studies, while the interactive/API path uses `stopping:'decision'` to save work when the decision is already settled.

This is the conservative **fixed-horizon anytime-valid** construction in Fischer and Ramdas, [*Sequential Monte Carlo testing by betting*](https://academic.oup.com/jrsssb/article/87/4/1200/8106328), JRSS B (2025). It does not implement their more sophisticated betting mixtures or anytime validity across post-hoc hypothesis selection, and it does not turn observational batch controls into causal identification.

## What the permutation p-value actually tests

The batch-conditional null fixes each batch's observed **independent-unit** label count and uniformly shuffles labels **inside** its batch, moving every repeated measurement with its subject when `subject_id` is supplied. Every shuffled dataset is processed by the **same deterministic three-repeat stratified-CV AUC statistic**, with the **same CV seed and folds generator** as the observed dataset. The finite Monte Carlo p-value is `(1 + number of shuffled statistics >= observed statistic) / (1 + shuffled datasets)`; the fixed budget is 100 shuffles by default. Previous revisions compared a three-repeat observed AUC to one-repeat shuffles with different fold seeds. Those numbers were not a valid exchangeable null for the same statistic and are **not** comparable to current p-values. The CI suite compares the core to an independent conditional-shuffle oracle over all three bundled scenarios and multiple frozen seeds.

The resulting p-value is valid **only under within-batch label exchangeability under the null**, with analysis choices (pipeline, seed, labels, controls and permutation budget) fixed before inspecting the result. It does **not** establish a causal effect or certify generalization, and it is **not anytime-valid**: repeatedly changing seeds, rerunning different tests, selecting favorable controls after looking, or stopping after a promising interim shuffle count can inflate false positives. A p-value above the threshold means *insufficient evidence against this conditional null*, **not** that a batch shortcut has been proved. Positive shortcut evidence must come from its separate batch-only or held-out-batch diagnostics.

For sequential or adaptively stopped Monte Carlo inference, a different anytime-valid method is required; see [Sequential Monte Carlo testing by betting](https://academic.oup.com/jrsssb/article/87/4/1200/8106328) (JRSS B, 2025). NULLGARDEN intentionally provides the simpler, fixed-budget conditional test and discloses its limits in the JSON response and exported Markdown.

**Foldwise AUC, not pooled scores:** Ordinary stratified cross-validation now computes AUROC **within each held-out fold** and averages those AUCs over folds and repeats. Each fold's predictions were produced by a different fitted model, so ranking their raw scores together is not an invariant statistic. The old pooled out-of-fold AUROC can change arbitrarily when separate fold models have different intercepts, even when each model perfectly distinguishes its own held-out samples. Our locked negative control reproduces the problem with offset-only score transformations. If a requested held-out fold is pure-label, its AUROC cannot be computed and the aggregate abstains. This is a change of **estimand**, not just a numerical stabilization, and prior reported AUCs are not comparable without rerunning the analysis. The matched permutation test uses the same revised foldwise statistic for observed and shuffled data.

**Non-leaking evaluation:** The mixed-batch centering control now learns every batch's feature means from each **training fold only**; feature selection and classifier fitting also remain training-only. When a test batch has no training members, the control abstains instead of estimating its mean from held-out features. The previous version estimated batch means globally and leaked test covariates; the correction measurably changes the same 57-sample control AUCs.

**Batch-transfer evaluation:** Leave-one-batch-out fits a separate classifier for each held-out batch. Because those independently fitted scores have unrelated offsets/scales, the displayed AUROC is now the **mean of per-held-out-batch AUROCs**, counting only batches with both labels; all-pure or single-mixed-batch designs are marked inconclusive. Previously all out-of-fold scores were pooled into one ranking across separately fitted classifiers, which can manufacture apparent discrimination from between-batch differences. The original pooled score is retained in the JSON as `pooledOofAuc` **for diagnostic comparison only**, never as the finding or verdict. Each held-out group's AUC and the counts of excluded pure-label batches are reported.

These are evaluation corrections, not a claim that batch effects are causally removed or that the resulting AUC proves biological generalization. Small samples, class imbalance and non-stationary batches remain substantive uncertainty.

## Numerically stable scientific receiving

The classifier now learns a **maximum-absolute feature scale from each training fold only**, and fits dimensionless class means, pooled variance, and diagonal-LDA weights. This replaces raw squared deviations (which overflow for finite inputs near `1e155`) and an absolute `1e-9` variance floor (which changed feature selection when a scientist simply changed units). Batch centering also learns normalized batch sums **only from training observations**, reconstructing finite means before held-out subtraction. Held-out measurements never alter the fitted scales, means or weights.

The receiving suite includes separate 280–310-order feature-unit changes, finite extreme observations, an unrepresentable heldout/train ratio, constant features, numeric non-finiteness, and heldout covariate perturbations. For supported finite ranges, AUC and independent predictions remain invariant under per-feature unit conversions; when a finite input cannot be represented after normalized scaling, the engine raises an explicit `NUMERIC_RANGE_UNSUPPORTED` range error rather than publishing `Infinity` or `NaN` as evidence. The simple LDA model and small-sample/causal interpretation limits remain unchanged.

This is numerical robustness for the stated fixed pipeline, **not** arbitrary-precision arithmetic, an adversarial-repository execution sandbox, or proof that other classifiers satisfy the same controls.

## Class-support and testability gates

A cross-validation fold with no positive or no negative **training** samples cannot identify this binary classifier, even when overall uploaded CSV labels include both classes. Earlier versions silently substituted all-zero predictions from such a fold and could display an invented AUC or permutation significance. Every repeated stratified fold now checks both training classes before scoring; a missing class produces **n/a**, not chance performance or a fabricated p-value. The batch-only comparison and batch-preserving permutation null also abstain if their baseline is not identified. Leave-one-batch-out explicitly counts batches whose held-out class leaves a one-class training set, alongside test batches with only one class (whose AUROC is mathematically undefined).

This behavior reaches the live API, exported Markdown, deterministic verdict and optional Nemotron explanation: missing evidence is labelled *unavailable*, no absent baseline reference line is drawn, and the model is not asked to interpret null as zero. A scientific design with only one minority-label sample should collect additional samples of both labels before evaluating discrimination. In particular, a valid mixed-label dataset does **not** imply every cross-validation or leave-batch-out training fold is valid.

## Methodological references

- [Data leakage inflates prediction performance in connectome-based machine learning models](https://www.nature.com/articles/s41467-024-46150-w) (Nature Communications, 2024): batch/site correction must be fitted within each cross-validation training split, rather than using held-out covariates.
- [Tournament leave-pair-out cross-validation for receiver operating characteristic analysis](https://pmc.ncbi.nlm.nih.gov/articles/PMC6745617/) (Statistical Methods in Medical Research, 2019): pooled out-of-fold AUC compares scores produced by distinct fitted prediction functions and can be biased; within-fold averaging estimates a different, explicitly defined estimand.

We use these results to choose transparent, non-leaking diagnostics; neither method constitutes a universal statistical guarantee of predictive performance.

## How it uses NVIDIA Nemotron on Nebius Token Factory

- **Endpoint and model:** `https://api.tokenfactory.nebius.com/v1/chat/completions` (OpenAI-compatible), model `nvidia/nemotron-3-super-120b-a12b` (override with `NULLGARDEN_MODEL`). The call is made server-side in `src/llm.mjs`; the key is read from `NEBIUS_API_KEY` and never sent to the browser.
- **Why this size:** the task is short structured reasoning over a small batch table. The 120B-A12B mixture-of-experts model follows the JSON format reliably and answers in about 3 to 10 seconds; a larger model would add latency without changing the checked output.
- **Propose hypotheses:** Nemotron reads the batch/label table and your note and names up to four shortcut hypotheses, each mapped to one control from the menu. The server drops any item whose control id is not on the menu. The app then runs the controls itself and marks each hypothesis supported or not supported from the computed AUCs. The model proposes; the controls decide.
- **Explain results:** Nemotron writes a short plain-language read-out of the computed table. Every decimal in its text is compared with the computed figures (tolerance 0.006) and the UI shows "N of M match", flagging any number the controls never produced.

## Honest limits

- The validity of any p-value still depends on an exchangeable null at the correct subject level; a subject ID alone does not identify independent participants or validate treatment assignment.
- The bundled datasets use the real bladderbatch batch/label counts (57 samples, 5 batches) but the expression values are synthetic.
- One fixed classifier, not your pipeline. The controls demonstrate the method; they are not a certification that a result is valid.
- Verdicts come from simple AUC thresholds on small samples and are heuristics. Batch-transfer macro AUC requires at least two evaluable mixed-label heldouts; even then, few batches imply high uncertainty.
- Nemotron output varies between calls; the figure check catches wrong numbers, not wrong reasoning.
- Experimental, not clinically validated.

## Repository components

The `components/` directory holds earlier standalone Python modules (batch-label schedules, batch-confounding diagnostic) contributed separately; see their READMEs. Provenance notes are in `docs/source-recovery.json`. The web app in `server.mjs`, `src/` and `public/` does not depend on them.

## License

MIT, see LICENSE.
