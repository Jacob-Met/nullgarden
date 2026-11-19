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

Open http://127.0.0.1:4412. Pick a dataset tab, press **Run controls**, read the verdict and the AUC ladder. With a key set, **Propose hypotheses** and **Explain results with Nemotron** are live. Without a key those two buttons show a plain message and everything else still works. `PORT` changes the port. A CSV can be loaded with the "Load CSV" button (columns: `sample_id,batch,label,feature...`; label 0 or 1; at least 10 samples).

## The controls

A fixed pipeline (top-20 feature selection by |t| inside each training fold, then diagonal LDA) is scored by cross-validation, then rerun under four controls: batch-only baseline, within-batch label permutation (with a shuffled-label null band), leave-one-batch-out, and batch-centered features. The rules in `src/core.mjs` turn the ladder of AUCs into a verdict. Nothing in that path calls a model.

## What the permutation p-value actually tests

The batch-conditional null fixes each batch's observed label count and uniformly shuffles labels **inside** its batch. Every shuffled dataset is processed by the **same deterministic three-repeat stratified-CV AUC statistic**, with the **same CV seed and folds generator** as the observed dataset. The finite Monte Carlo p-value is `(1 + number of shuffled statistics >= observed statistic) / (1 + shuffled datasets)`; the fixed budget is 100 shuffles by default. Previous revisions compared a three-repeat observed AUC to one-repeat shuffles with different fold seeds. Those numbers were not a valid exchangeable null for the same statistic and are **not** comparable to current p-values. The CI suite compares the core to an independent conditional-shuffle oracle over all three bundled scenarios and multiple frozen seeds.

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

- The bundled datasets use the real bladderbatch batch/label counts (57 samples, 5 batches) but the expression values are synthetic.
- One fixed classifier, not your pipeline. The controls demonstrate the method; they are not a certification that a result is valid.
- Verdicts come from simple AUC thresholds on small samples and are heuristics. Batch-transfer macro AUC requires at least two evaluable mixed-label heldouts; even then, few batches imply high uncertainty.
- Nemotron output varies between calls; the figure check catches wrong numbers, not wrong reasoning.
- Experimental, not clinically validated.

## Repository components

The `components/` directory holds earlier standalone Python modules (batch-label schedules, batch-confounding diagnostic) contributed separately; see their READMEs. Provenance notes are in `docs/source-recovery.json`. The web app in `server.mjs`, `src/` and `public/` does not depend on them.

## License

MIT, see LICENSE.
