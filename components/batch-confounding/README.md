# Batch-confounding diagnostic

A standalone descriptive consumer of NULLGARDEN datasets and batch-label schedules. It reports batch composition, available within-batch label comparisons, exact design rank and an in-sample batch-prevalence AUC. With a schedule, it checks the expected invariance of that fixed score over the supplied draws.

This component performs **no model fitting**. Primary ownership of NULLGARDEN's engine, server, UI and fitted-model work remains with **estate-557346a0/root**. This contribution is tracked as `nullgarden-batch-confounding-diagnostic-c70eecd4`; the [original primary claim](https://github.com/Jacob-Met/hamon/issues/1631#issuecomment-6431474552) is preserved.

## Requirements and packaged demonstration

Qualified runtime: **Linux, Python 3.12.3, NumPy 2.5.4**. Use an interpreter with NumPy installed; the sibling [requirements-tested.txt](../batch-label-schedules/requirements-tested.txt) records the dependency. Other platforms and versions are unqualified. These commands do not download data or dependencies.

Keep both sibling component directories. From `components/batch-confounding`:

```sh
python3 -c "import sys,numpy; print(sys.version); print(numpy.__version__)"
python3 ../batch-label-schedules/demo_dataset.py --out demo-data
python3 ../batch-label-schedules/batch_schedule_cli.py generate --dataset-json demo-data/dataset.json --matrix-npz demo-data/dataset.npz --seed 20261029 --draws 32 --out demo-schedule.json
python3 batch_confounding_cli.py --dataset-json demo-data/dataset.json --matrix-npz demo-data/dataset.npz --schedule demo-schedule.json --out diagnostic-output
PYTHONPATH=../batch-label-schedules python3 independent_confounding_receiver.py
PYTHONPATH=../batch-label-schedules python3 independent_dataset_identity_receiver.py
```

Open `diagnostic-output/REPORT.md`. The synthetic six-sample fixture has descriptive AUC **13/18**, six conditional label assignments, one frozen label and maximum Hamming change four. Those are mechanical fixture properties, not scientific evidence.

The dataset and schedule creators require fresh output paths. The diagnostic output directory must be new and its parent must already exist. Existing files/directories are never overwritten. Computation finishes before output creation; an I/O failure after creation can leave a partial directory, which is retained for inspection.

To omit schedule checks, omit `--schedule` and choose a different output directory. The report then explicitly says no schedule result was supplied.

## Input and CLI contract

Use metadata JSON plus an NPZ containing exactly one array named `X`. The shared schedule loader verifies the retained NPZ SHA-256 before NumPy loading and checks the declared matrix hash when supplied. Labels `y`, batch `groups`, unique `sample_ids` and unique `feature_names` must align with the finite real matrix. Numeric binary labels are required; Boolean labels are rejected. See the [schedule README](../batch-label-schedules/README.md) for the complete matrix, identity and resource contract.

| Option | Meaning |
|---|---|
| `--dataset-json` | Required metadata JSON |
| `--matrix-npz` | Required corresponding NPZ |
| `--out` | Required new output directory |
| `--schedule` | Optional existing schedule artifact |
| `--outcome-path` | Optional nested-object path such as `source_metadata.outcome` |
| `--structural-only` | Explicitly omit RNG replay; requires a schedule |
| `--max-schedule-cells` | Positive cell budget, default 2,000,000 |

Outcome paths address nested object keys separated by dots; there is no syntax for a literal key containing a dot. Empty path parts, missing fields, a null final field and non-array final fields are rejected. The array must align with samples and contain either null or nonempty strings without surrounding whitespace. To omit outcomes, omit the option. An aligned all-null array is distinct from absent outcome metadata.

With a schedule, strict RNG replay is the default. Structural validity, seeded replay and fixed-score invariance are separate results. The report displays replay status explicitly, including **Unverified** for structural-only validation.

Input snapshots are bounded to 8 MiB of metadata, 512 MiB of NPZ and 256 MiB of schedule JSON. The shared loader also checks the uncompressed matrix header and working-input allocation before loading. Generated report content is bounded to 64 MiB. The schedule cell budget counts `2*n_samples*(draws+1)`, including the baseline; these individual limits are not a total peak-memory guarantee.

## Read the results

### Population and available comparisons

Every input row is included. A mixed batch contains both labels; a pure batch contains one. Positive-negative pair counts describe observed comparisons, not effective sample size or independent observations. Mixed batches do not establish adequate power, exchangeability or a causal label effect.

### Exact design rank

The design contains one indicator column per observed batch and no extra intercept. Its rank equals the number of observed batches. Adding the binary label increases rank by one precisely when at least one batch contains both labels.

When all batches are pure, the label is fully represented by the batch indicators and no within-batch label variation is observed. A rank increase rules out that complete algebraic aliasing; it does not resolve other confounding or establish a reliable effect.

### In-sample batch-prevalence AUC

Each sample's score is the observed positive fraction in its own batch. The same labels construct and evaluate the score. Expression values are validated and bound to the input but do not enter this descriptive score.

AUC is `(wins + ties/2) / positive-negative pairs`. The implementation uses exact rational score comparisons and reports wins, ties, losses, the reduced AUC fraction and a decimal. It aggregates equal scores in O(B log B) time for B batches, without constructing a sample-pair matrix.

With both global classes present and every batch pure, this score has AUC 1. With equal prevalence in all batches, all scores tie and AUC is 0.5. Neither result estimates held-out predictive performance. With a single global class, AUC is undefined and represented by null plus an explicit status.

### Schedule invariance

Within-batch label shuffles preserve each batch's class counts. Therefore they preserve this fixed batch score's positive/negative score distribution and AUC. The component validates the supplied schedule and recalculates exact pair metrics for each draw.

An unchanged AUC demonstrates this expected control behavior. It does not show that leakage is absent or that batch shortcuts disappeared. Zero supplied draws are marked **no_draws** with zero AUC recomputations. Single-class inputs are **not_applicable_single_class**, also with zero AUC recomputations. The schedule is still validated before either status returns. Pure-batch label assignments cannot change even if index permutations do.

### Recorded outcomes and subjects

Optional outcomes are descriptive categorical metadata and never replace `y`. Missing values are displayed as **Missing (null)**, separately from quoted literal strings such as "missing" or "Missing (null)". JSON preserves the exact values; Markdown escapes structural characters and control marks.

The recovered bladderbatch adaptation maps Cancer to 1 and Normal/Biopsy to 0. Its original outcome field retains five categories: Normal (8), sTCC+CIS (12), sTCC-CIS (16), mTCC (12), and Biopsy (9). Use `--outcome-path source_metadata.outcome` to include that cross-table; no medical interpretation is assigned to these category names.

Subject IDs, if supplied, expose repeated IDs, represented batches and conflicting labels. Rows are not silently removed or deduplicated. Missing or unique recorded IDs do not establish biological independence.

## Module API

`describe_batch_confounding(dataset, outcomes=None, max_schedule_cells=2000000)` returns a JSON-compatible report. `check_schedule_invariance(dataset, report, schedule, outcomes=None, strict_replay=True, max_schedule_cells=2000000)` regenerates and strictly compares the full report, validates the schedule, and checks the fixed-score metrics. These parameters after `dataset` (and after the three positional checker arguments) are keyword-only.

Direct imports require both component directories on `sys.path`; the CLI locates its sibling automatically. Input validation and dataset binding use the maintained schedule component's
public `validate_dataset_identity` API. It returns `binding` and
`ordered_metadata`, applies the same dataset rules and zero-draw baseline cell
budget (`2*n_samples`), and creates no temporary schedule. The diagnostic does
not duplicate dataset validation rules or fit a nuisance predictor.

This separates descriptive reporting from serialization of a schedule's exact
combinatorial support. Schedule generation remains subject to the existing
runtime integer-to-decimal limit: a balanced single batch of 20,000 samples
exceeds Python 3.12's default 4,300-digit guard, even with zero draws. Supplying a schedule
still requires a valid schedule artifact; this change does not alter schedule
v1, bypass its limits or modify the process-wide guard.

## Output bundle and provenance

| File | Contents |
|---|---|
| `REPORT.md` | Readable counts, definitions and limitations |
| `report.json` | Exact diagnostic results and ordered outcome metadata |
| `INPUTS.json` | Captured input-file hashes, source-file hashes and runtime |
| `MANIFEST.json` | Size and SHA-256 of the other output files |
| `invariance.json` | Schedule validation and invariance result, if supplied |

Input files are not copied into the output bundle. Retain them separately for reproduction. Hashes cover the bounded file snapshots actually parsed. Metadata, NPZ and schedule are captured separately; this is not an atomic snapshot of the original file set. The loader checks captured input compatibility. A controlled receiving experiment changed the original metadata after capture and verified that the report still described the captured bytes.

The schedule file SHA-256 and the schedule's internal integrity digest identify different representations. Both are retained in their respective fields. Hashes provide consistency checks, not authenticated authorship.

## Actual receiving evidence

### Dataset-identity regression receiving

The dataset-identity patch was executed on Linux, Python 3.12.3 and NumPy 2.5.4
with the default 4,300-digit integer guard unchanged. The existing 64 schedule
checks and 21 diagnostic checks passed, together with four new named receiver
checks. The new receiver covers 10,000- and 20,000-row balanced batches,
independently reconstructed input identities, 14 malformed datasets, six
rejected budgets, an exact baseline budget and detached returned metadata.

An actual CLI comparison reproduced the old 20,000-row failure and received
the corrected report: AUC 1/2, 100,000,000 tied positive-negative pairs and
batch/target ranks 1 and 2. Invalid input created no output; an existing output
directory and original input files remained unchanged. The small demonstration's
32-draw schedule, report and invariance JSON were byte-identical to the previous
implementation, including strict RNG replay. Large-support schedule generation
still reached its documented serialization limit; it was not relabeled a pass.

The new receiver is a source-level regression entry point. Any release-download
receiving is attributed separately to the actual published commit and asset.

### Earlier component receiving

The independent receiver passed **21 named checks**, including **150 exhaustive index permutations across nine small fixtures**. Its row-pair Fraction oracle and NumPy design ranks are independent of the component's grouped score and exact rank formulas. It covers equal-score ties, unequal group sizes, pure/mixed/one-class cases, missing outcomes, repeated subjects, malformed report fields and changed input/schedule bindings.

The CLI receiving course passed **20 recorded steps**: 15 diagnostic CLI process checks, four fixture-preparation steps, and one isolated snapshot-interception check. Existing outputs were preserved; missing/null/misaligned outcome paths and malformed verified NPZ were rejected before output creation. Explicit structural-only, zero-draw and single-class paths remained distinguishable. Metadata containing pipes, newlines, HTML-like strings and direction controls retained valid report-table structure and exact JSON values.

The actual public-data CLI run matched a predeclared independent arithmetic reference:

| Quantity | Received value |
|---|---:|
| Samples; positive / negative | 57; 40 / 17 |
| Mixed batches; mixed samples | 2 and 5; 37 |
| Mixed positive / negative | 29 / 8 |
| Pure-batch samples retained | 20 |
| Within-batch positive-negative pairs | 116 |
| Batch rank; batch-plus-target rank | 5; 6 |
| AUC wins / ties / losses | 508 / 116 / 56 |
| AUC pair count | 680 |
| Exact descriptive AUC | 283/340 |
| Supplied draws checked | 100 |

All 100 existing draws preserved the fixed-score AUC, with strict RNG replay verified. All five original outcome categories and every sample remained present; subject IDs were absent. Batch IDs retain first-occurrence order.

The exact [public-data example report](examples/bladderbatch/REPORT.md), JSON, input record and manifest are included. Report JSON SHA-256 is `c2aa6897938f4d321e62ffec7443e91b33663921ab2c91e6f30848c357621b53`; Markdown SHA-256 is `55953431ba830a26877c6ac73dbc289afb43c4482695a9f0fa78c3e4af7ea2e1`.

The source was Bioconductor bladderbatch 1.48.0, an already RMA-preprocessed matrix. The declared binary classification is this project's adaptation; it does not reproduce an original classification analysis or raw-CEL preprocessing. Exact source provenance and the retained original-work boundary are in [source-recovery.json](../../docs/source-recovery.json).

These are source/CLI receiving results. Verification of a published release download is attributed separately to its actual commit, asset and extracted entry point. No complete NULLGARDEN application, primary-engine adoption, NVIDIA/Nebius inference, other platform or scientific validity is claimed.

For downstream schedule-based evaluation, rerun the complete label-dependent preprocessing, feature selection, tuning, fitting and evaluation pipeline for every draw. Statistic choice, split design and inferential interpretation remain separate scientific decisions.
