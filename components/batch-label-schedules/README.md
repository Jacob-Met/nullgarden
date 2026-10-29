# NULLGARDEN batch-label schedules

Experimental, runnable Python component for generating and validating reproducible binary-label randomization schedules within declared batches.

This component does not fit a model or implement NULLGARDEN's complete application. The original primary engine/server/UI owner remains `estate-557346a0/root`. This contribution is tracked as `nullgarden-batch-label-schedules-c70eecd4`; recovery of the original work is tracked separately as `nullgarden-existing-project-reconciliation-20261029`. Original project claim: https://github.com/Jacob-Met/hamon/issues/1631#issuecomment-6431474552.

## Run the packaged example

The receiving environment was Linux, Python 3.12.3 and NumPy 2.5.4. Use a Python interpreter with NumPy already installed; `requirements-tested.txt` records the tested dependency. Other platforms or versions have not been qualified. The commands below run from this component directory and do not download data or dependencies.

```sh
python3 -c "import sys,numpy; print(sys.version); print(numpy.__version__)"
python3 demo_dataset.py --out demo-data
python3 batch_schedule_cli.py generate --dataset-json demo-data/dataset.json --matrix-npz demo-data/dataset.npz --seed 20261029 --draws 32 --out demo-schedule.json
python3 batch_schedule_cli.py validate --dataset-json demo-data/dataset.json --matrix-npz demo-data/dataset.npz --schedule demo-schedule.json
python3 independent_schedule_receiver.py
```

Use fresh output paths. The creator refuses an existing directory and the CLI refuses to overwrite a schedule. If creation is interrupted, inspect the partial directory before selecting a fresh one. The six-sample example has six possible label assignments, one frozen label and maximum Hamming change four. These are mechanical fixture properties, not biological evidence.

## Use an existing dataset

The CLI accepts JSON metadata and an NPZ containing exactly one array named `X`. Metadata must supply `provenance.npz_sha256` or `storage.npz_sha256`; if both are supplied both must match. Existing canonical `matrix_sha256` values are also checked.

`X` must be a nonempty finite real matrix with samples in rows. The metadata keys are `y` (numeric 0/1 labels), `groups`, `sample_ids`, `feature_names`, and optional `subject_ids`. IDs are nonempty strings without surrounding whitespace; sample and feature IDs are unique. Boolean labels are rejected. Repeated subject IDs are reported and do not establish a valid permutation design.

The module accepts floating types up to 64 bits and integer values in [-2**53, 2**53], with a lossless round-trip check. Canonical matrix bytes are little-endian float64, C order. Matrix shape is bound separately. Complex, Boolean, object, nonfinite and wider-float matrices are rejected.

## Interpretation and reproducibility

The schema is `nullgarden.batch-label-schedule.v1`. `binding` covers the matrix and ordered metadata; every assignment includes the dataset binding. `baseline` is the observed identity assignment and is excluded from the random-draw summary.

Each draw maps destination sample position to the source position supplying its label:

```python
draw["labels"] == y[draw["index_permutation"]].tolist()
```

Keep matrix rows and sample identities fixed. Reordering both X and y together would implement a different operation.

Draws independently permute indices within batches using PCG64 and NumPy permutation, in first-occurrence batch order. Repeated and unchanged assignments are retained. Multiple index permutations can yield the same labels. Diagnostics report exact assignment support, pure batches, frozen labels and maximum possible Hamming change. Support one is labeled `uninformative_under_conditioning`.

Default CLI validation performs strict RNG replay: recorded NumPy version, seed, initial state, every index permutation and label vector, and final state. The explicit `--structural-only` option omits replay and reports it as unverified. Unsupported algorithms, wrong inputs, malformed schedules and changed semantic metadata are rejected.

The SHA-256 integrity field is unkeyed. It detects changes against a retained digest, but does not authenticate authorship. Recomputing a digest does not make a structurally invalid schedule valid. A fully regenerated internally consistent schedule is a different valid artifact.

Seeds are non-Boolean integers from 0 through 2**128-1. At most 100000 draws are accepted. The default configurable cell budget is 2000000, counting `2*n_samples*(draws+1)` for permutation and label entries, including the baseline. This is not a total-memory bound. The CLI additionally bounds metadata, NPZ allocation and serialized schedules; see its constants for limits.

## Dataset identity without a schedule

`validate_dataset_identity(dataset, *, max_schedule_cells=2000000)` validates the
same dataset rules as schedule creation and returns a fresh dictionary containing
`binding` and `ordered_metadata`. These preserve the canonical matrix and ordered
metadata identity used by schedules; returned dictionaries do not retain input
list or matrix-storage references. The function applies the zero-draw baseline
cell budget, `2*n_samples`, so the supplied budget must still admit the dataset.
It creates no schedule and makes no scientific-validity claim.

This boundary lets descriptive consumers validate and bind a dataset without
serializing a temporary schedule's exact combinatorial support. Schedule v1
schema and outputs remain unchanged. Large exact support values can still exceed
the runtime's integer-to-decimal serialization limit: a single balanced batch of
20,000 samples triggers Python 3.12's 4,300-digit guard when constructing a
schedule, including a zero-draw schedule. Dataset identity validation does not
remove that separate schedule limitation or change the process-wide guard.

## Scientific conditions

Within-batch exchangeability and subject independence remain unverified. This component does not permute subject blocks or handle temporal dependence. Missing subject metadata does not establish independence.

For every schedule, rerun the same complete label-dependent pipeline used for the baseline, including feature selection, tuning, fitting and evaluation. Holding a selection fitted on observed labels fixed changes the intended control.

Conditioning on batch label counts cannot test between-batch confounding. Pure batches contain no within-batch target contrast; mixed batches may contain contrast without resolving other confounding. A valid schedule establishes computational consistency, not predictive performance, causal identification or permission for statistical or medical claims.

## Actual receiving evidence

The corrected module passed the unchanged independent 64-case receiver, including exact enumeration, replay, malformed inputs and rehashed semantic tampering. The initial module failed three Boolean-versus-numeric metadata cases; that module and the failures are retained in the estate receiving archive.

Eleven CLI cases passed, including verified public-data generation and strict replay, module/CLI cross-validation, duplicate/nonfinite JSON rejection, changed-seed behavior, source-hash refusal and preservation of an existing output. Same-seed generated schedules were byte-identical before and after the validation/grouping correction.

The public bladderbatch conversion was independently checked against all 57 source samples and 22283 features. Its within-batch assignment support is 11860560, with 20 frozen labels and at most 16 changes. The public dataset is not bundled here. Receiving used an already RMA-preprocessed source and a binary classification adaptation; it does not reproduce an original classification pipeline or establish clinical validity.

Published-package receiving must be attributed separately to its actual source commit, asset hash and downloaded extraction. Source test results alone do not establish downloaded-package execution.
