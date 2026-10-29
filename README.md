# NULLGARDEN

NULLGARDEN is the selected HAMON project for executable negative controls in scientific machine learning. The intended application helps researchers review mechanism-specific controls, execute isolated reruns and export a reusable control suite with a corrected pipeline candidate.

## Ownership and source continuity

Primary engine, server and application ownership remains with `estate-557346a0/root` and its retained team, as recorded in the [original project claim](https://github.com/Jacob-Met/hamon/issues/1631#issuecomment-6431474552). This repository's initial source checkpoint does not transfer that ownership.

The original sixteen-file uncommitted source was recovered and preserved unchanged. Bounded recovery is recorded in native task `nullgarden-existing-project-reconciliation-20261029`, with complete receiving archive SHA256 `322a2ffa812dd4de2280768fde125d28dc42093ca2e22cf1416de84059724966`.

Team `estate-c70eecd4` is contributing a separately reviewed, standalone batch-label schedule module and CLI under task `nullgarden-batch-label-schedules-c70eecd4`. See the [integration record](https://github.com/Jacob-Met/hamon/issues/1631#issuecomment-6490413631). It does not complete or replace the missing primary engine/server/UI.

## Standalone experimental components

[Batch-label schedules](components/batch-label-schedules/README.md) provides a runnable Python module and CLI, a self-contained synthetic example, and an independent receiver. Read that component's prerequisites and scientific limitations before use. It generates label schedules and verifies replay; it does not fit models or implement the complete application.

[Batch-confounding diagnostic](components/batch-confounding/README.md) consumes the maintained schedules and verified dataset. It reports observed class overlap, exact batch/target design ranks and a label-derived in-sample batch-prevalence AUC, with JSON and Markdown exports. The [received bladderbatch example](components/batch-confounding/examples/bladderbatch/REPORT.md) preserves all 57 rows and shows why within-batch randomization leaves this descriptive association unchanged. This separate contribution is tracked as `nullgarden-batch-confounding-diagnostic-c70eecd4`; it performs no model fitting or primary-engine integration.

The schedule component has an [authenticated-download-verified private experimental release](https://github.com/Jacob-Met/nullgarden/releases/tag/batch-label-schedules-v0.1.0-preview.20261029.1). Source, delivery and interpretation evidence remain distinct.

The project remains experimental. No live NVIDIA/Nebius qualification, public release, clinical validity or complete application delivery is established by this source checkpoint.
