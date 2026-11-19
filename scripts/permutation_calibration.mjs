#!/usr/bin/env node
/* Frozen paired calibration gate for within-batch permutation p-values.
 *
 * Synthetic DATA are label-independent conditional on real bladderbatch batch
 * counts (57 samples, 5 groups); these are not patient samples or clinical
 * estimates. The candidate may not read a hidden generator outcome while
 * selecting a strategy; seed ranges and support envelopes are fixed here.
 *
 * Usage:
 *   node scripts/permutation_calibration.mjs /path/to/source_checkout
 *
 * A null family and an added within-batch-signal family are separately scored.
 * Success against these tolerances is a falsifying control, NOT proof that
 * every distribution/exchangeability model is calibrated.
 */
import { resolve, join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { performance } from 'node:perf_hooks';

const root = resolve(process.argv[2] || '.');
const { runControls } = await import(pathToFileURL(join(root,'src/core.mjs')).href);
const { build, REAL } = await import(pathToFileURL(join(root,'src/data.mjs')).href);

const SEED_START_NULL = 13000;
const NULL_CASES = 64;
const SEED_START_SIGNAL = 23000;
const SIGNAL_CASES = 16;
const CV_SEED = 20261030;
const PERMUTATIONS = 40;
const SELECTED_FEATURES = 20;
const ALPHA = 0.05;
const start = performance.now();

function distribution(name, seedStart, count, strength) {
  const pValues = [];
  const snapshots = [];
  for (let i = 0; i < count; i++) {
    const seed = seedStart + i;
    const ds = build(REAL, strength, seed, 200);
    const output = runControls(ds, ['within_batch_permutation'], {
      seed: CV_SEED, perms: PERMUTATIONS, k: SELECTED_FEATURES,
    });
    const row = output.results.within_batch_permutation;
    if (!Number.isFinite(row.p) || !Number.isFinite(output.results.naive.auc)) {
      throw new Error('Nonfinite classifier inference for ' + name + ' seed=' + seed);
    }
    if (row.statistic) {
      const expected = { seed: CV_SEED, folds:5, repeats:3 };
      if (JSON.stringify(row.statistic) !== JSON.stringify(expected)) {
        throw new Error('Observed/null estimator signature drift for ' + name + ' seed=' + seed);
      }
    }
    pValues.push(row.p);
    snapshots.push({ seed, p: row.p, observedAuc: output.results.naive.auc,
      nullAuc: row.nullMean });
  }
  const hits = pValues.filter((p) => p <= ALPHA).length;
  return { name, seedStart, count, hits, rate: hits / count,
    medianP: pValues.slice().sort((a,b)=>a-b)[Math.floor(count/2)],
    meanP: pValues.reduce((x,y)=>x+y,0)/count,
    snapshots };
}
const nullFamily = distribution('conditional_label_null',SEED_START_NULL,NULL_CASES,0);
const signalFamily = distribution('within_batch_signal',SEED_START_SIGNAL,SIGNAL_CASES,1.1);
const result = {
  schema: 'hamon.nullgarden.permutation-calibration.v1',
  source: root, settings: { cvSeed:CV_SEED,permutations:PERMUTATIONS,
    selectedFeatures:SELECTED_FEATURES, alpha:ALPHA,
    nullMaxFalsePositives:10, signalMinDetections:8 },
  nullFamily, signalFamily, elapsed_ms: Math.round(performance.now()-start),
};
console.log(JSON.stringify(result,null,2));
if (nullFamily.hits > 10) {
  process.stderr.write('Conditional-null error exceeds preregistered 10/64 gate\n');
  process.exitCode = 2;
}
if (signalFamily.hits < 8) {
  process.stderr.write('Within-batch signal falls below preregistered 8/16 gate\n');
  process.exitCode = 3;
}
