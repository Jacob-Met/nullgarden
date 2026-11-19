import test from 'node:test';
import assert from 'node:assert/strict';
import { repeatedCv, runControls, toMarkdown } from '../src/core.mjs';
import { SCENARIOS } from '../src/data.mjs';

// Locked failure: scoring functions fitted on different heldout folds can
// shift by arbitrary fold-specific intercepts. Between-fold ranks are not
// comparably calibrated. Only within-fold label pairs are legitimate here.
const ds = {
  ids: Array.from({length: 12}, (_, i) => 'id-' + i),
  y: Array.from({length: 12}, (_, i) => i % 2),
  X: Array.from({length: 12}, (_, i) => [i]),
  batch: Array.from({length: 12}, () => 'one'),
};

test('cross-validation AUC is invariant under arbitrary heldout-fold offsets', () => {
  const options = {seed: 1327, folds: 3, repeats: 3};
  const perfectlyRanked = (train, test) => test.map(i => ds.y[i]);
  const offsetRanked = (train, test) =>
    test.map(i => ds.y[i] + 10_000 * Math.min(...test));
  const offsetReversed = (train, test) =>
    test.map(i => (1 - ds.y[i]) + 10_000 * Math.min(...test));
  assert.equal(repeatedCv(ds, perfectlyRanked, options), 1);
  assert.equal(repeatedCv(ds, offsetRanked, options), 1,
    'comparison across different fitted fold models must not change AUC');
  assert.equal(repeatedCv(ds, offsetReversed, options), 0,
    'inverted within-fold ranking must not look predictive from fold offsets');
});

test('pure-label test folds abstain instead of silently pooling incompatible models', () => {
  const scarce = {...ds, y: Array.from({length: 12}, (_, i) => i < 2 ? 1 : 0)};
  const score = repeatedCv(scarce,
    (train, test) => test.map(i => scarce.y[i]), {seed: 7, folds: 5, repeats: 2});
  assert.equal(score, null);
});

test('the default product and exported report disclose the per-fold AUC estimand', () => {
  const run = runControls(SCENARIOS.balanced.make(), ['batch_only'],
    {seed: 20261030, perms: 10, k: 20});
  assert.equal(run.results.naive.aggregation, 'mean_per_heldout_fold');
  assert.equal(run.results.batch_only.aggregation, 'mean_per_heldout_fold');
  assert.ok(Number.isFinite(run.results.naive.auc));
  assert.match(toMarkdown('balanced', run), /within-fold|per-fold/i);
});
