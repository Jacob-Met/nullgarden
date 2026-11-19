import test from 'node:test';
import assert from 'node:assert/strict';
import { fitScore, repeatedCv, runControls } from '../src/core.mjs';
import { SCENARIOS } from '../src/data.mjs';

const sparse = {
  ids: Array.from({ length: 12 }, (_, i) => 's' + i),
  y: Array.from({ length: 12 }, (_, i) => i === 0 ? 1 : 0),
  batch: Array.from({ length: 12 }, (_, i) => ['A', 'B', 'C'][i % 3]),
  X: Array.from({ length: 12 }, (_, i) => [i / 3, (i % 2) - 0.5]),
};

const pure = {
  ids: Array.from({ length: 20 }, (_, i) => 'p' + i),
  y: Array.from({ length: 20 }, (_, i) => i < 10 ? 1 : 0),
  batch: Array.from({ length: 20 }, (_, i) => i < 10 ? 'P' : 'N'),
  X: Array.from({ length: 20 }, (_, i) => [i < 10 ? 10 + i / 10 : -10 + i / 10]),
};

test('direct classifier cannot emit a fake score after losing one class in training', () => {
  assert.throws(
    () => fitScore([[1], [2], [3]], [0, 0, 0], [[4]], 1),
    (error) => error.code === 'CV_TRAIN_CLASS_MISSING',
  );
});

test('one minority observation makes stratified fold training unsupported', () => {
  const score = repeatedCv(sparse,
    (train, test) => test.map(() => train.length / 12),
    { seed: 20261030, repeats: 3, folds: 5 });
  assert.equal(score, null);
});

test('scarce classes must abstain across every control, not fake permutation evidence', () => {
  const r = runControls(sparse, undefined, { seed: 20261030, perms: 20, k: 2 });
  assert.equal(r.results.naive.auc, null);
  assert.equal(r.results.naive.status, 'unavailable');
  assert.match(r.results.naive.finding, /class|fold|minority/i);
  assert.equal(r.results.batch_only.auc, null);
  assert.equal(r.results.batch_only.status, 'unavailable');
  assert.equal(r.results.within_batch_permutation.p, null);
  assert.equal(r.results.within_batch_permutation.auc, null);
  assert.equal(r.results.within_batch_permutation.status, 'unavailable');
  assert.equal(r.results.leave_batch_out.auc, null);
  assert.equal(r.results.batch_residualized.auc, null);
  assert.notEqual(r.verdict.id, 'signal');
  assert.notEqual(r.verdict.id, 'both');
});

test('holding out a class-exclusive batch must not fit a single-class model', () => {
  const r = runControls(pure, ['leave_batch_out'], {
    seed: 20261030, perms: 20, k: 1,
  });
  const held = r.results.leave_batch_out;
  assert.equal(held.auc, null);
  assert.equal(held.pooledOofAuc, null);
  assert.equal(held.nMixedHeldout, 0);
  assert.equal(held.nUnsupportedTrain, 2);
  assert.ok(held.perBatch.every((x) => x.trainSupported === false));
  assert.match(held.finding, /class|unidentifiable|train/i);
  assert.notEqual(r.verdict.id, 'signal');
});

test('adequate mixed-batch classes still run the unchanged evaluation', () => {
  const r = runControls(SCENARIOS.shortcut_only.make(), undefined,
    { seed: 20261030, perms: 40, k: 20 });
  assert.ok(Number.isFinite(r.results.naive.auc));
  assert.ok(Number.isFinite(r.results.within_batch_permutation.p));
  assert.equal(r.results.leave_batch_out.nUnsupportedTrain, 0);
  assert.equal(r.results.leave_batch_out.nMixedHeldout, 2);
  assert.equal(r.results.batch_residualized.preprocessing, 'training_fold_only');
  assert.equal(r.verdict.id, 'shortcut');
});
