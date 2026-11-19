import test from 'node:test';
import assert from 'node:assert/strict';
import { centerFoldByTrainingBatch, batchCenteredScorer, runControls } from '../src/core.mjs';
import { SCENARIOS } from '../src/data.mjs';

const X = [
  [1, 10], [3, 14], [15, 21], [17, 25],
  [-9, 3], [-7, 5], [9, 12], [11, 14],
];
const batch = ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'];
const y = [0, 1, 1, 0, 0, 1, 1, 0];
const train = [0, 1, 4, 5], heldout = [2, 3, 6, 7];

test('changed heldout values cannot contaminate training batch means', () => {
  const before = centerFoldByTrainingBatch(X, batch, train, heldout);
  const changed = X.map(row => row.slice());
  changed[2] = [1e8, -1e8];
  changed[6] = [-1e8, 1e8];
  const after = centerFoldByTrainingBatch(changed, batch, train, heldout);
  assert.deepEqual(before.missingBatches, []);
  assert.deepEqual(before.trainX, [[-1, -2], [1, 2], [-1, -1], [1, 1]]);
  assert.deepEqual(before.trainX, after.trainX);
  // Untouched test rows must also remain centered from training-only means.
  assert.deepEqual(before.testX[1], after.testX[1]);
  assert.deepEqual(before.testX[3], after.testX[3]);
});

test('fold-level predictions cannot depend on another heldout row', () => {
  const before = batchCenteredScorer({X, y, batch}, 2)(train, heldout);
  const mutated = X.map(row => row.slice());
  mutated[2] = [1e8, -1e8];
  const after = batchCenteredScorer({X: mutated, y, batch}, 2)(train, heldout);
  assert.equal(before.length, heldout.length);
  assert.equal(before[1], after[1]);
  assert.equal(before[2], after[2]);
  assert.equal(before[3], after[3]);
});

test('unobserved test batches fail closed rather than using transductive statistics', () => {
  const result = centerFoldByTrainingBatch(X, batch, [0, 1, 2], [4, 5]);
  assert.deepEqual(result.missingBatches, ['B']);
  assert.equal(result.trainX, null);
  assert.equal(result.testX, null);
  assert.throws(
    () => batchCenteredScorer({X, y, batch}, 2)([0, 1, 2], [4, 5]),
    (err) => err.code === 'BATCH_NOT_IN_TRAIN' && err.batches.includes('B'),
  );
});

test('cold control retains grounded shortcut and signal distinctions', () => {
  for (const name of ['shortcut_only','shortcut_and_signal','balanced']) {
    const result = runControls(SCENARIOS[name].make(), undefined,
      {seed:20261030,perms:40,k:20});
    assert.equal(result.results.batch_residualized.preprocessing,
      'training_fold_only');
    assert.ok(Number.isFinite(result.results.batch_residualized.auc));
    assert.equal(result.verdict.id,
      name === 'shortcut_only' ? 'shortcut' : 'signal');
  }
});
