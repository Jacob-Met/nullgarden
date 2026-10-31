import test from 'node:test';
import assert from 'node:assert/strict';
import * as core from '../src/core.mjs';

test('a nonsignificant batch-preserving permutation alone does not prove a shortcut', () => {
  const result = core.verdict({
    naive: { auc: 0.71 },
    within_batch_permutation: { status: 'shortcut', auc: 0.51, nullMean: 0.51, p: 0.51 },
  });
  assert.equal(result.id, 'open', 'failure to reject a null is not affirmative evidence of batch shortcut');
});

test('a positive batch-centering score alone cannot certify signal beyond a null', () => {
  const result = core.verdict({
    naive: { auc: 0.75 },
    batch_residualized: { status: 'signal', auc: 0.79 },
  });
  assert.equal(result.id, 'open', 'without a valid null, this is an exploratory indication');
});

test('fold batch centering never uses held-out values to compute another held-out prediction', () => {
  assert.equal(typeof core.centerBatchByTrainingFold, 'function', 'need fold-specific train-only preprocessing');
  const X = [[2], [4], [10], [12], [6], [7]];
  const b = ['A', 'A', 'B', 'B', 'C', 'C'];
  const train = [0, 2, 4], held = [1, 3, 5];
  const first = core.centerBatchByTrainingFold(X, b, train, held);
  assert.deepEqual(first.trainX, [[0], [0], [0]]);
  assert.deepEqual(first.testX, [[2], [2], [1]]);
  const changed = X.map((x) => x.slice()); changed[1][0] += 1000;
  const second = core.centerBatchByTrainingFold(changed, b, train, held);
  assert.deepEqual(second.testX[1], first.testX[1], 'altering another held-out sample must not perturb a held-out value');
  assert.deepEqual(second.trainX, first.trainX);
  const unseen = core.centerBatchByTrainingFold(X, b, [0, 2], [4]);
  assert.deepEqual(unseen.testX, [[0]], 'unseen test batch uses training-global mean (2+10)/2');
});
