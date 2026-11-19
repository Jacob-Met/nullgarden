import test from 'node:test';
import assert from 'node:assert/strict';
import { runControls, mean } from '../src/core.mjs';
import { SCENARIOS } from '../src/data.mjs';

const pureBatchLabels = {
  ids: Array.from({length:24},(_,i)=>'pure-'+i),
  batch: Array.from({length:24},(_,i)=>i < 12 ? 'A':'B'),
  y: Array.from({length:24},(_,i)=>i < 12 ? 1:0),
  X: Array.from({length:24},(_,i)=>[
    Math.sin(i * 9.13) + (i < 12 ? 0.15 : -0.15),
    Math.cos(i * 7.71),
    ((i * 13) % 17 - 8) / 11,
  ]),
};

const nullWithinBatches = {
  ids: Array.from({length:40},(_,i)=>'null-'+i),
  batch: Array.from({length:40},(_,i)=>['A','B','C','D'][Math.floor(i / 10)]),
  y: Array.from({length:40},(_,i)=>i % 10 < 5 ? 1 : 0),
  X: Array.from({length:40},()=>[0,0]),
};

test('permuting pure batches changes no labels and therefore cannot alter the CV statistic', () => {
  // The observed training statistic MUST be exactly the statistic applied to
  // each permuted dataset. Previously observed used 3 repeats but the null
  // used 1 repeat and a different seed for every permutation, so even an
  // identical label vector produced non-identical reference distributions.
  const r = runControls(pureBatchLabels, ['within_batch_permutation'], {
    seed: 20261030, perms: 32, k: 2,
  });
  const observed = r.results.naive.auc, perm = r.results.within_batch_permutation;
  assert.ok(observed > 0 && observed < 1);
  assert.equal(perm.p, 1);
  assert.equal(perm.nullMean, observed);
  assert.equal(perm.nullLo, observed);
  assert.equal(perm.nullHi, observed);
  assert.deepEqual(perm.statistic, { seed: 20261030, folds: 5, repeats: 3 });
});

test('failure to reject a no-signal null is not affirmative evidence of a batch shortcut', () => {
  const r = runControls(nullWithinBatches, ['within_batch_permutation'], {
    seed: 20261030, perms: 30, k: 2,
  });
  const permutation = r.results.within_batch_permutation;
  assert.equal(r.results.naive.auc, 0.5);
  assert.equal(permutation.p, 1);
  assert.equal(permutation.nullMean, 0.5);
  assert.equal(permutation.status, 'inconclusive');
  assert.equal(r.verdict.id, 'open');
  assert.doesNotMatch(permutation.finding, /batch structure alone explains/i);
});

test('fixed-statistic null still rejects strong genuine within-batch signal', () => {
  const r = runControls(SCENARIOS.balanced.make(), ['within_batch_permutation'], {
    seed: 20261030, perms: 40, k: 20,
  });
  assert.ok(r.results.within_batch_permutation.p <= 0.05);
  assert.equal(r.results.within_batch_permutation.status, 'signal');
  // One significant permutation test alone does not certify signal without
  // its separate fold-local centering corroborator.
  assert.equal(r.verdict.id, 'open');
});

test('constant or high-magnitude statistical moments are not changed by accumulation', () => {
  const value = 0.34722222222222227;
  assert.equal(mean(Array.from({ length: 32 }, () => value)), value);
  assert.equal(mean([1e308, 1e308, 1e308]), 1e308);
});
