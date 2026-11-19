import test from 'node:test';
import assert from 'node:assert/strict';
import { designTable, fitScore, mean, repeatedCv, rng, runControls, shuffle } from '../src/core.mjs';
import { SCENARIOS } from '../src/data.mjs';

// Independent reference: the observed dataset and *every* conditional-null
// relabeling must be processed through the IDENTICAL deterministic statistic.
// This is the key exchangeability invariant for a Monte Carlo permutation p.
function reference(ds, { seed, perms, k }) {
  const statistic = (y) => repeatedCv({ ...ds, y }, (train, heldout) =>
    fitScore(train.map((i) => ds.X[i]), train.map((i) => y[i]),
      heldout.map((i) => ds.X[i]), k), { seed, repeats: 3, folds: 5 });
  const observed = statistic(ds.y);
  assert.notEqual(observed, null, 'reference controls require identifiable training folds');
  const rand = rng(seed + 2);
  const groups = designTable(ds).map((group) => group.samples.map((s) => s.i));
  const nulls = Array.from({ length: perms }, () => {
    const labels = ds.y.slice();
    for (const ids of groups) {
      const relabeled = shuffle(ids.map((i) => ds.y[i]), rand);
      ids.forEach((i, j) => { labels[i] = relabeled[j]; });
    }
    return statistic(labels);
  });
  assert.ok(nulls.every(Number.isFinite));
  return {
    observed, nullMean: mean(nulls),
    p: (1 + nulls.filter((stat) => stat >= observed).length) / (perms + 1)
  };
}

test('frozen batch-conditional permutation oracle uses the SAME statistic as the observation', () => {
  for (const name of ['shortcut_only', 'shortcut_and_signal', 'balanced']) {
    const ds = SCENARIOS[name].make();
    for (const seed of [17, 20261030]) {
      const opts = { seed, perms: 23, k: 20 };
      const actual = runControls(ds, ['within_batch_permutation'], opts);
      const expected = reference(ds, opts);
      const result = actual.results.within_batch_permutation;
      assert.equal(actual.results.naive.auc, expected.observed, name + ' observed');
      assert.equal(result.nullMean, expected.nullMean, name + ' null statistic mismatch');
      assert.equal(result.p, expected.p, name + ' invalid permutation rank');
      assert.equal(result.statisticRepeats, 3, 'observed and null must both use three CV repeats');
      assert.equal(result.statisticSeed, seed, 'null must reuse the observation CV randomness');
      assert.equal(result.nullSampling, 'batch_conditional_fixed_statistic');
    }
  }
});

test('permutation inference reports its fixed budget and restricted scientific validity', () => {
  const result = runControls(SCENARIOS.balanced.make(),
    ['within_batch_permutation'], { seed: 9, perms: 23 }).results.within_batch_permutation;
  assert.match(result.inferenceLimit, /exchangeab/i);
  assert.match(result.inferenceLimit, /peek|repeat|post.hoc/i);
  assert.ok(result.p >= 1 / 24 && result.p <= 1);
  assert.ok(result.perms === 23);
});

test('failure to reject the conditional null is NOT positive evidence of batch shortcut', () => {
  const ds = SCENARIOS.shortcut_only.make();
  const r = runControls(ds, ['within_batch_permutation'],
    { seed: 20261030, perms: 39, k: 20 });
  assert.ok(r.results.within_batch_permutation.p > 0.05);
  assert.equal(r.results.within_batch_permutation.status, 'neutral');
  assert.equal(r.verdict.id, 'open',
    'non-significance cannot by itself attribute the entire score to batch');
  assert.doesNotMatch(r.results.within_batch_permutation.finding,
    /alone explains|proves.*batch/i);
});
