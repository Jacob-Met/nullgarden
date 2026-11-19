import test from 'node:test';
import assert from 'node:assert/strict';
import { auc, macroBatchAuc, runControls } from '../src/core.mjs';
import { SCENARIOS } from '../src/data.mjs';

test('cross-batch pooled AUC can fabricate discrimination from pure batch intercepts', () => {
  // Two pure-label batches completely dominate pooled out-of-fold ordering.
  // The sole mixed-label heldout batch has no discrimination at all.
  const batch = ['A','A','B','B','C','C'];
  const labels = [1,1,0,0,1,0];
  const scores = [10,10,-10,-10,0,0];
  assert.ok(auc(scores, labels) > 0.9);
  const result = macroBatchAuc(scores, labels, batch);
  assert.equal(result.auc, 0.5);
  assert.equal(result.nMixedHeldout, 1);
  assert.equal(result.nPureHeldout, 2);
  assert.deepEqual(result.perBatch.map(row => [row.batch,row.auc]),
    [['A', null], ['B', null], ['C', 0.5]]);
});

test('mixed-batch macro AUC does not mix incompatible independent score scales', () => {
  const batch = ['A','A','B','B','C','C'];
  const labels = [0,1,0,1,0,1];
  const scores = [100,101,-100,-99,0,1];
  const result = macroBatchAuc(scores, labels, batch);
  assert.equal(result.auc, 1);
  assert.equal(result.nMixedHeldout, 3);
});

test('real bladderbatch evaluation separates same-model macro AUC from pooled diagnostic', () => {
  const r = runControls(SCENARIOS.shortcut_only.make(),
    ['leave_batch_out'], {seed:20261030,perms:40,k:20});
  const c = r.results.leave_batch_out;
  assert.equal(c.nMixedHeldout, 2);
  assert.equal(c.nPureHeldout, 3);
  assert.ok(Number.isFinite(c.auc));
  assert.ok(Number.isFinite(c.pooledOofAuc));
  assert.equal(c.aggregation, 'mean_per_mixed_heldout_batch');
});

test('all pure heldout groups must not be reported as a valid AUROC', () => {
  const n = 12;
  const ds = {
    X: Array.from({length:n},(_,i)=>[i/10, (i%3)-1]),
    y: Array.from({length:n},(_,i)=>i<6 ? 1 : 0),
    batch: Array.from({length:n},(_,i)=>i<6 ? 'P':'N'),
    ids: Array.from({length:n},(_,i)=>String(i)),
  };
  const result=runControls(ds,['leave_batch_out'],{seed:20261030,perms:10,k:2});
  assert.equal(result.results.leave_batch_out.auc, null);
  assert.equal(result.results.leave_batch_out.nMixedHeldout, 0);
  assert.equal(result.results.leave_batch_out.status, 'neutral');
  assert.notEqual(result.verdict.id, 'signal');
});
