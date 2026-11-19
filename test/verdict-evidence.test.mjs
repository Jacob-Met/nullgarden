import test from 'node:test';
import assert from 'node:assert/strict';
import { verdict, runControls } from '../src/core.mjs';
import { SCENARIOS } from '../src/data.mjs';

const base = { naive: { id: 'naive', auc: 0.94 } };
const permutation = { id:'within_batch_permutation', status:'signal', p:0.01 };
const centered = { id:'batch_residualized', status:'signal', auc:0.91 };

test('a single favorable control cannot certify robust within-batch signal', () => {
  assert.equal(verdict({ ...base, within_batch_permutation: permutation }).id, 'open');
  assert.equal(verdict({ ...base, batch_residualized: centered }).id, 'open');
  assert.match(verdict({ ...base, batch_residualized: centered }).text,
    /both|two|independent|additional/i);
});

test('two distinct compatible signal controls can establish the qualified verdict', () => {
  assert.equal(verdict({ ...base, within_batch_permutation: permutation,
    batch_residualized: centered }).id, 'signal');
  assert.equal(verdict({ ...base, batch_only: {status: 'shortcut'},
    within_batch_permutation: permutation,
    batch_residualized: centered }).id, 'both');
});

test('a shortcut with just one favorable control is not labeled both', () => {
  assert.equal(verdict({ ...base, batch_only: {status: 'shortcut'},
    batch_residualized: centered }).id, 'shortcut');
  assert.equal(verdict({ ...base, batch_only: {status: 'shortcut'},
    within_batch_permutation: permutation }).id, 'shortcut');
});

test('real data with only one requested signal control cannot be certified', () => {
  const r=runControls(SCENARIOS.balanced.make(),['batch_residualized'],
    {seed:20261030,perms:40,k:20});
  assert.equal(r.results.batch_residualized.status, 'signal');
  assert.equal(r.verdict.id,'open');
});
