import test from 'node:test';
import assert from 'node:assert/strict';
import { auc, rng, runControls, designTable, verdict } from '../src/core.mjs';
import { SCENARIOS, parseCsv } from '../src/data.mjs';

test('auc matches hand-computed values', () => {
  assert.equal(auc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]), 1);
  assert.equal(auc([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]), 0);
  assert.equal(auc([1, 1, 1, 1], [0, 1, 0, 1]), 0.5);
  assert.equal(auc([1, 2], [1, 1]), null);
});
test('rng is deterministic', () => { const a = rng(5), b = rng(5); assert.equal(a(), b()); });
test('design table reports pure and mixed batches', () => {
  const t = designTable(SCENARIOS.shortcut_only.make());
  assert.equal(t.length, 5);
  assert.deepEqual(t.filter((b) => b.mixed).map((b) => b.batch).sort(), ['2', '5']);
  assert.equal(t.reduce((s, b) => s + b.n, 0), 57);
});
test('shortcut-only data is flagged as shortcut', () => {
  const r = runControls(SCENARIOS.shortcut_only.make());
  assert.equal(r.verdict.id, 'shortcut');
  assert.ok(r.results.batch_only.auc > 0.7);
  assert.ok(r.results.within_batch_permutation.p > 0.05);
});
test('real signal beats the within-batch null', () => {
  const r = runControls(SCENARIOS.balanced.make());
  assert.ok(r.results.within_batch_permutation.p <= 0.05);
  assert.ok(r.results.leave_batch_out.auc > 0.9);
  assert.equal(r.verdict.id, 'signal');
});
test('runs are reproducible', () => {
  const a = runControls(SCENARIOS.shortcut_and_signal.make(), ['batch_only']);
  const b = runControls(SCENARIOS.shortcut_and_signal.make(), ['batch_only']);
  assert.equal(a.results.naive.auc, b.results.naive.auc);
});
test('verdict without controls is unrun', () => assert.equal(verdict({ naive: { auc: 0.9 } }).id, 'unrun'));
test('csv parser validates input', () => {
  const rows = Array.from({ length: 12 }, (_, i) => `s${i},b${i % 2},${i % 2},${i},${i * 2}`);
  const ds = parseCsv(['sample_id,batch,label,f1,f2', ...rows].join('\n'));
  assert.equal(ds.X.length, 12);
  assert.throws(() => parseCsv(['sample_id,batch,label,f1', ...rows.map((r) => r.replace(/,\d+,\d+$/, ',x'))].join('\n')));
  assert.throws(() => parseCsv('a,b,c,d\n1,2,3,4'));
});
