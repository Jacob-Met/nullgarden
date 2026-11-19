import test from 'node:test';
import assert from 'node:assert/strict';
import { fitScore, auc, runControls } from '../src/core.mjs';
import { SCENARIOS } from '../src/data.mjs';

const Xtr = [
  [-3, 0.1], [-2, 0.6], [-1, -0.5],
  [1, 0.2], [2, -0.3], [3, 0.4],
];
const ytr = [0, 0, 0, 1, 1, 1];
const Xte = [[-1.5, 0.2], [1.5, -0.1], [0, 0.1]];

test('diagonal LDA scores are invariant to independent feature units', () => {
  const plain = fitScore(Xtr, ytr, Xte, 2);
  const scales = [1e-140, 1e140];
  const rescale = (X) => X.map(row => row.map((x,j) => x * scales[j]));
  const other = fitScore(rescale(Xtr), ytr, rescale(Xte), 2);
  assert.ok(plain[1] > plain[0]);
  for (let i = 0; i < plain.length; i++) {
    assert.ok(Number.isFinite(other[i]));
    assert.ok(Math.abs(other[i] - plain[i]) <= 1e-8 * (1 + Math.abs(plain[i])),
      `unit drift at ${i}: before=${plain[i]} after=${other[i]}`);
  }
});

test('very large but finite measurements do not overflow the within-fold variance', () => {
  const X = [[-4e155], [-3e155], [-2e155], [2e155], [3e155], [4e155]];
  const y = [0, 0, 0, 1, 1, 1];
  const out = fitScore(X, y, [[-3.5e155], [3.5e155]], 1);
  assert.equal(out.length, 2);
  assert.ok(out.every(Number.isFinite));
  assert.ok(out[0] < -1 && out[1] > 1, JSON.stringify(out));
});

test('tiny and huge features remain independently usable', () => {
  const inputs = Xtr.map(([x,z]) => [x * 1e-155, z * 1e155]);
  const tests = Xte.map(([x,z]) => [x * 1e-155, z * 1e155]);
  const out = fitScore(inputs, ytr, tests, 2);
  const plain = fitScore(Xtr, ytr, Xte, 2);
  out.forEach((v,i)=>{
    assert.ok(Number.isFinite(v));
    assert.ok(Math.abs(v-plain[i])<=1e-8*(1+Math.abs(plain[i])));
  });
});

test('a changed heldout feature cannot refit another heldout prediction', () => {
  const a = fitScore(Xtr,ytr,Xte,2);
  const changed = Xte.map(row=>row.slice());
  changed[0]=[1e150, -1e150];
  const b = fitScore(Xtr,ytr,changed,2);
  assert.equal(a[1], b[1]);
  assert.equal(a[2], b[2]);
});

test('a test-to-train dynamic-range ratio beyond IEEE double fails loudly', () => {
  assert.throws(() => fitScore([[1e-300], [2e-300], [-1e-300], [-2e-300]],
    [0,0,1,1], [[1e300]], 1), /normalized range|representable range/i);
});

test('a constant feature is neutral and never emits NaN', () => {
  const out=fitScore([[0],[0],[0],[0]],[0,0,1,1],[[1e300],[-1e300]],1);
  assert.deepEqual(out,[0,0]);
});

test('AUROC refuses non-finite evidence rather than ranking NaNs', () => {
  assert.throws(()=>auc([0,Number.POSITIVE_INFINITY],[0,1]), /non-finite/i);
  assert.throws(()=>auc([NaN,1],[0,1]), /non-finite/i);
});

test('scientific control AUCs are preserved under 300-order feature unit changes', () => {
  const ds=SCENARIOS.balanced.make();
  const converted={...ds,X:ds.X.map(row=>row.map((v,j)=>v*(j<100?1e155:1e-155)))};
  const controls=['batch_residualized','leave_batch_out'];
  const cfg={seed:20261030,perms:10,k:20};
  const before=runControls(ds,controls,cfg), after=runControls(converted,controls,cfg);
  for(const id of ['naive','batch_residualized','leave_batch_out']){
    const x=before.results[id].auc,y=after.results[id].auc;
    assert.ok(Number.isFinite(y),id);
    assert.ok(Math.abs(x-y)<=1e-9,`AUC changed for ${id}: ${x} vs ${y}`);
  }
});
