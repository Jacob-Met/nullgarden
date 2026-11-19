import test from 'node:test';
import assert from 'node:assert/strict';
import { permutationDecisionBounds, runControls, toMarkdown } from '../src/core.mjs';
import { SCENARIOS } from '../src/data.mjs';
import { makeServer } from '../server.mjs';
import { checkFigures, explainPrompt } from '../src/llm.mjs';

// Exhaust every possible exceedance stream: stopping after a prefix must
// NEVER change the decision of the prespecified completed-B permutation test.
test('early decisions are identical to all possible unseen suffixes (exact oracle)', () => {
  const budget = 9, alpha = 0.3;
  let rejected = 0, retained = 0;
  for (let pattern = 0; pattern < 1 << budget; pattern++) {
    const losses = Array.from({length: budget}, (_, i) => (pattern >> i) & 1);
    const finalReject = (1 + losses.reduce((a,b)=>a+b,0))/(budget+1) <= alpha;
    let count = 0;
    for (let drawn = 0; drawn <= budget; drawn++) {
      const b = permutationDecisionBounds(count, drawn, budget, alpha);
      assert.equal(b.lower, (count + 1)/(budget+1));
      assert.equal(b.upper, (count + 1 + budget - drawn)/(budget+1));
      assert.ok(b.lower <= b.upper);
      if (b.decision !== 'continue') {
        assert.equal(b.decision === 'reject', finalReject,
          'interim outcome disagreed with completed fixed-budget test');
        if (b.decision === 'reject') rejected++; else retained++;
        break;
      }
      if (drawn < budget) count += losses[drawn];
    }
  }
  assert.ok(rejected > 0 && retained > 0);
});

test('the production conditional null uses a valid conservative anytime upper p bound', () => {
  for (const id of ['shortcut_only', 'shortcut_and_signal', 'balanced']) {
    const ds = SCENARIOS[id].make();
    const args = {seed:20261030, perms:40, k:20};
    const full = runControls(ds,['within_batch_permutation'],args).results.within_batch_permutation;
    const stopped = runControls(ds,['within_batch_permutation'],
      {...args, stopping:'decision'}).results.within_batch_permutation;
    assert.equal(stopped.status,full.status,id + ' fixed-horizon decision changed');
    assert.ok(stopped.permsRun > 0 && stopped.permsRun <= 40);
    assert.equal(stopped.permsBudget,40);
    assert.ok(stopped.pLower <= full.p && full.p <= stopped.pUpper);
    assert.equal(stopped.p,stopped.pUpper,'p must be conservative anytime upper');
    assert.equal(stopped.pExact,stopped.permsRun === stopped.permsBudget);
    assert.equal(stopped.stopping,'decision');
    if (!stopped.pExact) {
      assert.equal(stopped.auc,null,'truncated/adaptively selected null means are not calibrated');
      assert.equal(stopped.nullLo,undefined);
    }
  }
  const shortcut = runControls(SCENARIOS.shortcut_only.make(),
    ['within_batch_permutation'], {seed:20261030,perms:40,stopping:'decision'});
  assert.ok(shortcut.results.within_batch_permutation.permsRun < 40,
    'null control should save work, not just carry metadata');
  assert.match(toMarkdown('shortcut',shortcut), /anytime|conservative|bounded/i);
});

test('cold HTTP run and report use decision stopping with honest bounded inference', async () => {
  const server = makeServer();
  await new Promise((resolve) => server.listen(0,'127.0.0.1',resolve));
  const base = 'http://127.0.0.1:' + server.address().port;
  const post = (path, payload) => fetch(base + path,
    {method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(payload)});
  try {
    const data = await (await post('/api/dataset',{scenario:'shortcut_only'})).json();
    const run = await (await post('/api/run',
      {datasetId:data.datasetId,controls:['within_batch_permutation']})).json();
    const p = run.results.within_batch_permutation;
    assert.equal(p.stopping,'decision');
    assert.ok(p.permsRun < p.permsBudget);
    assert.equal(p.p,p.pUpper);
    assert.equal(run.verdict.id,'open');
    const md = await (await post('/api/report',
      {datasetId:data.datasetId,controls:['within_batch_permutation']})).text();
    assert.match(md,/conservative|p-value bound/i);
  } finally { await new Promise(resolve => server.close(resolve)); }
});

test('LLM explanation and figure grounding accept early bounds without inventing a null mean', () => {
  const run = runControls(SCENARIOS.shortcut_only.make(),
    ['within_batch_permutation'], {seed:20261030,perms:40,stopping:'decision'});
  const p = run.results.within_batch_permutation;
  assert.equal(p.pExact,false);
  assert.equal(p.nullMean,null);
  const prompt = explainPrompt(run)[1].content;
  assert.match(prompt,/conservative upper bound/);
  assert.doesNotMatch(prompt,/null_mean=NaN/);
  const reported = p.pLower.toFixed(3) + ' to ' + p.pUpper.toFixed(3);
  const validation = checkFigures(reported,run);
  assert.equal(validation.total,2);
  assert.equal(validation.matched,2);
  assert.deepEqual(validation.unmatched,[]);
});
