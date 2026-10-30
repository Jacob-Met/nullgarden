import test from 'node:test';
import assert from 'node:assert/strict';
import { chat, extractJson, validateProposal, checkFigures, proposePrompt } from '../src/llm.mjs';
import { runControls } from '../src/core.mjs';
import { SCENARIOS } from '../src/data.mjs';

test('no key gives a graceful message and makes no request', async () => {
  let called = false;
  const r = await chat([{ role: 'user', content: 'x' }], { env: {}, fetchImpl: async () => { called = true; } });
  assert.equal(r.ok, false); assert.equal(r.reason, 'no-key'); assert.equal(called, false);
});
test('chat posts an OpenAI-style request to Token Factory and returns content', async () => {
  let seen;
  const fetchImpl = async (url, init) => { seen = { url, init }; return { ok: true, json: async () => ({ model: 'nvidia/m', choices: [{ message: { content: ' hi ', reasoning_content: 'ignored' } }] }) }; };
  const r = await chat([{ role: 'user', content: 'x' }], { env: { NEBIUS_API_KEY: 'k' }, fetchImpl, maxTokens: 2000 });
  assert.equal(r.text, 'hi');
  assert.equal(seen.url, 'https://api.tokenfactory.nebius.com/v1/chat/completions');
  const body = JSON.parse(seen.init.body);
  assert.match(body.model, /^nvidia\//); assert.ok(body.max_tokens >= 1500);
  assert.equal(seen.init.headers.authorization, 'Bearer k');
});
test('http failures are reported, not thrown', async () => {
  const r = await chat([], { env: { NEBIUS_API_KEY: 'k' }, fetchImpl: async () => ({ ok: false, status: 429 }) });
  assert.equal(r.ok, false); assert.match(r.message, /429/);
});
test('proposals are filtered to the known control menu', () => {
  const obj = extractJson('noise {"hypotheses":[{"shortcut":"a","control":"batch_only","why":"w"},{"shortcut":"b","control":"rm -rf","why":"w"},{"shortcut":"c","control":"__proto__","why":"w"}]} tail');
  const v = validateProposal(obj);
  assert.equal(v.length, 1); assert.equal(v[0].control, 'batch_only');
  assert.equal(extractJson('no json'), null);
});
test('prompt contains the batch table', () => {
  const p = proposePrompt([{ batch: '7', n: 5, pos: 5, neg: 0 }], '');
  assert.match(p[1].content, /batch 7: n=5, label1=5, label0=0/);
});
test('figure check flags numbers the controls never produced', () => {
  const run = runControls(SCENARIOS.shortcut_only.make(), ['batch_only']);
  const a = run.results.naive.auc.toFixed(2);
  const c = checkFigures(`AUC was ${a} and also 0.123 somewhere`, run);
  assert.equal(c.total, 2); assert.equal(c.matched, 1); assert.deepEqual(c.unmatched, [0.123]);
});
