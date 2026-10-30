import test from 'node:test';
import assert from 'node:assert/strict';
import { makeServer } from '../server.mjs';

async function withServer(fn) {
  const s = makeServer(); await new Promise((r) => s.listen(0, '127.0.0.1', r));
  const base = `http://127.0.0.1:${s.address().port}`;
  try { await fn(base); } finally { s.close(); }
}
const post = (base, p, b) => fetch(base + p, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(b) });

test('serves the app and runs controls end to end without a key', async () => {
  const saved = process.env.NEBIUS_API_KEY; delete process.env.NEBIUS_API_KEY;
  await withServer(async (base) => {
    assert.match(await (await fetch(base + '/')).text(), /NULLGARDEN/);
    const d = await (await post(base, '/api/dataset', { scenario: 'shortcut_only' })).json();
    assert.equal(d.n, 57);
    const run = await (await post(base, '/api/run', { datasetId: d.datasetId, controls: ['batch_only', 'nonsense'] })).json();
    assert.deepEqual(Object.keys(run.results), ['naive', 'batch_only']);
    const p = await (await post(base, '/api/propose', { datasetId: d.datasetId })).json();
    assert.equal(p.ok, false); assert.equal(p.reason, 'no-key');
    const md = await (await post(base, '/api/report', { datasetId: d.datasetId })).text();
    assert.match(md, /# NULLGARDEN control report/);
    assert.equal((await post(base, '/api/run', { datasetId: 'nope' })).status, 404);
  });
  if (saved) process.env.NEBIUS_API_KEY = saved;
});
