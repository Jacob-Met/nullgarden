import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { join, dirname, extname } from 'node:path';
import { randomUUID } from 'node:crypto';
import { runControls, CONTROLS, designTable, toMarkdown } from './src/core.mjs';
import { SCENARIOS, parseCsv } from './src/data.mjs';
import { chat, extractJson, proposePrompt, validateProposal, explainPrompt, checkFigures, MODEL } from './src/llm.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const TYPES = { '.html': 'text/html; charset=utf-8', '.css': 'text/css', '.js': 'text/javascript', '.mjs': 'text/javascript' };
const store = new Map();

function remember(name, ds) {
  const id = randomUUID();
  store.set(id, { name, ds });
  if (store.size > 20) store.delete(store.keys().next().value);
  return id;
}
function describe(id) {
  const { name, ds } = store.get(id);
  return { datasetId: id, name, n: ds.y.length, features: ds.features.length, design: designTable(ds), model: MODEL };
}
async function body(req, limit = 6e6) {
  let size = 0; const chunks = [];
  for await (const c of req) { size += c.length; if (size > limit) throw Object.assign(new Error('Request too large.'), { status: 413 }); chunks.push(c); }
  return chunks.length ? JSON.parse(Buffer.concat(chunks).toString()) : {};
}
function controlsFrom(b) {
  const sel = Array.isArray(b.controls) ? b.controls.filter((c) => Object.hasOwn(CONTROLS, c)) : Object.keys(CONTROLS);
  return sel;
}

export function makeServer() {
  return createServer(async (req, res) => {
    const send = (code, obj, type = 'application/json') => { res.writeHead(code, { 'content-type': type, 'cache-control': 'no-store' }); res.end(type === 'application/json' ? JSON.stringify(obj) : obj); };
    try {
      const url = new URL(req.url, 'http://x');
      if (req.method === 'GET' && url.pathname === '/api/scenarios') return send(200, Object.entries(SCENARIOS).map(([id, s]) => ({ id, name: s.name, blurb: s.blurb })));
      if (req.method === 'GET' && url.pathname === '/api/status') return send(200, { model: MODEL, keyConfigured: Boolean((process.env.NEBIUS_API_KEY || '').trim()), controls: CONTROLS });
      if (req.method === 'POST' && url.pathname === '/api/dataset') {
        const b = await body(req);
        let name, ds;
        if (b.csv) { ds = parseCsv(String(b.csv)); name = String(b.name || 'Uploaded CSV').slice(0, 80); }
        else if (SCENARIOS[b.scenario]) { ds = SCENARIOS[b.scenario].make(); name = SCENARIOS[b.scenario].name; }
        else return send(400, { error: 'Unknown scenario.' });
        return send(200, describe(remember(name, ds)));
      }
      if (req.method === 'POST' && ['/api/run', '/api/propose', '/api/explain', '/api/report'].includes(url.pathname)) {
        const b = await body(req);
        const entry = store.get(b.datasetId);
        if (!entry) return send(404, { error: 'Dataset expired. Load it again.' });
        if (url.pathname === '/api/run') return send(200, runControls(entry.ds, controlsFrom(b)));
        if (url.pathname === '/api/propose') {
          const r = await chat(proposePrompt(designTable(entry.ds), String(b.note || '').slice(0, 500)));
          if (!r.ok) return send(200, r);
          const hypotheses = validateProposal(extractJson(r.text));
          if (!hypotheses.length) return send(200, { ok: false, reason: 'unparseable', message: 'Nemotron answered but not in the expected format. Try again.' });
          return send(200, { ok: true, hypotheses, model: r.model, ms: r.ms });
        }
        if (url.pathname === '/api/explain') {
          const run = runControls(entry.ds, controlsFrom(b));
          const r = await chat(explainPrompt(run), { maxTokens: 3000 });
          if (!r.ok) return send(200, r);
          return send(200, { ok: true, text: r.text, model: r.model, ms: r.ms, check: checkFigures(r.text, run) });
        }
        const run = runControls(entry.ds, controlsFrom(b));
        return send(200, toMarkdown(entry.name, run, b.narrative ? String(b.narrative).slice(0, 4000) : ''), 'text/markdown; charset=utf-8');
      }
      if (req.method === 'GET') {
        const f = url.pathname === '/' ? '/index.html' : url.pathname;
        if (!/^\/[\w.-]+$/.test(f)) return send(404, { error: 'Not found' });
        const data = await readFile(join(here, 'public', f)).catch(() => null);
        if (!data) return send(404, { error: 'Not found' });
        res.writeHead(200, { 'content-type': TYPES[extname(f)] || 'application/octet-stream' });
        return res.end(data);
      }
      send(405, { error: 'Method not allowed' });
    } catch (e) {
      send(e.status || 400, { error: e.status ? e.message : e.message || 'Bad request' });
    }
  });
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const port = Number(process.env.PORT || 4412);
  makeServer().listen(port, '127.0.0.1', () => console.log(`NULLGARDEN on http://127.0.0.1:${port}  (Nemotron: ${(process.env.NEBIUS_API_KEY || '').trim() ? MODEL : 'off, no NEBIUS_API_KEY'})`));
}
