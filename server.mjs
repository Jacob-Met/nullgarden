import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { join, dirname, extname } from 'node:path';
import { randomUUID, createHash } from 'node:crypto';
import { runControls, CONTROLS, designTable, toMarkdown, conditionalUnits, conditionalClusters, conditionalOrbitSize } from './src/core.mjs';
import { SCENARIOS, parseCsv } from './src/data.mjs';
import { runFamilywiseK, planFamilywiseK } from './src/familywise.mjs';
import { planClusterTrialOutcome, runClusterTrialOutcome, trialOutcomeMarkdown } from './src/cluster-trial.mjs';
import { chat, extractJson, proposePrompt, validateProposal, explainPrompt, checkFigures, MODEL } from './src/llm.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const TYPES = { '.html': 'text/html; charset=utf-8', '.css': 'text/css', '.js': 'text/javascript', '.mjs': 'text/javascript' };
const store = new Map();

// Randomization is a declared experimental design, never inferred from
// observed labels or a shared cluster_id. Capture it once at dataset intake.
function registerRandomization(ds, request) {
  if(request===undefined||request===null)return ds;
  if(!request||typeof request!=='object'||Array.isArray(request)
    ||Object.keys(request).sort().join(',')!=='mechanism,protocolId,unit'
    ||request.unit!=='cluster'||request.mechanism!=='uniform_within_batch'
    ||typeof request.protocolId!=='string'
    ||!request.protocolId.trim()||request.protocolId.length>200) {
    throw new Error('Register exact randomization {unit:"cluster",mechanism:"uniform_within_batch",protocolId:"..."} at dataset intake; no observational inference upgrade');
  }
  if(!ds.subject||!ds.cluster) {
    throw new Error('Cluster-randomized assignment requires subject_id and cluster_id columns');
  }
  // Enforce whole-cluster, uniform-label, batch-nested trial allocation
  // before storing any registration or computing a scientific score.
  conditionalOrbitSize(ds.batch,ds.y,1,ds.subject,ds.cluster,'cluster');
  return {...ds,randomization:Object.freeze({...request})};
}

function registerTrialOutcome(ds, proposal) {
  if(proposal===undefined||proposal===null)return ds;
  const locked=planClusterTrialOutcome(ds,proposal);
  return {...ds,trialOutcome:locked};
}

function remember(name, ds, proposal) {
  // Register the family at dataset intake, before any NULLGARDEN results are
  // returned. Future family calls cannot change its hypotheses or seed.
  const family=planFamilywiseK(ds, proposal ?? {});
  if (family.stopping !== 'decision' || family.alpha !== 0.05) {
    throw new Error('The public family API fixes alpha=0.05 and decision stopping at intake');
  }
  const registrationDigest=createHash('sha256').update(JSON.stringify({
    schema:'nullgarden.family-intake.v1',
    data:{ids:ds.ids,batch:ds.batch,subject:ds.subject??null,cluster:ds.cluster??null,randomization:ds.randomization??null,trialOutcome:ds.trialOutcome??null,y:ds.y,X:ds.X,features:ds.features},
    family
  })).digest('hex');
  const id=randomUUID();
  store.set(id,{name,ds,family,registrationDigest});
  if(store.size>20)store.delete(store.keys().next().value);
  return id;
}
function describe(id) {
  const { name, ds, family, registrationDigest } = store.get(id);
  const units = conditionalUnits(ds.batch, ds.y, ds.subject ?? null);
  const clusters=ds.cluster?conditionalClusters(ds.batch,ds.y,ds.subject,ds.cluster):null;
  return { datasetId: id, name, n: ds.y.length, nSubjects: units.length,
    subjectMode: ds.subject ? 'declared_subject_blocks' : 'assumed_independent_rows',
    ...(clusters?{nClusters:clusters.length,nestedClusterMode:ds.randomization
      ? 'cluster_disjoint_whole_cluster_randomization'
      : 'cluster_disjoint_within_cluster_permutation'}:{}),
    ...(ds.randomization?{randomization:ds.randomization}:{}),
    ...(ds.trialOutcome?{trialOutcomeRegistration:ds.trialOutcome}:{}),
    features: ds.features.length, design: designTable(ds), model: MODEL,
    familyRegistration: {lockedAt:'dataset_intake',sha256:registrationDigest,...family} };
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
        ds=registerTrialOutcome(registerRandomization(ds,b.randomization),b.trialOutcome);
        return send(200, describe(remember(name, ds, b.family)));
      }
      if (req.method === 'POST' && ['/api/run', '/api/family', '/api/trial-outcome', '/api/propose', '/api/explain', '/api/report'].includes(url.pathname)) {
        const b = await body(req);
        if (Object.hasOwn(b,'randomization')||Object.hasOwn(b,'trialOutcome')||Object.hasOwn(b,'confidenceSet')) throw new Error('Treatment assignment, primary trial outcome, effect model and alpha are immutable after /api/dataset intake; reload dataset to preregister a distinct trial');
        const entry = store.get(b.datasetId);
        if (!entry) return send(404, { error: 'Dataset expired. Load it again.' });
        if (url.pathname === '/api/trial-outcome') {
          if (Object.keys(b).some(key=>key!=='datasetId')) {
            throw new Error('Trial outcome, seed and budget were locked at dataset intake; only datasetId is accepted');
          }
          if (!entry.ds.trialOutcome) {
            throw new Error('No primary trial outcome was registered when this dataset was loaded');
          }
          return send(200,{...runClusterTrialOutcome(entry.ds),
            registrationDigest:entry.registrationDigest,lockedAt:'dataset_intake'});
        }
        if (url.pathname === '/api/run') return send(200, runControls(entry.ds, controlsFrom(b), { stopping: 'decision' }));
        if (url.pathname === '/api/family') {
          if (Object.keys(b).some(key=>key!=='datasetId')) {
            throw new Error('Family hypotheses, seed and permutation budget are locked at /api/dataset intake; /api/family accepts only datasetId');
          }
          return send(200,{...runFamilywiseK(entry.ds,entry.family),
            familyLockedAt:'dataset_intake',registrationDigest:entry.registrationDigest});
        }
        if (url.pathname === '/api/propose') {
          const r = await chat(proposePrompt(designTable(entry.ds), String(b.note || '').slice(0, 500)));
          if (!r.ok) return send(200, r);
          const hypotheses = validateProposal(extractJson(r.text));
          if (!hypotheses.length) return send(200, { ok: false, reason: 'unparseable', message: 'Nemotron answered but not in the expected format. Try again.' });
          return send(200, { ok: true, hypotheses, model: r.model, ms: r.ms });
        }
        if (url.pathname === '/api/explain') {
          const run = runControls(entry.ds, controlsFrom(b), { stopping: 'decision' });
          const r = await chat(explainPrompt(run), { maxTokens: 3000 });
          if (!r.ok) return send(200, r);
          return send(200, { ok: true, text: r.text, model: r.model, ms: r.ms, check: checkFigures(r.text, run) });
        }
        const run = runControls(entry.ds, controlsFrom(b), { stopping: 'decision' });
        const main=toMarkdown(entry.name, run, b.narrative ? String(b.narrative).slice(0, 4000) : '');
        const trial=entry.ds.trialOutcome ? '\n'+trialOutcomeMarkdown(runClusterTrialOutcome(entry.ds)) : '';
        return send(200,main+trial,'text/markdown; charset=utf-8');
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
