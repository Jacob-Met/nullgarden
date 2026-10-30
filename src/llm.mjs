// Nemotron on Nebius Token Factory (OpenAI-compatible). Server-side only; the key never reaches the browser.
import { CONTROLS } from './core.mjs';

export const BASE_URL = process.env.NEBIUS_BASE_URL || 'https://api.tokenfactory.nebius.com/v1';
export const MODEL = process.env.NULLGARDEN_MODEL || 'nvidia/nemotron-3-super-120b-a12b';

export async function chat(messages, { fetchImpl = fetch, env = process.env, maxTokens = 3000, model = MODEL, timeoutMs = 90000 } = {}) {
  const key = (env.NEBIUS_API_KEY || '').trim();
  if (!key) return { ok: false, reason: 'no-key', message: 'NEBIUS_API_KEY is not set on the server. The deterministic controls still run; Nemotron proposals and explanations are off.' };
  const t0 = Date.now();
  try {
    const res = await fetchImpl(`${BASE_URL}/chat/completions`, {
      method: 'POST',
      headers: { 'content-type': 'application/json', authorization: `Bearer ${key}` },
      body: JSON.stringify({ model, messages, max_tokens: maxTokens, temperature: 0.2 }),
      signal: AbortSignal.timeout(timeoutMs),
    });
    if (!res.ok) return { ok: false, reason: 'http', message: `Token Factory returned HTTP ${res.status}.` };
    const j = await res.json();
    const text = j?.choices?.[0]?.message?.content;
    if (!text) return { ok: false, reason: 'empty', message: 'The model returned no content.' };
    return { ok: true, text: text.trim(), model: j.model || model, ms: Date.now() - t0 };
  } catch (e) {
    return { ok: false, reason: 'network', message: `Could not reach Token Factory (${e.name}).` };
  }
}

export function extractJson(text) {
  const s = text.indexOf('{'), e = text.lastIndexOf('}');
  if (s < 0 || e <= s) return null;
  try { return JSON.parse(text.slice(s, e + 1)); } catch { return null; }
}

export function proposePrompt(design, suspicion) {
  const rows = design.map((d) => `batch ${d.batch}: n=${d.n}, label1=${d.pos}, label0=${d.neg}`).join('\n');
  const menu = Object.entries(CONTROLS).map(([id, c]) => `- ${id}: ${c.question}`).join('\n');
  return [
    { role: 'system', content: 'You are a careful reviewer of scientific machine-learning evaluations. You reply with JSON only.' },
    { role: 'user', content: `A binary classifier is scored by cross-validation on this design.\n${rows}\n\nResearcher note: ${suspicion || '(none)'}\n\nAvailable negative controls:\n${menu}\n\nName up to 4 specific shortcut hypotheses this design makes plausible. For each, choose exactly one control id from the list that would test it. Reply as JSON: {"hypotheses":[{"shortcut":"<=18 words","control":"<id>","why":"<=25 words, refer to the batch counts"}]}` },
  ];
}

export function validateProposal(obj) {
  const out = [];
  for (const h of obj?.hypotheses || []) {
    if (typeof h?.shortcut === 'string' && typeof h?.why === 'string' && Object.hasOwn(CONTROLS, h.control)) out.push({ shortcut: h.shortcut.slice(0, 200), control: h.control, why: h.why.slice(0, 300) });
  }
  return out.slice(0, 4);
}

export function explainPrompt(run) {
  const lines = Object.values(run.results).map((r) => `${r.id}: AUC=${r.auc === null ? 'n/a' : r.auc.toFixed(2)}${r.p !== undefined ? ` p=${r.p.toFixed(3)} null_mean=${r.nullMean.toFixed(2)}` : ''}${r.status ? ` status=${r.status}` : ''}`).join('\n');
  return [
    { role: 'system', content: 'You explain machine-learning evaluation results to a bench scientist in plain English. Use only the numbers given. Round to two decimals. No hype.' },
    { role: 'user', content: `Results of negative controls on one classifier:\n${lines}\nVerdict from the deterministic rules: ${run.verdict.label}.\n\nIn at most 120 words: say what the model learned and what it merely exploited, quoting the figures, then one concrete next step.` },
  ];
}

// Every decimal the model writes must match a computed figure (to 0.006), else it is flagged.
export function checkFigures(text, run) {
  const known = [];
  for (const r of Object.values(run.results)) for (const k of ['auc', 'p', 'nullMean', 'nullLo', 'nullHi']) if (typeof r[k] === 'number') known.push(r[k]);
  const found = (text.match(/\d+\.\d+/g) || []).map(Number);
  const bad = found.filter((v) => !known.some((k) => Math.abs(k - v) <= 0.006));
  return { total: found.length, matched: found.length - bad.length, unmatched: bad };
}
