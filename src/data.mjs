import { rng } from './core.mjs';

// Batch/label counts are the real bladderbatch design recorded in
// components/batch-confounding/examples/bladderbatch (57 samples, 5 batches).
// Expression values are SYNTHETIC: a batch effect on 80 features and, optionally, a label signal on 20.
export const REAL = [['3', 0, 4], ['2', 14, 4], ['5', 15, 4], ['1', 11, 0], ['4', 0, 5]];
const BALANCED = REAL.map(([b, p, n]) => [b, Math.floor((p + n) / 2), (p + n) - Math.floor((p + n) / 2)]);

export function build(design, signal, seed, nFeat = 200) {
  const r = rng(seed);
  const gauss = () => { let u = 0, v = 0; while (!u) u = r(); v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
  const eff = {};
  for (const [b] of design) eff[b] = Array.from({ length: 80 }, () => gauss() * 1.2);
  const ids = [], batch = [], y = [], X = [];
  for (const [b, pos, neg] of design) {
    for (let m = 0; m < pos + neg; m++) {
      const lab = m < pos ? 1 : 0;
      const row = Array.from({ length: nFeat }, () => gauss());
      for (let j = 0; j < 80; j++) row[j] += eff[b][j];
      for (let j = 80; j < 100; j++) row[j] += signal * (lab ? 0.5 : -0.5);
      ids.push(`B${b}-${String(m + 1).padStart(2, '0')}`); batch.push(b); y.push(lab); X.push(row);
    }
  }
  return { ids, batch, y, X, features: Array.from({ length: nFeat }, (_, j) => `g${j + 1}`) };
}

export const SCENARIOS = {
  shortcut_only: { name: 'Shortcut only', blurb: 'bladderbatch batch design, no label signal in the measurements', make: () => build(REAL, 0, 1) },
  shortcut_and_signal: { name: 'Shortcut plus signal', blurb: 'bladderbatch batch design, weak real signal added', make: () => build(REAL, 1.1, 12) },
  balanced: { name: 'Balanced design', blurb: 'same batches, labels balanced inside each batch, real signal', make: () => build(BALANCED, 1.1, 13) },
};

export function parseCsv(text, { maxRows = 2000, maxCols = 2000 } = {}) {
  const lines = text.replace(/\r/g, '').split('\n').filter((l) => l.trim());
  if (lines.length < 11) throw new Error('Need a header and at least 10 samples.');
  if (lines.length - 1 > maxRows) throw new Error(`At most ${maxRows} samples.`);
  const head = lines[0].split(',').map((s) => s.trim());
  if (head.length < 4 || head.length > maxCols + 3) throw new Error('Columns must be: sample_id, batch, label, then at least one numeric feature.');
  const ids = [], batch = [], y = [], X = [];
  lines.slice(1).forEach((l, i) => {
    const c = l.split(',').map((s) => s.trim());
    if (c.length !== head.length) throw new Error(`Row ${i + 2} has ${c.length} fields, expected ${head.length}.`);
    if (c[2] !== '0' && c[2] !== '1') throw new Error(`Row ${i + 2}: label must be 0 or 1.`);
    const row = c.slice(3).map(Number);
    if (row.some((v) => !Number.isFinite(v))) throw new Error(`Row ${i + 2}: features must be finite numbers.`);
    if (!c[1]) throw new Error(`Row ${i + 2}: batch is empty.`);
    ids.push(c[0]); batch.push(c[1]); y.push(Number(c[2])); X.push(row);
  });
  if (new Set(ids).size !== ids.length) throw new Error('sample_id values must be unique.');
  if (!y.includes(0) || !y.includes(1)) throw new Error('Need both labels.');
  return { ids, batch, y, X, features: head.slice(3) };
}
