import { rng } from './core.mjs';
import { conditionalUnits, conditionalClusters } from './exact-orbit.mjs';

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
  const withSubject = head[3] === 'subject_id';
  const withCluster = withSubject && head[4] === 'cluster_id';
  const featureStart = withCluster ? 5 : withSubject ? 4 : 3;
  if (head.slice(0, 3).join(',') !== 'sample_id,batch,label'
      || head.length <= featureStart || head.length > maxCols + featureStart
      || head.slice(featureStart).includes('subject_id')
      || head.slice(featureStart).includes('cluster_id')) {
    throw new Error('Columns must be sample_id,batch,label,[subject_id,[cluster_id,]], then at least one numeric feature. Nested cluster_id requires subject_id.');
  }
  const ids = [], batch = [], y = [], X = [], subject = [], cluster = [];
  lines.slice(1).forEach((l, i) => {
    const c = l.split(',').map((s) => s.trim());
    if (c.length !== head.length) throw new Error(`Row ${i + 2} has ${c.length} fields, expected ${head.length}.`);
    if (c[2] !== '0' && c[2] !== '1') throw new Error(`Row ${i + 2}: label must be 0 or 1.`);
    if (!c[0]) throw new Error(`Row ${i + 2}: sample_id is empty.`);
    if (withSubject && !c[3]) throw new Error(`Row ${i + 2}: subject_id is empty.`);
    if (withCluster && !c[4]) throw new Error(`Row ${i + 2}: cluster_id is empty.`);
    const row = c.slice(featureStart).map(Number);
    if (row.some((v) => !Number.isFinite(v))) throw new Error(`Row ${i + 2}: features must be finite numbers.`);
    if (!c[1]) throw new Error(`Row ${i + 2}: batch is empty.`);
    ids.push(c[0]); batch.push(c[1]); y.push(Number(c[2])); X.push(row);
    if (withSubject) subject.push(c[3]);
    if (withCluster) cluster.push(c[4]);
  });
  if (new Set(ids).size !== ids.length) throw new Error('sample_id values must be unique.');
  if (!y.includes(0) || !y.includes(1)) throw new Error('Need both labels.');
  // Reject cross-batch donors or contradictory labels *at intake*, before
  // any CV, p-value or export can silently treat dependent rows as independent.
  if (withCluster) conditionalClusters(batch,y,subject,cluster);
  else if (withSubject) conditionalUnits(batch,y,subject);
  return { ids, batch, y, X, features: head.slice(featureStart),
    ...(withSubject ? {subject} : {}),
    ...(withCluster ? {cluster} : {}) };
}
