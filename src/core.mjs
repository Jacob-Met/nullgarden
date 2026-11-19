// Deterministic core: a small, fully specified pipeline plus the negative controls that run it.
// Nothing here calls a model. Nemotron proposes which controls to look at; this file executes them.

export function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function shuffle(arr, r) {
  const a = arr.slice();
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(r() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

export function auc(scores, y) {
  const idx = scores.map((s, i) => i).sort((a, b) => scores[a] - scores[b]);
  let nPos = 0, rankSum = 0;
  for (let i = 0; i < idx.length; ) {
    let j = i;
    while (j + 1 < idx.length && scores[idx[j + 1]] === scores[idx[i]]) j++;
    const avg = (i + j) / 2 + 1;
    for (let k = i; k <= j; k++) if (y[idx[k]] === 1) { nPos++; rankSum += avg; }
    i = j + 1;
  }
  const nNeg = y.length - nPos;
  if (nPos === 0 || nNeg === 0) return null;
  return (rankSum - (nPos * (nPos + 1)) / 2) / (nPos * nNeg);
}

// The pipeline under test: top-k feature selection by |t|, then diagonal-LDA scoring.
// Selection happens inside every training fold, as it must.
export function fitScore(Xtr, ytr, Xte, k = 20) {
  const p = Xtr[0].length;
  const n1 = ytr.filter((v) => v === 1).length, n0 = ytr.length - n1;
  if (n1 === 0 || n0 === 0) return Xte.map(() => 0);
  const mu1 = new Float64Array(p), mu0 = new Float64Array(p), v = new Float64Array(p);
  for (let i = 0; i < Xtr.length; i++) for (let j = 0; j < p; j++) (ytr[i] ? mu1 : mu0)[j] += Xtr[i][j];
  for (let j = 0; j < p; j++) { mu1[j] /= n1; mu0[j] /= n0; }
  for (let i = 0; i < Xtr.length; i++) for (let j = 0; j < p; j++) {
    const d = Xtr[i][j] - (ytr[i] ? mu1[j] : mu0[j]); v[j] += d * d;
  }
  const t = [];
  for (let j = 0; j < p; j++) {
    const s2 = v[j] / Math.max(1, Xtr.length - 2) + 1e-9;
    t.push({ j, t: (mu1[j] - mu0[j]) / Math.sqrt(s2 * (1 / n1 + 1 / n0)), s2 });
  }
  t.sort((a, b) => Math.abs(b.t) - Math.abs(a.t));
  const top = t.slice(0, Math.min(k, p));
  return Xte.map((x) => top.reduce((s, f) => s + ((mu1[f.j] - mu0[f.j]) / f.s2) * (x[f.j] - (mu1[f.j] + mu0[f.j]) / 2), 0));
}

export function stratifiedFolds(y, k, r) {
  const folds = Array.from({ length: k }, () => []);
  for (const cls of [0, 1]) {
    const ids = shuffle(y.map((v, i) => (v === cls ? i : -1)).filter((i) => i >= 0), r);
    ids.forEach((i, n) => folds[n % k].push(i));
  }
  return folds;
}

export function batchFolds(batch) {
  const names = [...new Set(batch)];
  return names.map((b) => batch.map((v, i) => (v === b ? i : -1)).filter((i) => i >= 0));
}

/** Each batch's heldout samples share ONE fitted model, hence one score scale.
 * Cross-batch pooled ranks of independently fitted models are not a valid
 * substitute for within-heldout-batch AUC, especially for pure-label batches.
 * Pure groups have no AUROC by definition and are explicitly accounted for.
 */
export function macroBatchAuc(scores, y, batch) {
  if (scores.length !== y.length || batch.length !== y.length) {
    throw new Error('AUC scores, labels and batch IDs must have equal length');
  }
  const groups = new Map();
  for (let i = 0; i < batch.length; i++) {
    const group = groups.get(batch[i]) || [];
    group.push(i);
    groups.set(batch[i], group);
  }
  const perBatch = [...groups].map(([name, ids]) => ({
    batch: name, n: ids.length,
    positives: ids.filter((i) => y[i] === 1).length,
    auc: auc(ids.map((i) => scores[i]), ids.map((i) => y[i])),
  }));
  const mixed = perBatch.filter((group) => group.auc !== null);
  return { auc: mixed.length ? mean(mixed.map((group) => group.auc)) : null,
    nMixedHeldout: mixed.length, nPureHeldout: perBatch.length - mixed.length,
    perBatch };
}

/** The raw pooled AUC is preserved as an explicit *diagnostic only*. */
export function leaveBatchOutEvaluation(ds, k = 20) {
  const scores = oof(ds.y.length, batchFolds(ds.batch), modelScorer(ds.X, ds.y, k));
  return { ...macroBatchAuc(scores, ds.y, ds.batch),
    pooledOofAuc: auc(scores, ds.y) };
}


function pick(arr, ids) { return ids.map((i) => arr[i]); }

// Pooled out-of-fold scores for a scorer(trainIdx, testIdx) -> scores
export function oof(n, folds, scorer) {
  const out = new Array(n).fill(0);
  for (const test of folds) {
    const inTest = new Set(test);
    const train = [...Array(n).keys()].filter((i) => !inTest.has(i));
    scorer(train, test).forEach((s, m) => { out[test[m]] = s; });
  }
  return out;
}

function modelScorer(X, y, k) {
  return (train, test) => fitScore(pick(X, train), pick(y, train), pick(X, test), k);
}

function batchPrevalenceScorer(batch, y) {
  return (train, test) => {
    const c = new Map();
    let pos = 0;
    for (const i of train) { const e = c.get(batch[i]) || [0, 0]; e[0] += y[i]; e[1]++; c.set(batch[i], e); pos += y[i]; }
    const global = pos / train.length;
    return test.map((i) => { const e = c.get(batch[i]); return e ? e[0] / e[1] : global; });
  };
}

// Batch centering is part of the fitted preprocessing pipeline. The
// held-out fold never contributes a feature value to its training estimate.
// A previously unseen test batch is not centerable without transduction;
// fail closed rather than substituting a test-fold batch mean.
export function centerFoldByTrainingBatch(X, batch, train, test) {
  const p = X[0].length;
  const observed = new Map();
  for (const i of train) {
    let item = observed.get(batch[i]);
    if (!item) {
      item = { sum: new Float64Array(p), count: 0 };
      observed.set(batch[i], item);
    }
    item.count++;
    for (let j = 0; j < p; j++) item.sum[j] += X[i][j];
  }
  const missingBatches = [...new Set(test.map((i) => batch[i])
    .filter((b) => !observed.has(b)))].sort();
  if (missingBatches.length) return { trainX: null, testX: null, missingBatches };
  const center = (i) => {
    const item = observed.get(batch[i]);
    return X[i].map((v, j) => v - item.sum[j] / item.count);
  };
  return { trainX: train.map(center), testX: test.map(center), missingBatches: [] };
}

export function batchCenteredScorer(ds, k = 20) {
  return (train, test) => {
    const c = centerFoldByTrainingBatch(ds.X, ds.batch, train, test);
    if (c.missingBatches.length) {
      const err = new Error('Cannot center a held-out batch absent from training: '
        + c.missingBatches.join(', '));
      err.code = 'BATCH_NOT_IN_TRAIN';
      err.batches = c.missingBatches;
      throw err;
    }
    return fitScore(c.trainX, pick(ds.y, train), c.testX, k);
  };
}

export function mean(a) { return a.reduce((s, v) => s + v, 0) / a.length; }
export function quantile(a, q) { const s = a.slice().sort((x, y) => x - y); return s[Math.min(s.length - 1, Math.floor(q * s.length))]; }

export function designTable(ds) {
  const m = new Map();
  ds.batch.forEach((b, i) => { const e = m.get(b) || { batch: b, n: 0, pos: 0, neg: 0, samples: [] }; e.n++; ds.y[i] ? e.pos++ : e.neg++; e.samples.push({ i, id: ds.ids[i], y: ds.y[i] }); m.set(b, e); });
  return [...m.values()].map((e) => ({ ...e, mixed: e.pos > 0 && e.neg > 0 }));
}

export function repeatedCv(ds, scorerFactory, { seed, repeats = 3, folds = 5 }) {
  const r = rng(seed);
  const aucs = [];
  for (let rep = 0; rep < repeats; rep++) {
    const f = stratifiedFolds(ds.y, folds, r);
    const s = oof(ds.y.length, f, scorerFactory);
    aucs.push(auc(s, ds.y));
  }
  return mean(aucs.filter((a) => a !== null));
}

export const CONTROLS = {
  batch_only: { title: 'Batch-only baseline', question: 'Does the batch label alone, with no measurements, score as well as the model?' },
  within_batch_permutation: { title: 'Within-batch label permutation', question: 'Is the score higher than when labels are shuffled inside each batch and the whole pipeline is rerun?' },
  leave_batch_out: { title: 'Leave-one-batch-out', question: 'Does the score survive when an entire batch is held out?' },
  batch_residualized: { title: 'Batch-centered features', question: 'Is there signal left within mixed batches after each batch mean is removed?' },
};

export function runControls(ds, selected = Object.keys(CONTROLS), { seed = 20261030, perms = 100, k = 20 } = {}) {
  const n = ds.y.length;
  const table = designTable(ds);
  const mixedSet = new Set(table.filter((t) => t.mixed).map((t) => t.batch));
  const results = {};
  const naive = repeatedCv(ds, modelScorer(ds.X, ds.y, k), { seed });
  results.naive = { id: 'naive', title: 'Naive random-split CV', question: 'The headline number a shortcut can inflate.', auc: naive };

  if (selected.includes('batch_only')) {
    const a = repeatedCv(ds, batchPrevalenceScorer(ds.batch, ds.y), { seed: seed + 1 });
    const same = a >= naive - 0.05;
    results.batch_only = { id: 'batch_only', ...CONTROLS.batch_only, auc: a, status: same ? 'shortcut' : 'neutral',
      finding: same ? 'The batch label alone reproduces the model score. The measurements are not needed to reach it.' : 'The batch label alone scores clearly below the model.' };
  }
  if (selected.includes('within_batch_permutation')) {
    const r = rng(seed + 2);
    const nulls = [];
    for (let b = 0; b < perms; b++) {
      const y2 = ds.y.slice();
      for (const t of table) {
        const ids = t.samples.map((s) => s.i);
        const sh = shuffle(ids.map((i) => ds.y[i]), r);
        ids.forEach((i, m) => { y2[i] = sh[m]; });
      }
      nulls.push(repeatedCv({ ...ds, y: y2 }, modelScorer(ds.X, y2, k), { seed: seed + 100 + b, repeats: 1 }));
    }
    const ge = nulls.filter((v) => v >= naive).length;
    const p = (1 + ge) / (perms + 1);
    const beats = p <= 0.05;
    results.within_batch_permutation = { id: 'within_batch_permutation', ...CONTROLS.within_batch_permutation, auc: mean(nulls), nullMean: mean(nulls), nullLo: quantile(nulls, 0.025), nullHi: quantile(nulls, 0.975), p, perms,
      status: beats ? 'signal' : 'shortcut',
      finding: beats ? `The observed score is above the batch-preserving null (p = ${p.toFixed(3)}), so some of it depends on which sample carries which label inside a batch.`
        : `Shuffled labels reach a mean AUC of ${mean(nulls).toFixed(2)} and the observed score is not distinguishable from that null (p = ${p.toFixed(3)}). The batch structure alone explains it.` };
  }
  if (selected.includes('leave_batch_out')) {
    const summary = leaveBatchOutEvaluation(ds, k);
    const adequate = summary.nMixedHeldout >= 2;
    // One held-out mixed batch gives an AUC for that batch, not evidence of
    // transfer *across batches*. Keep its value in perBatch, abstain overall.
    const a = adequate ? summary.auc : null;
    const drop = a === null ? null : naive - a;
    const shortcut = drop !== null && drop >= 0.15;
    const finding = a === null
      ? `Only ${summary.nMixedHeldout} held-out batch(es) contain both labels; batch-transfer AUROC is not identifiable across multiple batches. ${summary.nPureHeldout} pure-label batch(es) cannot contribute AUROC.`
      : shortcut
        ? `Holding out whole batches drops mean within-mixed-batch AUC by ${drop.toFixed(2)} across ${summary.nMixedHeldout} evaluable batches; ${summary.nPureHeldout} pure-label batches excluded from AUC.`
        : `Mean within-mixed-batch AUC changes little across ${summary.nMixedHeldout} evaluable held-out batches; ${summary.nPureHeldout} pure-label batches excluded from AUC.`;
    results.leave_batch_out = { id: 'leave_batch_out', ...CONTROLS.leave_batch_out,
      auc: a, aggregation: 'mean_per_mixed_heldout_batch',
      nMixedHeldout: summary.nMixedHeldout, nPureHeldout: summary.nPureHeldout,
      perBatch: summary.perBatch, pooledOofAuc: summary.pooledOofAuc,
      status: shortcut ? 'shortcut' : 'neutral', finding };
  }
  if (selected.includes('batch_residualized')) {
    // Restrict the diagnostic to mixed-label batches as before; crucially,
    // learn every batch mean from the TRAINING fold only, just as feature
    // selection and LDA are already fitted inside the training fold.
    const keep = ds.batch.map((b, i) => (mixedSet.has(b) ? i : -1)).filter((i) => i >= 0);
    let a = null, unavailable = null;
    if (keep.length >= 10) {
      const sub = { X: pick(ds.X, keep), y: pick(ds.y, keep), batch: pick(ds.batch, keep) };
      try {
        a = repeatedCv(sub, batchCenteredScorer(sub, k), { seed: seed + 3 });
      } catch (error) {
        if (error?.code !== 'BATCH_NOT_IN_TRAIN') throw error;
        unavailable = error.batches;
      }
    }
    const survives = a !== null && a >= 0.65;
    const finding = a === null
      ? unavailable
        ? 'Train-only batch centering is not identifiable: a held-out fold includes batch '
          + unavailable.join(', ') + ' absent from its training fold.'
        : 'Too few samples sit in batches with both labels to test this.'
      : survives
        ? `With training-fold batch means removed and only mixed batches kept (${keep.length} samples), the score is ${a.toFixed(2)}. Discrimination survives this fold-local control, but biological causality is not established.`
        : `With training-fold batch means removed and only mixed batches kept (${keep.length} samples), the score is ${a.toFixed(2)}. Discrimination is weak under this control; this does not establish absence of biological signal.`;
    results.batch_residualized = { id: 'batch_residualized', ...CONTROLS.batch_residualized,
      auc: a, nUsed: keep.length, preprocessing: 'training_fold_only',
      status: a === null ? 'neutral' : survives ? 'signal' : 'shortcut',
      finding };
  }
  return { n, nBatches: table.length, mixedBatches: [...mixedSet], design: table.map(({ samples, ...t }) => t), results, verdict: verdict(results) };
}

export function verdict(res) {
  const shortcut = ['batch_only', 'leave_batch_out'].some((id) => res[id]?.status === 'shortcut') || (res.within_batch_permutation?.status === 'shortcut');
  const sigs = [res.within_batch_permutation, res.batch_residualized].filter(Boolean);
  const signal = sigs.length > 0 && sigs.every((r) => r.status === 'signal');
  const ran = Object.keys(res).length > 1;
  if (!ran) return { id: 'unrun', label: 'No controls run', text: 'Select at least one control.' };
  if (shortcut && !signal) return { id: 'shortcut', label: 'Mostly shortcut', text: 'The headline score is explained by batch structure. The controls found no evidence of a within-batch signal.' };
  if (shortcut && signal) return { id: 'both', label: 'Signal plus shortcut', text: 'Part of the score is real within-batch signal; part is batch structure. Report the controlled number, not the naive one.' };
  if (signal) return { id: 'signal', label: 'Signal holds up', text: 'The score exceeds every batch-preserving control that was run.' };
  return { id: 'open', label: 'Inconclusive', text: 'No control flagged a shortcut and none showed signal beyond the null. Add controls or samples.' };
}

export function toMarkdown(name, run, narrative) {
  const L = [`# NULLGARDEN control report: ${name}`, '', `Verdict: **${run.verdict.label}**. ${run.verdict.text}`, '', `${run.n} samples, ${run.nBatches} batches, ${run.mixedBatches.length} with both labels.`, '',
    '| Control | AUC | Status |', '|---|---|---|'];
  for (const r of Object.values(run.results)) L.push(`| ${r.title} | ${r.auc === null ? 'n/a' : r.auc.toFixed(3)} | ${r.status || 'baseline'} |`);
  L.push('', '## Findings');
  for (const r of Object.values(run.results)) if (r.finding) L.push(`- ${r.title}: ${r.finding}`);
  if (narrative) L.push('', '## Nemotron explanation (figures checked against the table above)', '', narrative);
  L.push('', '## Corrected evaluation candidate', '', 'Report mean per-mixed-heldout-batch AUC (pure-label heldouts excluded; at least two mixed groups needed) and training-fold-only batch-centered AUC next to the batch-only baseline. Scores pooled across separately fitted heldout-batch models are only diagnostic, not cross-batch discrimination evidence. Treat only the amount above the within-batch permutation null as tentative evidence of learned signal.');
  return L.join('\n') + '\n';
}
