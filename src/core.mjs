import { conditionalOrbitSize, conditionalLabelAssignments } from './exact-orbit.mjs';
import { unitGroups, stratifiedUnitFolds, unitPermutationDesign } from './units.mjs';
export { unitGroups, stratifiedUnitFolds, unitPermutationDesign } from './units.mjs';
export { conditionalOrbitSize, conditionalLabelAssignments } from './exact-orbit.mjs';
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

function numericalFailure(detail) {
  const error = new RangeError('Unrepresentable normalized range: ' + detail
    + '. Rescale measurements or exclude this unsupported magnitude.');
  error.code = 'NUMERIC_RANGE_UNSUPPORTED';
  return error;
}

/** Independent-unit AUROC: repeated measurements from one subject count as
 * ONE prediction, the arithmetic mean of its heldout scores. This leaves
 * already-independent datasets unchanged, and never uses test labels when
 * aggregating technical replicates.
 */
export function unitAuc(scores, y, unit) {
  if (!unit) return auc(scores, y);
  if (scores.length !== y.length || y.length !== unit.length) {
    throw new RangeError('Unit AUC requires aligned scores, labels and unit IDs');
  }
  const map = new Map();
  for (let i = 0; i < scores.length; i++) {
    const g = map.get(unit[i]);
    if (g) {
      if (g.label !== y[i]) throw new Error('Unit AUC cannot combine inconsistent labels within the same independent unit');
      g.sum += scores[i]; g.n++;
    } else map.set(unit[i], {label:y[i],sum:scores[i],n:1});
  }
  const units = [...map.values()];
  return auc(units.map(g=>g.sum/g.n), units.map(g=>g.label));
}

export function auc(scores, y) {
  if (scores.length !== y.length || scores.some((s) => !Number.isFinite(s))) {
    throw new RangeError('AUROC requires matching labels and non-finite-free evidence scores');
  }
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

// The pipeline under test: top-k feature selection by |t|, then diagonal-LDA.
// Every feature is normalized by its max absolute TRAINING-fold observation,
// before class moments and variance are fitted. This makes the regularizer
// dimensionless and prevents overflow for finite but large measurements,
// while leaving both selection and normalization independent of heldout rows.
export function fitScore(Xtr, ytr, Xte, k = 20) {
  if (!Array.isArray(Xtr) || !Xtr.length || !Array.isArray(ytr)
      || Xtr.length !== ytr.length || !Array.isArray(Xte)) {
    throw new TypeError('fitScore requires aligned nonempty training and test matrices');
  }
  const p = Xtr[0].length;
  if (!Number.isInteger(p) || p < 1 || !Number.isInteger(k) || k < 1) {
    throw new TypeError('fitScore requires positive feature count and top-k');
  }
  const scales = new Float64Array(p);
  for (const [i, x] of Xtr.entries()) {
    if (!Array.isArray(x) || x.length !== p || ![0, 1].includes(ytr[i])) {
      throw new TypeError('fitScore training matrix or labels are malformed');
    }
    for (let j = 0; j < p; j++) {
      if (!Number.isFinite(x[j])) throw numericalFailure('non-finite training feature ' + j);
      scales[j] = Math.max(scales[j], Math.abs(x[j]));
    }
  }
  for (let j = 0; j < p; j++) if (scales[j] === 0) scales[j] = 1;
  for (const x of Xte) {
    if (!Array.isArray(x) || x.length !== p || !x.every(Number.isFinite)) {
      throw numericalFailure('non-finite or malformed heldout feature vector');
    }
  }
  const n1 = ytr.filter((v) => v === 1).length, n0 = ytr.length - n1;
  if (n1 === 0 || n0 === 0) {
    const error = new Error('Training fold has no samples from one class; prediction is unidentifiable');
    error.code = 'CV_TRAIN_CLASS_MISSING';
    error.classCounts = { positive: n1, negative: n0 };
    throw error;
  }

  const mu1 = new Float64Array(p), mu0 = new Float64Array(p),
    variance = new Float64Array(p);
  for (let i = 0; i < Xtr.length; i++) {
    for (let j = 0; j < p; j++) {
      (ytr[i] ? mu1 : mu0)[j] += Xtr[i][j] / scales[j];
    }
  }
  for (let j = 0; j < p; j++) { mu1[j] /= n1; mu0[j] /= n0; }
  for (let i = 0; i < Xtr.length; i++) {
    for (let j = 0; j < p; j++) {
      const d = Xtr[i][j] / scales[j] - (ytr[i] ? mu1[j] : mu0[j]);
      variance[j] += d * d;
    }
  }
  const t = [];
  for (let j = 0; j < p; j++) {
    const s2 = variance[j] / Math.max(1, Xtr.length - 2) + 1e-9;
    const difference = mu1[j] - mu0[j];
    const statistic = difference / Math.sqrt(s2 * (1 / n1 + 1 / n0));
    if (!Number.isFinite(statistic)) throw numericalFailure('feature ' + j + ' statistic');
    t.push({ j, t: statistic, difference, midpoint: (mu1[j] + mu0[j]) / 2,
      weight: difference / s2 });
  }
  t.sort((a, b) => Math.abs(b.t) - Math.abs(a.t) || a.j - b.j);
  const top = t.slice(0, Math.min(k, p));
  return Xte.map((x) => {
    let score = 0;
    for (const feature of top) {
      if (feature.weight === 0) continue; // constant feature is strictly neutral.
      const centered = x[feature.j] / scales[feature.j] - feature.midpoint;
      const term = feature.weight * centered;
      if (!Number.isFinite(term)) throw numericalFailure('heldout feature ' + feature.j);
      score += term;
      if (!Number.isFinite(score)) throw numericalFailure('heldout discriminant sum');
    }
    return score;
  });
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
export function macroBatchAuc(scores, y, batch, unit) {
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
    auc: unitAuc(ids.map((i) => scores[i]), ids.map((i) => y[i]),
      unit && ids.map(i=>unit[i])),
  }));
  const mixed = perBatch.filter((group) => group.auc !== null);
  return { auc: mixed.length ? mean(mixed.map((group) => group.auc)) : null,
    nMixedHeldout: mixed.length, nPureHeldout: perBatch.length - mixed.length,
    perBatch };
}

/** The raw pooled AUC is preserved as an explicit *diagnostic only*. */
export function leaveBatchOutEvaluation(ds, k = 20) {
  const n = ds.y.length, all = [...Array(n).keys()];
  const scores = new Array(n).fill(null);
  const perBatch = [];
  for (const test of batchFolds(ds.batch)) {
    const tested = new Set(test);
    const train = all.filter((i) => !tested.has(i));
    const trainingY = pick(ds.y, train);
    const positiveTraining = trainingY.filter((v) => v === 1).length;
    const negativeTraining = train.length - positiveTraining;
    const trainSupported = positiveTraining > 0 && negativeTraining > 0;
    const positives = test.filter((i) => ds.y[i] === 1).length;
    if (!trainSupported) {
      perBatch.push({ batch: ds.batch[test[0]], n: test.length, positives,
        auc: null, trainSupported: false,
        reason: 'Heldout batch leaves training without both classes' });
      continue;
    }
    const fitted = fitScore(pick(ds.X, train), trainingY, pick(ds.X, test), k);
    fitted.forEach((v, j) => { scores[test[j]] = v; });
    perBatch.push({ batch: ds.batch[test[0]], n: test.length, positives,
      auc: unitAuc(fitted, pick(ds.y, test),
        ds.unit && pick(ds.unit, test)), trainSupported: true });
  }
  const valid = perBatch.filter((row) => row.trainSupported && row.auc !== null);
  const nUnsupportedTrain = perBatch.filter((row) => !row.trainSupported).length;
  return {
    auc: valid.length ? mean(valid.map((row) => row.auc)) : null,
    nMixedHeldout: valid.length,
    nPureHeldout: perBatch.length - valid.length - nUnsupportedTrain,
    nUnsupportedTrain,
    perBatch,
    // Missing one prediction invalidates the pooled cross-model diagnostic,
    // never substitute a constant or compare missing scores as zeros.
    pooledOofAuc: nUnsupportedTrain ? null : unitAuc(scores, ds.y, ds.unit),
  };
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

function batchPrevalenceScorer(batch, y, unit) {
  return (train, test) => {
    const c = new Map();
    let pos = 0, nUnits = 0;
    const counted = new Set();
    for (const i of train) {
      // Training folds include complete units. Count each independent donor
      // once; a donor with many technical replicates gets no extra vote.
      if (unit && counted.has(unit[i])) continue;
      if (unit) counted.add(unit[i]);
      const e = c.get(batch[i]) || [0, 0];
      e[0] += y[i]; e[1]++; c.set(batch[i], e);
      pos += y[i]; nUnits++;
    }
    const global = pos / nUnits;
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
  // Accumulate training-batch normalized moments. Raw sums can exceed the
  // finite IEEE range even when each measurement and batch mean is finite.
  // The first pass learns scales from TRAINING rows only, preventing leakage.
  for (const i of train) {
    let item = observed.get(batch[i]);
    if (!item) {
      item = { scale: new Float64Array(p), sum: new Float64Array(p), count: 0 };
      observed.set(batch[i], item);
    }
    item.count++;
    for (let j = 0; j < p; j++) {
      const value = X[i][j];
      if (!Number.isFinite(value)) throw numericalFailure('batch training feature ' + j);
      item.scale[j] = Math.max(item.scale[j], Math.abs(value));
    }
  }
  const missingBatches = [...new Set(test.map((i) => batch[i])
    .filter((b) => !observed.has(b)))].sort();
  if (missingBatches.length) return { trainX: null, testX: null, missingBatches };
  for (const item of observed.values()) {
    for (let j = 0; j < p; j++) if (item.scale[j] === 0) item.scale[j] = 1;
  }
  for (const item of observed.values()) item.mean = new Float64Array(p);
  for (const i of train) {
    const item = observed.get(batch[i]);
    for (let j = 0; j < p; j++) item.sum[j] += X[i][j] / item.scale[j];
  }
  for (const item of observed.values()) {
    for (let j = 0; j < p; j++) {
      item.mean[j] = (item.sum[j] / item.count) * item.scale[j];
      if (!Number.isFinite(item.mean[j])) {
        throw numericalFailure('batch training mean feature ' + j);
      }
    }
  }
  const center = (i) => {
    const item = observed.get(batch[i]);
    return X[i].map((value, j) => {
      if (!Number.isFinite(value)) throw numericalFailure('batch heldout feature ' + j);
      // Reconstruct a stable finite training mean BEFORE subtraction.
      // This preserves exact binary-integer offsets for ordinary inputs,
      // without amplifying normalized cancellation when multiplying back.
      const corrected = value - item.mean[j];
      if (!Number.isFinite(corrected)) {
        throw numericalFailure('batch-heldout centered feature ' + j);
      }
      return corrected;
    });
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
  return [...m.values()].map((e) => ({
    ...e, nUnits: new Set(e.samples.map(row => ds.unit?.[row.i] ?? row.i)).size,
    mixed: e.pos > 0 && e.neg > 0,
  }));
}

export function repeatedCv(ds, scorerFactory, { seed, repeats = 3, folds = 5 }) {
  if (!Number.isInteger(folds) || folds < 2 || folds > ds.y.length
      || !Number.isInteger(repeats) || repeats < 1) {
    throw new RangeError('Cross validation requires 2..n folds and at least one repeat');
  }
  const r = rng(seed);
  const aucs = [];
  for (let rep = 0; rep < repeats; rep++) {
    const f = ds.unit
      ? stratifiedUnitFolds(ds.y, ds.unit, folds, r)
      : stratifiedFolds(ds.y, folds, r);
    // A fold with a missing training class cannot fit the requested
    // binary classifier. Refuse a spurious score instead of the historical
    // all-zero prediction, whose AUROC could be mistaken for evidence.
    if (f.some((test) => {
      const inTest = new Set(test);
      let nPositive = 0, nNegative = 0;
      ds.y.forEach((label, i) => {
        if (inTest.has(i)) return;
        if (label === 1) nPositive++;
        else if (label === 0) nNegative++;
      });
      return nPositive === 0 || nNegative === 0;
    })) return null;
    // Different heldout folds use DIFFERENT fitted models. Pooling their
    // raw scores ranks unrelated model intercepts/scales against one another
    // and can turn perfectly discriminating within-fold predictions into
    // an arbitrarily poor pooled AUC. Evaluate only within each heldout fold.
    const foldAucs = [];
    for (const test of f) {
      const tested = new Set(test);
      const train = ds.y.map((_, i) => i).filter((i) => !tested.has(i));
      const testLabels = pick(ds.y, test);
      // Unlike pooled OOF rankings, fold AUC has no definition when the
      // heldout fold is pure. The entire requested repeated-CV estimand
      // abstains rather than silently discarding or inventing those pairs.
      if (!testLabels.includes(0) || !testLabels.includes(1)) return null;
      const value = unitAuc(scorerFactory(train, test), testLabels,
        ds.unit && pick(ds.unit, test));
      if (value === null) return null;
      foldAucs.push(value);
    }
    aucs.push(mean(foldAucs));
  }
  return mean(aucs);
}

export const CONTROLS = {
  batch_only: { title: 'Batch-only baseline', question: 'Does the batch label alone, with no measurements, score as well as the model?' },
  within_batch_permutation: { title: 'Within-batch label permutation', question: 'Is the score higher than when labels are shuffled inside each batch and the whole pipeline is rerun?' },
  leave_batch_out: { title: 'Leave-one-batch-out', question: 'Does the score survive when an entire batch is held out?' },
  batch_residualized: { title: 'Batch-centered features', question: 'Is there signal left within mixed batches after each batch mean is removed?' },
};

/** For the predeclared fixed-budget B permutation test, the final exact
 * p=(1+L_B)/(B+1) lies in [lower,upper] after t draws, with L_t exceedances.
 * Stopping when either bound establishes the final alpha decision is
 * pathwise identical to completing B draws, for EVERY unseen suffix.
 * Since upper>=final p on every path, upper is an anytime-valid conservative
 * p-process (fixed B, no budget switching). This is not the exact final p
 * unless t=B, nor an anytime process across changed budgets/hypotheses.
 */
export function permutationDecisionBounds(losses, drawn, budget, alpha = 0.05) {
  if (!Number.isInteger(budget) || budget < 1
      || !Number.isInteger(drawn) || drawn < 0 || drawn > budget
      || !Number.isInteger(losses) || losses < 0 || losses > drawn
      || !Number.isFinite(alpha) || alpha <= 0 || alpha >= 1) {
    throw new RangeError('Permutation decision bounds require 0<=losses<=drawn<=budget, budget>=1 and 0<alpha<1');
  }
  const lower = (1 + losses) / (budget + 1);
  const upper = (1 + losses + budget - drawn) / (budget + 1);
  return {lower, upper, decision: upper <= alpha ? 'reject'
    : lower > alpha ? 'retain' : 'continue'};
}

export function runControls(ds, selected = Object.keys(CONTROLS),
  { seed = 20261030, perms = 100, k = 20, stopping = 'fixed' } = {}) {
  if (!['fixed', 'decision'].includes(stopping)) {
    throw new RangeError('Permutation stopping must be fixed or decision with a predeclared budget');
  }
  const n = ds.y.length;
  const permutationUnits = unitPermutationDesign(ds);
  const nUnits = permutationUnits.groups.length;
  const permutationBlocks = new Map();
  for (const group of permutationUnits.groups) {
    if (!permutationBlocks.has(group.batch)) permutationBlocks.set(group.batch, []);
    permutationBlocks.get(group.batch).push(group);
  }
  const table = designTable(ds);
  const mixedSet = new Set(table.filter((t) => t.mixed).map((t) => t.batch));
  const results = {};
  const naive = repeatedCv(ds, modelScorer(ds.X, ds.y, k), { seed });
  results.naive = { id: 'naive', title: 'Naive random-split CV',
    question: 'The headline number a shortcut can inflate.', auc: naive,
    aggregation: ds.unit ? 'mean_per_heldout_fold_and_unit' : 'mean_per_heldout_fold',
    status: naive === null ? 'unavailable' : 'baseline',
    finding: naive === null
      ? 'Training folds lose one class; the headline AUC is unidentifiable, not chance performance.'
      : undefined };
  if (selected.includes('batch_only')) {
    const a = repeatedCv(ds, batchPrevalenceScorer(ds.batch, ds.y, ds.unit), { seed: seed + 1 });
    const identifiable = a !== null && naive !== null;
    const same = identifiable && a >= naive - 0.05;
    results.batch_only = { id: 'batch_only', ...CONTROLS.batch_only,
      auc: identifiable ? a : null,
      aggregation: ds.unit ? 'mean_per_heldout_fold_and_unit' : 'mean_per_heldout_fold',
      status: identifiable ? (same ? 'shortcut' : 'neutral') : 'unavailable',
      finding: !identifiable
        ? 'Class support in at least one training fold is insufficient for a batch-only comparison.'
        : same
          ? 'The batch label alone reproduces the model score. The measurements are not needed to reach it.'
          : 'The batch label alone scores clearly below the model.' };
  }
  if (selected.includes('within_batch_permutation')) {
    // Exact enumeration is preferable to Monte Carlo when the complete
    // batch-conditional orbit fits a deterministic work budget. Saturated
    // combinatorial counts avoid factorial overflow on uploaded datasets.
    const exactMax = Math.min(512,
      Math.floor(25000000 / Math.max(1, n * ds.X[0].length * 15)));
    const orbitSize = exactMax >= 1 && naive !== null
      && Number.isInteger(perms) && perms >= 1
      ? conditionalOrbitSize(permutationUnits.batch, permutationUnits.y, exactMax) : null;
    if (orbitSize !== null && orbitSize <= exactMax) {
      const nulls = [];
      let observedAllocations = 0;
      for (const unitLabels of conditionalLabelAssignments(
        permutationUnits.batch, permutationUnits.y, {maxAssignments: exactMax})) {
        if (unitLabels.every((label, i) => label === permutationUnits.y[i])) observedAllocations++;
        const labels = permutationUnits.expand(unitLabels);
        const score = repeatedCv({...ds, y: labels},
          modelScorer(ds.X, labels, k), {seed, repeats: 3});
        nulls.push(score);
        if (score === null || !Number.isFinite(score)) break;
      }
      const valid = nulls.length === orbitSize && observedAllocations === 1
        && nulls.every((v) => v !== null && Number.isFinite(v));
      const metadata = {
        perms, permsBudget: orbitSize, permsRun: nulls.length,
        orbitSize, pExact: valid, stopping: 'exact',
        permutationUnit: ds.unit ? 'unit' : 'sample', nUnits,
        statisticRepeats: 3, statisticSeed: seed,
        statisticAggregation: 'mean_per_heldout_fold',
        nullSampling: 'complete_batch_conditional_orbit',
        inferenceLimit: 'Exact enumeration holds feature data, batch memberships and within-batch positive counts fixed and treats every distinct allowed label assignment equally. Validity requires within-batch exchangeability and a predeclared statistic and hypothesis; it does not prove biological causation or correct selection across multiple hypotheses.',
      };
      if (!valid) {
        results.within_batch_permutation = {
          id: 'within_batch_permutation', ...CONTROLS.within_batch_permutation,
          ...metadata, auc: null, p: null, pLower: null, pUpper: null,
          status: 'unavailable',
          finding: 'The complete conditional orbit includes an assignment with undefined heldout-fold AUROC; exact inference abstains rather than deleting unsupported assignments.',
        };
      } else {
        const exceedances = nulls.filter((v) => v >= naive).length;
        if (exceedances < 1) {
          throw new Error('Exact conditional-orbit invariant failed: observed allocation must contribute at least one tied/exceeding statistic');
        }
        const p = exceedances / orbitSize;
        const nullMean = mean(nulls);
        results.within_batch_permutation = {
          id: 'within_batch_permutation', ...CONTROLS.within_batch_permutation,
          ...metadata, auc: nullMean, nullMean,
          nullLo: quantile(nulls, 0.025), nullHi: quantile(nulls, 0.975),
          p, pLower: p, pUpper: p,
          status: p <= 0.05 ? 'signal' : 'neutral',
          finding: 'Complete exact conditional randomization over '
            + orbitSize + ' distinct batch-preserving label assignments yields p = '
            + p.toFixed(4) + ' (all ties counted). '
            + (p <= 0.05
              ? 'The observed foldwise score is unusually high under this exact conditional null.'
              : 'No significant within-batch discrimination is established; this does not prove a shortcut.'),
        };
      }
    } else {
      const r = rng(seed + 2);
      const nulls = [];
      let prefix = null;
      if (naive !== null && Number.isInteger(perms) && perms >= 1) {
        let exceedances = 0;
        for (let b = 0; b < perms; b++) {
          const y2 = ds.y.slice();
          for (const blocks of permutationBlocks.values()) {
            const relabeled = shuffle(blocks.map((g) => g.label), r);
            blocks.forEach((group, j) => {
              for (const i of group.indices) y2[i] = relabeled[j];
            });
          }
          const value = repeatedCv({ ...ds, y: y2 }, modelScorer(ds.X, y2, k),
            // The observed and shuffled labels share the exact same frozen
            // three-repeat foldwise AUC statistic and CV seed.
            { seed, repeats: 3 });
          nulls.push(value);
          if (value === null || !Number.isFinite(value)) break;
          if (value >= naive) exceedances++;
          prefix = permutationDecisionBounds(exceedances, nulls.length, perms);
          // Two monotone bounds prove the final B-permutation alpha-decision
          // despite any unseen continuation, not just in expectation.
          if (stopping === 'decision' && prefix.decision !== 'continue') break;
        }
      }
      const complete = nulls.length === perms;
      const inferenceLimit = 'Requires label exchangeability within each batch and a predeclared statistic, seed, permutation budget and hypothesis. Decision stopping within that one fixed budget is anytime-valid for the reported conservative upper p bound; changing the budget, rerunning selected seeds or post-hoc choosing controls is not covered.';
      const valid = naive !== null && nulls.length > 0
        && nulls.every((v) => v !== null && Number.isFinite(v))
        && (complete || (stopping === 'decision'
          && prefix && prefix.decision !== 'continue'));
      const metadata = {
        perms, permsBudget: perms, permsRun: nulls.length,
        pLower: valid ? prefix.lower : null, pUpper: valid ? prefix.upper : null,
        pExact: valid && complete, stopping,
        permutationUnit: ds.unit ? 'unit' : 'sample', nUnits,
        statisticRepeats: 3, statisticSeed: seed,
        statisticAggregation: 'mean_per_heldout_fold',
        nullSampling: 'batch_conditional_fixed_statistic', inferenceLimit,
      };
      if (!valid) {
        results.within_batch_permutation = { id: 'within_batch_permutation',
          ...CONTROLS.within_batch_permutation, ...metadata,
          auc: null, p: null, status: 'unavailable',
          finding: 'A valid conditional permutation comparison requires identifiable two-class heldout folds for the observation and every sampled relabeling.' };
      } else {
        // The right bound is a valid conservative anytime p-process because
        // it bounds the fixed-B exact Monte Carlo p on EVERY possible suffix.
        const p = prefix.upper;
        const beats = p <= 0.05;
        // A sequentially selected prefix is not an unbiased sample for
        // displaying the full fixed-budget null distribution's mean/band.
        const nullMean = complete ? mean(nulls) : null;
        const finding = complete
          ? (beats
            ? 'The observed score is above the batch-preserving conditional null (exact finite-budget p = '
              + p.toFixed(3) + '); within-batch label assignments matter under the exchangeability assumption.'
            : 'The observed score is not distinguishable from the conditional null (exact finite-budget p = '
              + p.toFixed(3) + '); nonrejection does not prove that batch structure explains the result.')
          : (beats
            ? 'Anytime-valid conservative upper p bound = ' + p.toFixed(3)
              + ' after ' + nulls.length + '/' + perms + ' planned shuffles. Every possible unseen suffix still rejects at 0.05; the unobserved exact p is only bounded.'
            : 'Stopped after ' + nulls.length + '/' + perms + ' planned shuffles: even the smallest possible final p = '
              + prefix.lower.toFixed(3) + ' exceeds 0.05. The exact final p was not estimated; the reported conservative anytime upper p bound is '
              + p.toFixed(3) + '. This is nonrejection, not affirmative evidence of a shortcut.');
        results.within_batch_permutation = {
          id: 'within_batch_permutation', ...CONTROLS.within_batch_permutation,
          ...metadata, auc: nullMean, nullMean,
          ...(complete ? {nullLo: quantile(nulls, 0.025),
            nullHi: quantile(nulls, 0.975)} : {}),
          p, status: beats ? 'signal' : 'neutral', finding,
        };
      }
    }
  }
  if (selected.includes('leave_batch_out')) {
    const summary = leaveBatchOutEvaluation(ds, k);
    const adequate = summary.nMixedHeldout >= 2 && naive !== null;
    const a = adequate ? summary.auc : null;
    const drop = a === null ? null : naive - a;
    const shortcut = drop !== null && drop >= 0.15;
    let finding;
    if (summary.nUnsupportedTrain) {
      finding = summary.nUnsupportedTrain
        + ' held-out batch(es) leave a single-class training set; these predictions are not identifiable. '
        + summary.nMixedHeldout + ' held-out batches remain evaluable with both test labels.';
    } else if (naive === null) {
      finding = 'Class support is insufficient in the random-split baseline; no valid cross-validation drop can be calculated.';
    } else if (a === null) {
      finding = 'Only ' + summary.nMixedHeldout
        + ' held-out batch(es) contain both labels; batch-transfer AUC is not identifiable across multiple batches. '
        + summary.nPureHeldout + ' pure-label batch(es) cannot contribute AUC.';
    } else if (shortcut) {
      finding = 'Holding out whole batches drops mean within-mixed-batch AUC by '
        + drop.toFixed(2) + ' across ' + summary.nMixedHeldout
        + ' evaluable batches; ' + summary.nPureHeldout + ' pure-label batches excluded from AUC.';
    } else {
      finding = 'Mean within-mixed-batch AUC changes little across '
        + summary.nMixedHeldout + ' evaluable held-out batches; '
        + summary.nPureHeldout + ' pure-label batches excluded from AUC.';
    }
    results.leave_batch_out = { id: 'leave_batch_out', ...CONTROLS.leave_batch_out,
      auc: a, aggregation: ds.unit
        ? 'mean_per_mixed_heldout_batch_and_unit'
        : 'mean_per_mixed_heldout_batch',
      nMixedHeldout: summary.nMixedHeldout, nPureHeldout: summary.nPureHeldout,
      nUnsupportedTrain: summary.nUnsupportedTrain,
      perBatch: summary.perBatch, pooledOofAuc: summary.pooledOofAuc,
      status: a === null ? 'unavailable' : shortcut ? 'shortcut' : 'neutral',
      finding };
  }
  if (selected.includes('batch_residualized')) {
    // Restrict the diagnostic to mixed-label batches as before; crucially,
    // learn every batch mean from the TRAINING fold only, just as feature
    // selection and LDA are already fitted inside the training fold.
    const keep = ds.batch.map((b, i) => (mixedSet.has(b) ? i : -1)).filter((i) => i >= 0);
    let a = null, unavailable = null;
    if (keep.length >= 10) {
      const sub = { X: pick(ds.X, keep), y: pick(ds.y, keep), batch: pick(ds.batch, keep),
        ...(ds.unit ? {unit:pick(ds.unit,keep)} : {}) };
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
        : keep.length < 10
          ? 'Too few samples sit in batches with both labels to test this.'
          : 'At least one centered cross-validation training fold lacks both classes; its AUC is unidentifiable.'
      : survives
        ? `With training-fold batch means removed and only mixed batches kept (${keep.length} samples), the score is ${a.toFixed(2)}. Discrimination survives this fold-local control, but biological causality is not established.`
        : `With training-fold batch means removed and only mixed batches kept (${keep.length} samples), the score is ${a.toFixed(2)}. Discrimination is weak under this control; this does not establish absence of biological signal.`;
    results.batch_residualized = { id: 'batch_residualized', ...CONTROLS.batch_residualized,
      auc: a, nUsed: keep.length, preprocessing: 'training_fold_only',
      aggregation: ds.unit ? 'mean_per_heldout_fold_and_unit' : 'mean_per_heldout_fold',
      status: a === null ? 'unavailable' : survives ? 'signal' : 'shortcut',
      finding };
  }
  return { n, nUnits, permutationUnit: ds.unit ? 'unit' : 'sample',
    nBatches: table.length, mixedBatches: [...mixedSet],
    design: table.map(({ samples, ...t }) => t), results, verdict: verdict(results) };
}

export function verdict(res) {
  const shortcut = ['batch_only', 'leave_batch_out'].some((id) => res[id]?.status === 'shortcut');
  // One favorable control is useful evidence, not independent corroboration:
  // absolute residualized AUC alone is not a permutation-test p-value, and
  // permutation significance alone does not verify a corrected batch pathway.
  const permutationSignal = res.within_batch_permutation?.status === 'signal';
  const residualSignal = res.batch_residualized?.status === 'signal';
  const signal = permutationSignal && residualSignal;
  const partialSignal = permutationSignal || residualSignal;
  const ran = Object.keys(res).length > 1;
  if (!ran) return { id: 'unrun', label: 'No controls run', text: 'Select at least one control.' };
  if (shortcut && !signal) return { id: 'shortcut', label: 'Mostly shortcut', text: partialSignal
    ? 'A shortcut was detected. One within-batch signal control is favorable, but the complementary second control has not corroborated it.'
    : 'Batch-only or batch-transfer controls indicate shortcut sensitivity. No corroborated within-batch signal was established; absence of significance is not proof of no signal.' };
  if (shortcut && signal) return { id: 'both', label: 'Signal plus shortcut', text: 'Both complementary within-batch signal controls are favorable, while batch structure also explains part of the headline score. Report the controlled number, not the naive one.' };
  if (signal) return { id: 'signal', label: 'Signal holds up', text: 'Both the batch-preserving permutation null and train-fold-only batch centering support within-batch discrimination. This is evidence, not proof of biological causality.' };
  if (partialSignal) return { id: 'open', label: 'Incomplete evidence', text: 'Only one signal control is favorable; both the batch-preserving permutation null and training-fold batch centering must support a result before calling it robust.' };
  return { id: 'open', label: 'Inconclusive', text: 'No control flagged a shortcut and none showed corroborated signal beyond the null. Add controls or samples.' };
}

export function toMarkdown(name, run, narrative) {
  const L = [`# NULLGARDEN control report: ${name}`, '', `Verdict: **${run.verdict.label}**. ${run.verdict.text}`, '',
    `${run.n} samples, ${run.nUnits} independent ${run.permutationUnit === 'unit' ? 'units (unit_id: grouped CV and whole-unit shuffling)' : 'sample units'}, ${run.nBatches} batches, ${run.mixedBatches.length} with both labels.`, '',
    '| Control | AUC | Status |', '|---|---|---|'];
  for (const r of Object.values(run.results)) L.push(`| ${r.title} | ${r.auc === null ? 'n/a' : r.auc.toFixed(3)} | ${r.status || 'baseline'} |`);
  L.push('', '## Cross-validation estimand', '',
    run.permutationUnit === 'unit'
      ? 'Repeated measurements of one independent unit are kept in the same training/heldout fold. Their heldout scores are averaged into one unit score, then mean within-fold unit-level AUROC is computed; each mixed-label heldout fold is averaged across folds and repeats. No row from the heldout unit appears in training. A pure-label heldout-unit fold abstains.'
      : 'Random stratified CV reports mean within-fold AUROC: each mixed-label heldout fold is scored separately, then averaged over folds and repeats. Raw prediction ranks from different fitted models must not be pooled; if any requested test fold is pure-label, its AUROC is undefined and the aggregate abstains.', '', '## Findings');
  for (const r of Object.values(run.results)) if (r.finding) L.push(`- ${r.title}: ${r.finding}`);
  const permutation = run.results.within_batch_permutation;
  if (permutation) L.push('', '## Permutation inference limits', '',
    permutation.inferenceLimit,
    `Statistic: ${permutation.statisticRepeats} CV repeats, seed ${permutation.statisticSeed}; ${permutation.nullSampling === 'complete_batch_conditional_orbit'
      ? 'all ' + permutation.orbitSize + ' distinct conditional label assignments evaluated exactly'
      : permutation.permsBudget + ' predeclared Monte Carlo shuffles, ' + permutation.permsRun + ' actually evaluated (' + permutation.stopping + ' stopping)'}. The null uses the same statistic and CV randomization as the observed dataset.`,
    permutation.p === null ? 'No permutation inference was identifiable.'
      : permutation.nullSampling === 'complete_batch_conditional_orbit'
        ? `Exact conditional-orbit p: ${permutation.p.toFixed(4)} across all ${permutation.orbitSize} permitted label allocations; the observed allocation is included once and ties count.`
        : permutation.pExact ? `Exact complete-Monte-Carlo-budget p: ${permutation.p.toFixed(4)}; this is NOT an enumeration of the full conditional label orbit.`
        : `Anytime-valid conservative p-value bound: ${permutation.p.toFixed(4)}. The unknown completed-budget p is in [${permutation.pLower.toFixed(4)}, ${permutation.pUpper.toFixed(4)}]; do not report a selected-prefix null mean or treat nonrejection as affirmative evidence.`);
  if (narrative) L.push('', '## Nemotron explanation (figures checked against the table above)', '', narrative);
  L.push('', '## Corrected evaluation candidate', '', 'Report mean per-mixed-heldout-batch AUC (pure-label heldouts excluded; at least two mixed groups needed) and training-fold-only batch-centered AUC next to the batch-only baseline. Scores pooled across separately fitted heldout-batch models are only diagnostic, not cross-batch discrimination evidence. Treat only the amount above the within-batch permutation null as tentative evidence of learned signal.');
  return L.join('\n') + '\n';
}
