import { runControls } from './core.mjs';

// Holm's step-down correction controls FWER under arbitrary dependency when
// every preregistered marginal p is superuniform. Conservative anytime-fixed-B
// upper p bounds from NULLGARDEN preserve that condition. No independence
// assumption between candidate models or their shared permutations is needed.
export function holmAdjust(pValues) {
  if (!Array.isArray(pValues) || !pValues.length || pValues.some(p =>
    typeof p !== 'number' || !Number.isFinite(p) || p < 0 || p > 1)) {
    throw new RangeError('Holm correction requires a nonempty fixed family of finite p-values in [0,1]');
  }
  const ranked = pValues.map((p,i) => ({p,i})).sort((a,b) => a.p-b.p || a.i-b.i);
  const adjusted = Array(pValues.length);
  let bound = 0;
  for (let rank=0; rank<ranked.length; rank++) {
    bound = Math.max(bound, Math.min(1, (ranked.length-rank)*ranked[rank].p));
    adjusted[ranked[rank].i] = bound;
  }
  return adjusted;
}


export function planFamilywiseK(ds, spec={}) {
  if (spec === null || typeof spec !== 'object' || Array.isArray(spec)) {
    throw new TypeError('Family registration requires a JSON object fixed at dataset intake');
  }
  const allowed = new Set(['ks','seed','perms','stopping','alpha']);
  const unexpected = Object.keys(spec).filter(key=>!allowed.has(key));
  if (unexpected.length) {
    throw new Error('Unknown family registration field(s): '+unexpected.join(', '));
  }
  if (!Array.isArray(ds.X) || !Array.isArray(ds.X[0]) || !ds.X[0].length) {
    throw new TypeError('Familywise model family requires an explicit nonempty feature matrix');
  }
  const nFeatures=ds.X[0].length;
  const chosen=spec.ks === undefined
    ? [...new Set([1,Math.min(5,nFeatures),Math.min(20,nFeatures)])]
    : spec.ks;
  if (!Array.isArray(chosen) || !chosen.length || chosen.length>8
      || chosen.some(k=>!Number.isSafeInteger(k)||k<1||k>nFeatures)
      || new Set(chosen).size!==chosen.length) {
    throw new RangeError('Predeclare 1..8 DISTINCT integer top-k pipeline sizes within feature count; repeated or postselected hypotheses are unsupported');
  }
  const seed=spec.seed ?? 20261030, perms=spec.perms ?? 100;
  const stopping=spec.stopping ?? 'decision', alpha=spec.alpha ?? 0.05;
  if (!Number.isSafeInteger(seed) || seed<0 || seed>4294967295) {
    throw new RangeError('Family seed must be an unsigned 32-bit integer fixed before analysis');
  }
  if (!Number.isSafeInteger(perms) || perms<1 || perms>1000) {
    throw new RangeError('Family permutation budget must be a preregistered integer 1..1000');
  }
  if (!['fixed','decision'].includes(stopping)) {
    throw new RangeError('Family stopping rule must be fixed or decision');
  }
  if (!Number.isFinite(alpha) || alpha<=0 || alpha>=1) {
    throw new RangeError('Familywise alpha must be strictly between 0 and 1');
  }
  return Object.freeze({ks:Object.freeze(chosen.slice()),seed,perms,stopping,alpha});
}

export function runFamilywiseK(ds, spec={}) {
  const {ks:chosen,seed,perms,stopping,alpha}=planFamilywiseK(ds,spec);
  const marginal=chosen.map(k=>runControls(ds,['within_batch_permutation'],
    {seed,perms,k,stopping}).results.within_batch_permutation);
  const valid=marginal.every(r=>r && Number.isFinite(r.p) && r.p>=0 && r.p<=1);
  const adjusted=valid ? holmAdjust(marginal.map(r=>r.p)) : chosen.map(()=>null);
  const tests=chosen.map((k,i)=>({
    k, rawP:marginal[i]?.p??null, adjustedP:adjusted[i],
    pExact:marginal[i]?.pExact??false,
    nullSampling:marginal[i]?.nullSampling??null,
    permutationsRun:marginal[i]?.permsRun??0,
    decision:valid ? (adjusted[i]<=alpha?'reject':'retain') : 'unavailable',
  }));
  return {
    method:'Holm-Bonferroni familywise error control',
    alpha, seed, predeclaredKs:chosen, permutationsBudget:perms, stopping,
    status:valid?'valid':'unavailable', tests,
    inferenceLimit:'Controls familywise Type-I error for this predeclared top-k hypothesis family only if each conditional permutation p is valid under its stated row/unit exchangeability null. Holm accommodates arbitrarily dependent p-values; repeated subject rows are not independent when subject_id is provided. Fixed-budget early-stopped upper bounds are conservative. It does not cover post-hoc changes of family, seeds, budgets, repeat queries or unmodeled endpoint selection, and does not prove causation.',
  };
}
