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

export function runFamilywiseK(ds, {ks, seed=20261030, perms=100, stopping='decision', alpha=0.05}={}) {
  if (!Array.isArray(ds.X) || !Array.isArray(ds.X[0]) || !ds.X[0].length) {
    throw new TypeError('Familywise model family requires an explicit nonempty feature matrix');
  }
  const nFeatures=ds.X[0].length;
  const chosen=ks === undefined
    ? [...new Set([1,Math.min(5,nFeatures),Math.min(20,nFeatures)])]
    : ks;
  if (!Array.isArray(chosen) || !chosen.length || chosen.length>8
      || chosen.some(k=>!Number.isSafeInteger(k)||k<1||k>nFeatures)
      || new Set(chosen).size!==chosen.length) {
    throw new RangeError('Predeclare 1..8 DISTINCT integer top-k pipeline sizes within feature count; repeated or postselected hypotheses are unsupported');
  }
  if (!Number.isFinite(alpha) || alpha<=0 || alpha>=1) {
    throw new RangeError('Familywise alpha must be strictly between 0 and 1');
  }
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
