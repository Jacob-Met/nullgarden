// Frozen complete-null receiving, NOT a generative model of biology.
// Null: 12 independent donors, six batches, one donor per label per batch.
// Features encode only donor identity, never the randomly allocated outcome.
// Every one of the 2^6 donor-level label assignments is evaluated, so the
// false-rejection rate is an exact finite-population enumeration (for the
// chosen design and frozen Monte Carlo seed), not an asymptotic estimate.
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { performance } from 'node:perf_hooks';
import { conditionalLabelAssignments, runControls, repeatedCv, fitScore } from '../src/core.mjs';

const CONFIG = Object.freeze({
  schema:'nullgarden.subject-null-calibration.v1',
  nullSeed:20261118, permutationSeed:20261030,
  repetitions:[1,3], batches:6, subjectsPerBatch:2,
  nullPermutations:40, nullAlpha:0.05,
  alternativeSubjects:24, alternativeRowsPerSubject:3, alternativePermutations:199,
});
function makeNull(replicates) {
  const ds={ids:[],subject:[],batch:[],y:[],X:[],
    features:Array.from({length:12},(_,i)=>'identity-'+i)};
  for(let u=0;u<12;u++)for(let r=0;r<replicates;r++) {
    ds.ids.push('s'+u+'-reading'+r);
    ds.subject.push('s'+u);
    ds.batch.push('batch'+Math.floor(u/2));
    ds.y.push(u%2);
    ds.X.push(Array.from({length:12},(_,j)=>u===j?1:0));
  }
  return ds;
}
function statistic(ds,y,k=12) {
  return repeatedCv({...ds,y},(train,test)=>fitScore(
    train.map(i=>ds.X[i]),train.map(i=>y[i]),
    test.map(i=>ds.X[i]),k),{seed:CONFIG.permutationSeed,repeats:3,folds:5});
}
function pDistribution(scores) {
  return scores.map(v=>scores.filter(other=>other>=v).length/scores.length);
}
function nullReceiving(repeats) {
  const ds=makeNull(repeats), start=performance.now();
  const legal=[...conditionalLabelAssignments(ds.batch,ds.y,
    {maxAssignments:64,subject:ds.subject})];
  assert.equal(legal.length,64);
  assert.equal(new Set(legal.map(y=>y.join(''))).size,64);
  const unitScores=legal.map(y=>statistic(ds,y));
  assert.ok(unitScores.every(Number.isFinite));
  const unitPs=pDistribution(unitScores);
  const groupFpr=unitPs.filter(p=>p<=CONFIG.nullAlpha).length/64;
  assert.ok(groupFpr<=CONFIG.nullAlpha,
    'independent exact blocked conditional p must be superuniform');

  // Incumbent comparator: same data, labels and full production pipeline,
  // but remove subject metadata so row-based CV and null shuffles are applied.
  // With one reading per donor this should remain identical; with repeats the
  // row-null is *not exchangeable* under this donor-level sampling experiment.
  const rowDs={...ds,subject:undefined}, rowP=[], rowAuc=[], groupAuc=[];
  for(const y of legal) {
    const result=runControls({...rowDs,y},['within_batch_permutation'],
      {seed:CONFIG.permutationSeed,perms:CONFIG.nullPermutations,k:12,stopping:'fixed'});
    const p=result.results.within_batch_permutation;
    assert.ok(p.p!==null, 'row-level incumbent must produce a comparison');
    rowP.push(p.p);rowAuc.push(result.results.naive.auc);
    groupAuc.push(statistic(ds,y));
  }
  const rowFpr=rowP.filter(p=>p<=CONFIG.nullAlpha).length/64;
  const mean=a=>a.reduce((s,v)=>s+v,0)/a.length;
  const result={
    readingsPerSubject:repeats, nMeasurements:ds.y.length,
    nIndependentSubjects:12, nullOrbitUnits:64,
    rowFpr, blockedExactFpr:groupFpr,
    rowRejected:rowP.filter(p=>p<=CONFIG.nullAlpha).length,
    blockedRejected:unitPs.filter(p=>p<=CONFIG.nullAlpha).length,
    rowAucMean:mean(rowAuc), blockedAucMean:mean(groupAuc),
    rowMinP:Math.min(...rowP), blockedMinP:Math.min(...unitPs),
    elapsedMs:Math.round(performance.now()-start),
  };
  return result;
}
function positiveControl() {
  const ds={ids:[],subject:[],batch:[],y:[],X:[],
    features:Array.from({length:25},(_,i)=>'feature'+i)};
  const units=CONFIG.alternativeSubjects;
  for(let u=0;u<units;u++)for(let r=0;r<CONFIG.alternativeRowsPerSubject;r++){
    const label=u%2;
    ds.ids.push('p'+u+'-'+r);ds.subject.push('p'+u);
    ds.batch.push('batch'+Math.floor(u/4));ds.y.push(label);
    // Population signal in feature 24 transfers to unseen subjects.
    // 24 donor fingerprint features are nuisance covariates.
    const row=Array.from({length:25},(_,i)=>i===u?1:0);
    row[24]=(label?5:-5)+((u*17+r*7)%13)/100;
    ds.X.push(row);
  }
  const result=runControls(ds,['within_batch_permutation'],
    {seed:CONFIG.permutationSeed,perms:CONFIG.alternativePermutations,k:25,stopping:'fixed'});
  const p=result.results.within_batch_permutation;
  assert.equal(result.nSubjects,units);
  assert.ok(result.results.naive.auc>0.95,'held-out subject signal should transfer');
  assert.ok(p.p<=0.05,'a strong real subject-level signal should remain detectable');
  return {nSubjects:units,nRows:ds.y.length,observedAuc:result.results.naive.auc,
    p:p.p,permsRun:p.permsRun,method:p.nullSampling,status:p.status};
}
const start=performance.now();
const nulls=CONFIG.repetitions.map(nullReceiving);
const alternative=positiveControl();
const report={...CONFIG,nulls,alternative,totalMs:Math.round(performance.now()-start),
  limitations:[
    'Exact finite-population null within one frozen donor/feature design, not universal nominal type-I guarantee',
    'The row comparator is invalid because its row-permutation orbit violates donor-level exchangeability',
    'The alternative is deterministic synthetic signal; not clinical or causal validation',
    'No multiple-hypothesis search is calibrated',
  ]};
const canonical=JSON.stringify(report);
console.log(JSON.stringify({report,sha256:createHash('sha256').update(canonical).digest('hex')}));
