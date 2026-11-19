// Reproducible probe-panel / phenotype-contrast sensitivity of the ORIGINAL
// Bioconductor RMA-normalized bladderEset. The initial pilot panel is included
// and byte-for-byte verified, then five new frozen label-blind probe panels.
// This is a retrospective sensitivity family on ONE cohort, not prospective
// preregistration, nor proof of independent donor-level exchangeability.
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { parseCsv } from '../src/data.mjs';
import { runControls, toMarkdown, conditionalOrbitSize } from '../src/core.mjs';
import { holmAdjust } from '../src/familywise.mjs';

const sourceCsv=process.env.NULLGARDEN_REAL_CSV;
if(!sourceCsv)throw new Error('NULLGARDEN_REAL_CSV must point to the original Bioconductor assay export');
const panelSeeds=Object.freeze([20261118,20261119,20261120,20261121,20261122,20261123]);
const cases=Object.freeze([
  {name:'cancer-v-pooled-reference',suffix:'',n:57,batches:5,mixed:2,neg:17,pos:40,
    note:'Cancer contrasted with Normal and Biopsy pooled; phenotype heterogeneity is a substantive confound'},
  {name:'cancer-v-normal-only',suffix:'.cancer-v-normal.csv',n:48,batches:4,mixed:1,neg:8,pos:40,
    note:'Only one processing batch has both labels; multi-batch heldout AUROC cannot be estimated'},
  {name:'cancer-v-biopsy-only',suffix:'.cancer-v-biopsy.csv',n:49,batches:4,mixed:1,neg:9,pos:40,
    note:'Only one processing batch has both labels; multi-batch heldout AUROC cannot be estimated'},
]);
const pilotSha=Object.freeze({
  'cancer-v-pooled-reference':'20e9f0f8778a747e7922287b0cd3c1e995384302967a7ef3e8caf5426e3dba8d',
  'cancer-v-normal-only':'5ff767f3b5a842125f88f6b8362f48c999766c8ad8de889e73ef4dedc6234049',
  'cancer-v-biopsy-only':'ac93ba3ca8988370d5fff333250f68120547a0a9668180eaf44ff840fc874093',
});
const alpha=0.05, perms=499, familySize=panelSeeds.length*cases.length;
const minAttainableHolm=familySize/(perms+1);
if(minAttainableHolm>alpha)
  throw new Error('Permutation budget '+perms+' cannot reject at familywise alpha '+alpha+
    ' across '+familySize+' hypotheses even if all null scores are lower; increase B to at least '+
    (Math.ceil(familySize/alpha)-1));
const format=(n)=>n===null?null:Number(n);
function receive(panelSeed,cfg){
  const base=panelSeed===panelSeeds[0]?sourceCsv:sourceCsv+'.panel-'+panelSeed;
  const bytes=readFileSync(base+cfg.suffix);
  const sha=createHash('sha256').update(bytes).digest('hex');
  if(panelSeed===panelSeeds[0] && sha!==pilotSha[cfg.name])
    throw new Error('Pilot source regression: '+cfg.name+' CSV SHA changed, cannot compare preexisting measurements');
  const ds=parseCsv(bytes.toString('utf8'));
  if(ds.y.length!==cfg.n || new Set(ds.batch).size!==cfg.batches || ds.features.length!==200)
    throw new Error('Unexpected official assay shape '+cfg.name+' / '+panelSeed);
  if(ds.subject!==undefined)
    throw new Error('Sample IDs are not independently verified donors');
  const byBatch=new Map();
  for(let i=0;i<ds.y.length;i++){
    const id=ds.batch[i],row=byBatch.get(id)||{batch:id,n:0,reference:0,cancer:0};
    row.n++;if(ds.y[i])row.cancer++;else row.reference++;
    byBatch.set(id,row);
  }
  const design=[...byBatch.values()].sort((a,b)=>String(a.batch).localeCompare(String(b.batch)));
  const nMixed=design.filter(g=>g.reference>0&&g.cancer>0).length;
  if(nMixed!==cfg.mixed || ds.y.filter(y=>y===0).length!==cfg.neg
     || ds.y.filter(y=>y===1).length!==cfg.pos)
    throw new Error('Clinical phenotype or batch rank changed in '+cfg.name+' / '+panelSeed);
  const r=runControls(ds,undefined,{seed:20261030,perms,k:20,stopping:'fixed'});
  const md=toMarkdown('Bioconductor real-bladderbatch '+cfg.name+' seed '+panelSeed,r);
  if(!md.includes('No subject IDs were supplied'))
    throw new Error('Exported report omitted independence caveat in '+cfg.name+' / '+panelSeed);
  const p=r.results.within_batch_permutation?.p??null;
  if(!Number.isFinite(p) || p<1/(perms+1)-1e-12)
    throw new Error('Undefined or below-resolution permutation probability in '+cfg.name+' / '+panelSeed);
  const a=r.results;
  const orbit=conditionalOrbitSize(ds.batch,ds.y,513);
  return {
    panelSeed,contrast:cfg.name,nMeasurements:ds.y.length,
    measurementCsvSha256:sha,
    nMixedBatches:nMixed,batchDesign:design,
    featureCount:ds.features.length,unverifiedIndependentSubjectUnits:true,
    conditionalOrbitCount:orbit<=513?orbit:null,conditionalOrbitAbove513:orbit>513,
    naiveAuc:format(a.naive?.auc??null),
    batchOnlyAuc:format(a.batch_only?.auc??null),batchOnlyStatus:a.batch_only?.status??null,
    conditionalNullMeanAuc:format(a.within_batch_permutation?.auc??null),
    rawPermutationP:p,nullSampling:a.within_batch_permutation?.nullSampling??null,
    leaveBatchOutAuc:format(a.leave_batch_out?.auc??null),
    mixedHeldoutBatches:a.leave_batch_out?.nMixedHeldout??0,
    batchCenteredAuc:format(a.batch_residualized?.auc??null),
    heuristicVerdict:r.verdict.id,
    scientificLimit:cfg.note
  };
}
const start=performance.now();
const results=panelSeeds.flatMap(seed=>cases.map(cfg=>receive(seed,cfg)));
if(results.length!==familySize)throw new Error('Missing predeclared hypotheses in family');
const pValues=results.map(r=>r.rawPermutationP);
const adjusted=holmAdjust(pValues);
results.forEach((result,i)=>{
  result.holmAdjustedP=adjusted[i];
  result.holmRejectAt005=adjusted[i]<=alpha;
});
function bounds(values){
  if(!values.length || values.some(v=>v===null||!Number.isFinite(v)))return null;
  return {minimum:Math.min(...values),maximum:Math.max(...values)};
}
const sensitivity=cases.map(cfg=>{
  const sub=results.filter(r=>r.contrast===cfg.name);
  return {
    contrast:cfg.name,panels:sub.length,
    naiveAucRange:bounds(sub.map(r=>r.naiveAuc)),
    rawPermutationPRange:bounds(sub.map(r=>r.rawPermutationP)),
    holmPRange:bounds(sub.map(r=>r.holmAdjustedP)),
    meanNaiveAuc:sub.reduce((a,x)=>a+x.naiveAuc,0)/sub.length,
    familywiseRejectingPanels:sub.filter(r=>r.holmRejectAt005).length,
    mixedBatchCounts:[...new Set(sub.map(r=>r.nMixedBatches))],
    batchTransferAvailablePanels:sub.filter(r=>r.leaveBatchOutAuc!==null).length
  };
});
const payload={
  schema:'nullgarden.bladderbatch-measured-probe-family-v3',
  sourcePackage:'bladderbatch 1.50.0, original RMA normalized bladderEset',
  sourceTarSha256:'431b6c2e2b751f706a897a2d6162e6735617975d8f3d783775418a35c6e9de74',
  sourceUrl:'https://bioconductor.org/packages/3.23/data/experiment/src/contrib/bladderbatch_1.50.0.tar.gz',
  labelDefinition:'Cancer(40) vs Normal(8) and/or Biopsy(9) using original three-level cancer phenotype',
  panelSampling:{seeds:panelSeeds,probesPerPanel:200,selection:'base R sample.int (Mersenne-Twister) with labels hidden'},
  frozenModel:{seed:20261030,k:20,repeats:3,folds:5},
  family:{method:'Holm-Bonferroni',alpha,hypotheses:familySize,
    monteCarloShufflesPerHypothesis:perms,
    minRawP:1/(perms+1),minPossibleFirstHolmAdjustedP:minAttainableHolm,
    selectionStatus:'retrospective sensitivity family chosen after original one-panel pilot; not prospective preregistration',
    correctionAssumption:'every marginal p must be superuniform under within-batch independent-unit exchangeability, not verified in this dataset'},
  results,sensitivity,
  epistemicBoundary:[
    'No donor IDs or independence proof; observed row-level permutation probabilities may not be valid p-values',
    'The Normal and Biopsy reference phenotypes differ; combining them changes the clinical contrast',
    'AUC is internal and not independently replicated in a prospective validation cohort',
    'Only two mixed batches in pooled reference and only one in homogeneous reference contrasts',
    'All 18 tests share one observational cohort; six probe panels are not six biological replications',
    'Holm adjustment accounts for arbitrary dependence across a fixed family only with valid marginal p and no post hoc selection',
  ]
};
const json=JSON.stringify(payload);const hash=createHash('sha256').update(json).digest('hex');
console.log('real_bladderbatch_family_report='+json);
console.log('real_bladderbatch_family_sha256='+hash);
console.log('real_bladderbatch_family_elapsed_ms='+Math.round(performance.now()-start));
