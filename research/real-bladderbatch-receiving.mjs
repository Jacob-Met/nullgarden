// External, measured biological assay receiving. Not clinical validation.
// The companion R code reads ORIGINAL Bioconductor RMA-normalized assays,
// selects a frozen 200-probe subset without labels, and exports temporary CSVs.
// Scientific sensitivity: the published 'cancer' phenotype has THREE classes:
// Cancer, Normal and Biopsy. Grouping the latter two as one binary reference
// changes what is being estimated; report every contrast and its batch rank.
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { parseCsv } from '../src/data.mjs';
import { runControls, toMarkdown, conditionalOrbitSize } from '../src/core.mjs';

const path=process.env.NULLGARDEN_REAL_CSV;
if(!path)throw new Error('NULLGARDEN_REAL_CSV must point to official real-assay export');
const config=[
  {name:'cancer-v-pooled-reference',suffix:'',n:57,batches:5,mixed:2,neg:17,pos:40,
    note:'The 0 class pools phenotypically different Normal and Biopsy samples; non-cancer is NOT a homogeneous healthy comparator'},
  {name:'cancer-v-normal-only',suffix:'.cancer-v-normal.csv',n:48,batches:4,mixed:1,neg:8,pos:40,
    note:'Only one processing batch is mixed, so within-batch replication across distinct batches is not identifiable'},
  {name:'cancer-v-biopsy-only',suffix:'.cancer-v-biopsy.csv',n:49,batches:4,mixed:1,neg:9,pos:40,
    note:'Only one processing batch is mixed, so within-batch replication across distinct batches is not identifiable'},
];
function receive(cfg){
  const bytes=readFileSync(path+cfg.suffix);
  const ds=parseCsv(bytes.toString('utf8'));
  if(ds.y.length!==cfg.n || new Set(ds.batch).size!==cfg.batches || ds.features.length!==200)
    throw new Error('Pinned real-data dimension mismatch in '+cfg.name);
  if(ds.subject!==undefined)throw new Error('Sample numbers do not establish independently verified donor IDs');
  const table=new Map();
  for(let i=0;i<ds.y.length;i++){
    const b=ds.batch[i],x=table.get(b)||{batch:b,n:0,label0:0,label1:0};
    x.n++;if(ds.y[i])x.label1++;else x.label0++;table.set(b,x);
  }
  const designs=[...table.values()].sort((a,b)=>String(a.batch).localeCompare(String(b.batch)));
  const mixed=designs.filter(x=>x.label0>0&&x.label1>0);
  if(mixed.length!==cfg.mixed || ds.y.filter(x=>x===0).length!==cfg.neg
     || ds.y.filter(x=>x===1).length!==cfg.pos)
    throw new Error('Unanticipated phenotype/batch support in '+cfg.name);
  const t=performance.now();
  const run=runControls(ds,undefined,{seed:20261030,perms:100,k:20,stopping:'fixed'});
  const results=Object.fromEntries(Object.entries(run.results).map(([name,x])=>[name,{
    auc:x.auc??null,status:x.status??null,
    p:x.p??null,pLower:x.pLower??null,pUpper:x.pUpper??null,pExact:x.pExact??null,
    nullSampling:x.nullSampling??null,nMixedHeldout:x.nMixedHeldout??null,
    nPureHeldout:x.nPureHeldout??null,nUnsupportedTrain:x.nUnsupportedTrain??null
  }]));
  const report=toMarkdown('Bioconductor bladderbatch '+cfg.name,run);
  if(!report.includes('No subject IDs were supplied'))
    throw new Error('Exported report omitted unverified subject independence in '+cfg.name);
  const orbit=conditionalOrbitSize(ds.batch,ds.y,513);
  return {
    contrast:cfg.name,interpretation:cfg.note,
    temporaryCsvSha256:createHash('sha256').update(bytes).digest('hex'),
    nMeasurements:ds.y.length,nProbes:ds.features.length,nBatches:designs.length,
    nMixedBatches:mixed.length,
    batchDesign:designs,independentDonorIdsPresent:false,
    conditionalOrbitCap:513,conditionalOrbitCount:orbit<=513?orbit:null,
    orbitExceedsCap:orbit>513,
    results,verdict:run.verdict.id,elapsedMs:Math.round(performance.now()-t)
  };
}
const contrasts=config.map(receive);
const payload={
  schema:'nullgarden.bladderbatch-measured-external-receiving.v2',
  dataset:'Original RMA-normalized bladderEset (Bioconductor bladderbatch 1.50.0)',
  sourceTarSha256:'431b6c2e2b751f706a897a2d6162e6735617975d8f3d783775418a35c6e9de74',
  sourceUrl:'https://bioconductor.org/packages/3.23/data/experiment/src/contrib/bladderbatch_1.50.0.tar.gz',
  phenotype:'Published pData$cancer has exactly Cancer(40), Normal(8), Biopsy(9)',
  featureSelection:'200 unlabeled microarray probes randomly sampled with base R sample.int and seed 20261118',
  frozenAnalysis:{seed:20261030,selectedFeatures:20,permutationBudget:100,stopping:'fixed',cvRepeats:3,folds:5},
  contrasts,
  inferenceLimits:[
    'Pooling distinct Normal and Biopsy phenotypes gives two mixed batches but changes the scientific contrast',
    'Restricting references to Normal or Biopsy separately leaves only one mixed processing batch per comparison',
    'No verified donor ID is provided; sample IDs cannot validate subject independence or exchangeability',
    'A frozen randomly sampled 200-probe subset is not a clinically validated genome-wide classifier',
    'The conditional permutation p values presume exchangeable units within each batch under the null',
    'No causal effect, prospective diagnostic accuracy or external cohort replication is established',
  ]
};
const json=JSON.stringify(payload);
console.log('real_bladderbatch_report='+json);
console.log('real_bladderbatch_result_sha256='+createHash('sha256').update(json).digest('hex'));
