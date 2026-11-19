// External, measured biology receiving. This DOES NOT certify clinical validity.
// The companion R export reads the *real* Bioconductor bladderEset assays,
// randomly selects 200 probes without labels, and emits a temporary CSV.
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { parseCsv } from '../src/data.mjs';
import { runControls, toMarkdown, conditionalOrbitSize } from '../src/core.mjs';

const path=process.env.NULLGARDEN_REAL_CSV;
if(!path)throw new Error('NULLGARDEN_REAL_CSV must point to the official real-assay export');
const bytes=readFileSync(path);
const ds=parseCsv(bytes.toString('utf8'));
if(ds.y.length!==57 || new Set(ds.batch).size!==5 || ds.features.length!==200)
  throw new Error('bladderbatch provenance gate: expected 57 observations, 5 batches, 200 selected probes');
if(ds.subject!==undefined)
  throw new Error('Official sample numbers do not establish independent donor IDs');
const table=new Map();
for(let i=0;i<ds.y.length;i++){
  const b=ds.batch[i];const x=table.get(b)||{batch:b,n:0,label0:0,label1:0};
  x.n++;if(ds.y[i])x.label1++;else x.label0++;table.set(b,x);
}
const designs=[...table.values()].sort((a,b)=>String(a.batch).localeCompare(String(b.batch)));
const mixed=designs.filter(x=>x.label0>0&&x.label1>0);
if(mixed.length<2)throw new Error('Not enough mixed processing batches to attempt conditional comparison');
const start=performance.now();
const r=runControls(ds,undefined,{seed:20261030,perms:100,k:20,stopping:'fixed'});
const results=Object.fromEntries(Object.entries(r.results).map(([name,x])=>[name,{
  auc:x.auc??null,status:x.status??null,
  p:x.p??null,pLower:x.pLower??null,pUpper:x.pUpper??null,pExact:x.pExact??null,
  nullSampling:x.nullSampling??null,nMixedHeldout:x.nMixedHeldout??null,
  nPureHeldout:x.nPureHeldout??null,nUnsupportedTrain:x.nUnsupportedTrain??null
}]));
const oracle={
  schema:'nullgarden.bladderbatch-measured-external-receiving.v1',
  dataSource:'Bioconductor bladderbatch 1.50.0 (bladderEset, RMA-normalized)',
  sourceUrl:'https://bioconductor.org/packages/3.23/data/experiment/src/contrib/bladderbatch_1.50.0.tar.gz',
  datasetCsvSha256:createHash('sha256').update(bytes).digest('hex'),
  nMeasurements:57,nFeatures:ds.features.length,nBatches:5,
  featureSelection:'base R sample.int(n_probes, 200) seed 20261118, no label inspection',
  labelDefinition:'binary factor levels of original pData cancer field, reported by R exporter',
  observedBatchDesign:designs,unitMode:'assumed_independent_rows_no_verified_subject_id',
  permutationOrbitSizeBounded:conditionalOrbitSize(ds.batch,ds.y,513),
  frozenAnalysis:{seed:20261030,k:20,perms:100,stopping:'fixed'},
  results,verdict:r.verdict.id,elapsedMs:Math.round(performance.now()-start),
  caveats:[
    'The original data are measured but pData does not identify independently validated donor clusters',
    'A random fixed 200-probe subset is NOT a validated genome-wide clinical signature',
    'Same-sample random CV may be optimistic; batch-transfer evidence requires adequate mixed heldout batches',
    'Conditional permutation p assumes within-batch unit exchangeability; not demonstrated by public metadata',
    'No causal effect, prospective performance, or clinical generalization can be inferred',
  ]
};
const report=toMarkdown('Bioconductor bladderbatch: real 57-sample RMA assays, 200 frozen probes',r);
if(!report.includes('assumed independent') && !report.includes('assumed to be independent'))
  throw new Error('Exported report failed to disclose unverified subject independence');
const payload=JSON.stringify(oracle);
console.log('real_bladderbatch_report='+payload);
console.log('real_bladderbatch_result_sha256='+createHash('sha256').update(payload).digest('hex'));
