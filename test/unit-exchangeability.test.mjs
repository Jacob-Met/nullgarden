import test from 'node:test';
import assert from 'node:assert/strict';
import { auc, unitAuc, repeatedCv, runControls, stratifiedUnitFolds } from '../src/core.mjs';
import { parseCsv } from '../src/data.mjs';
import { makeServer } from '../server.mjs';

// Twelve independent donors, each observed twice. Six collection batches,
// each containing one label-0 and one label-1 donor. No donor spans batches.
const ds = {
  ids: Array.from({length:24},(_,i)=>'s'+i),
  unit: Array.from({length:24},(_,i)=>'U'+Math.floor(i/2)),
  batch: Array.from({length:24},(_,i)=>'B'+Math.floor(i/4)),
  y: Array.from({length:24},(_,i)=>Math.floor(i/2)%2),
  X: Array.from({length:24},(_,i)=>[Math.floor(i/2)%2?3:-3,
    Math.floor(i/2)+((i%2)?0.002:0)]),
  features:['f1','f2'],
};

test('unit-stratified CV never splits technical replicates across train and test',()=>{
  const seen=[];
  const scorer=(train,test)=>{
    const trainIds=new Set(train.map(i=>ds.unit[i]));
    const testIds=new Set(test.map(i=>ds.unit[i]));
    assert.ok([...testIds].every(id=>!trainIds.has(id)),
      'identical donor in both train and test is leakage');
    seen.push(testIds.size);
    return test.map(()=>0.5);
  };
  const cv=repeatedCv(ds,scorer,{seed:77,folds:3,repeats:2});
  assert.equal(cv,0.5);
  assert.equal(seen.length,6);
  assert.ok(seen.every(n=>n>0));
  const folds=stratifiedUnitFolds(ds.y,ds.unit,3,()=>0.5);
  assert.equal(folds.length,3);
  assert.ok(folds.flat().length===ds.y.length);
  for(const f of folds)for(const i of f)
    assert.ok(f.includes(...ds.ids.map((_,j)=>j).filter(j=>ds.unit[j]===ds.unit[i])));
});

test('unit AUC counts independent donors once, not technical replicates',()=>{
  const scores=[0.9,0.1,0.4,0.2,0.3];
  const y=[1,1,1,0,0],unit=['A','A','B','C','D'];
  assert.equal(auc(scores,y),4/6);
  assert.equal(unitAuc(scores,y,unit),1);
});

test('grouped conditional permutation enumerates 64 whole-unit labels, not row-level shuffles',()=>{
  const run=runControls(ds,['within_batch_permutation'],{seed:20261030,perms:100,k:2,stopping:'decision'});
  const p=run.results.within_batch_permutation;
  assert.equal(run.nUnits,12);
  assert.equal(p.orbitSize,64);
  assert.equal(p.nullSampling,'complete_batch_conditional_orbit');
  assert.equal(p.permutationUnit,'unit');
  assert.equal(p.pExact,true);
  assert.ok(Number.isFinite(p.p));
  assert.ok(p.p>=1/64);
  assert.ok(p.p<=1);
});

test('CSV supports explicit unit_id and validates label/batch constancy per unit',()=>{
  const header='sample_id,batch,label,unit_id,f1,f2';
  const csv=[header,...ds.ids.map((id,i)=>[id,ds.batch[i],ds.y[i],
    ds.unit[i],...ds.X[i]].join(','))].join('\n');
  const parsed=parseCsv(csv);
  assert.deepEqual(parsed.unit,ds.unit);
  assert.equal(parsed.X[0].length,2);
  assert.equal(parsed.features.length,2);
  const badLabel=csv.replace('s1,B0,0,U0','s1,B0,1,U0');
  assert.throws(()=>parseCsv(badLabel),/unit|same.*label|inconsistent/i);
  const badBatch=csv.replace('s1,B0,0,U0','s1,B5,0,U0');
  assert.throws(()=>parseCsv(badBatch),/unit|same.*batch|inconsistent/i);
});

test('cold HTTP uploaded repeated-unit CSV retains true conditional orbit and unit counts',async()=>{
  const csv=['sample_id,batch,label,unit_id,f1,f2',
    ...ds.ids.map((id,i)=>[id,ds.batch[i],ds.y[i],ds.unit[i],
      ...ds.X[i]].join(','))].join('\n');
  const server=makeServer();
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
  const base='http://127.0.0.1:'+server.address().port;
  const post=(path,payload)=>fetch(base+path,
    {method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(payload)});
  try{
    const loaded=await (await post('/api/dataset',{csv})).json();
    assert.equal(loaded.n,24);
    assert.equal(loaded.nUnits,12);
    const run=await (await post('/api/run',
      {datasetId:loaded.datasetId,controls:['within_batch_permutation']})).json();
    assert.equal(run.nUnits,12);
    assert.equal(run.results.within_batch_permutation.orbitSize,64);
    const report=await (await post('/api/report',
      {datasetId:loaded.datasetId,controls:['within_batch_permutation']})).text();
    assert.match(report,/independent units|donor|unit-level/i);
  }finally{await new Promise(resolve=>server.close(resolve))}
});
