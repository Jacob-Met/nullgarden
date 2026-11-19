import test from 'node:test';
import assert from 'node:assert/strict';
import { conditionalOrbitSize, conditionalLabelAssignments, conditionalUnits,
  sampleConditionalLabels, stratifiedFolds, repeatedCv, fitScore, runControls,
  subjectAuc } from '../src/core.mjs';
import { rng } from '../src/core.mjs';
import { parseCsv } from '../src/data.mjs';
import { makeServer } from '../server.mjs';

// Twelve independent biological units, two technical readings each.
// Every batch has one positive and one negative SUBJECT: 2^6 not 2^12
// admissible label assignments. Features deliberately repeat within subjects.
const ds = (() => {
  const ids=[], subject=[], batch=[], y=[], X=[];
  for(let u=0;u<12;u++)for(let rep=0;rep<2;rep++){
    ids.push('sample-'+u+'-'+rep);
    subject.push('donor-'+u);
    batch.push('batch-'+Math.floor(u/2));
    y.push(u%2);
    X.push([(u%2 ? 2 : -2)+u/100+rep/1000, (u%3)*0.13]);
  }
  return {ids,subject,batch,y,X,features:['f1','f2']};
})();

const statistic = (labels) => repeatedCv({...ds,y:labels}, (train,test)=>
  fitScore(train.map(i=>ds.X[i]),train.map(i=>labels[i]),
    test.map(i=>ds.X[i]),1),{seed:20261030,repeats:3,folds:5});

function independentLabels(mask) {
  return ds.y.map((_,i)=>{
    const u=Math.floor(i/2), b=Math.floor(u/2);
    return ((mask>>b)&1) ? u%2 : 1-u%2;
  });
}

test('subject ID intake is live, explicit and rejects impossible allocation designs',()=>{
  const csv=['sample_id,batch,label,subject_id,f1,f2',
    ...ds.ids.map((id,i)=>[id,ds.batch[i],ds.y[i],ds.subject[i],...ds.X[i]].join(','))].join('\n');
  const parsed=parseCsv(csv);
  assert.equal(parsed.subject.length,24);
  assert.equal(conditionalUnits(parsed.batch,parsed.y,parsed.subject).length,12);
  assert.deepEqual(parsed.features,['f1','f2']);
  assert.throws(()=>parseCsv(csv.replace('donor-0','donor-1')),
    (e)=>e.code==='SUBJECT_LABEL_CONFLICT');
  assert.throws(()=>parseCsv(csv.replace('sample-0-1,batch-0','sample-0-1,batch-1')),
    (e)=>e.code==='SUBJECT_SPANS_BATCH');
  assert.throws(()=>parseCsv(csv.replace('sample-0-1,batch-0,0,donor-0','sample-0-1,batch-0,1,donor-0')),
    (e)=>e.code==='SUBJECT_LABEL_CONFLICT');
});

test('group-aware CV never splits donors across train/test and AUC weighs donors equally',()=>{
  const folds=stratifiedFolds(ds.y,5,rng(25),ds.subject);
  const seen=new Map();
  folds.forEach((fold,k)=>fold.forEach(i=>{
    if(seen.has(ds.subject[i]))assert.equal(seen.get(ds.subject[i]),k);
    else seen.set(ds.subject[i],k);
  }));
  assert.equal(seen.size,12);
  assert.ok(Number.isFinite(statistic(ds.y)));
  assert.equal(subjectAuc([0,1,1,1,1],[0,1,1,1,1],['a','b','b','b','b']),1);
  assert.throws(()=>stratifiedFolds([0,1],2,rng(1),['same','same']),
    (e)=>e.code==='SUBJECT_LABEL_CONFLICT');
});

test('exact subject-block orbit enumerates each legal assignment once and no invalid row shuffle',()=>{
  assert.equal(conditionalOrbitSize(ds.batch,ds.y,512,ds.subject),64);
  assert.equal(conditionalOrbitSize(ds.batch,ds.y,512),513);
  const allocations=[...conditionalLabelAssignments(ds.batch,ds.y,
    {maxAssignments:512,subject:ds.subject})];
  assert.equal(allocations.length,64);
  assert.equal(new Set(allocations.map(x=>x.join(''))).size,64);
  assert.equal(allocations.filter(x=>x.every((v,i)=>v===ds.y[i])).length,1);
  for(const labels of allocations){
    assert.ok(ds.subject.every((unit,i)=>labels[i]===labels[2*Math.floor(i/2)]));
    for(let b=0;b<6;b++)assert.equal(labels[4*b]+labels[4*b+2],1);
  }
  assert.deepEqual(ds.y.slice(0,4),[0,0,1,1]);
});

test('independent 64-mask oracle matches actual grouped-CV exact p and finite-null type I',()=>{
  const observed=statistic(ds.y);
  const nulls=Array.from({length:64},(_,m)=>statistic(independentLabels(m)));
  assert.ok(nulls.every(Number.isFinite));
  const expected=nulls.filter(x=>x>=observed).length/64;
  const run=runControls(ds,['within_batch_permutation'],
    {seed:20261030,perms:100,k:1,stopping:'decision'});
  const p=run.results.within_batch_permutation;
  assert.equal(run.nSubjects,12);
  assert.equal(run.unitMode,'declared_subject_blocks');
  assert.equal(p.orbitSize,64);
  assert.equal(p.pExact,true);
  assert.equal(p.permsRun,64);
  assert.equal(p.p,expected);
  for(const alpha of [0.01,0.03125,0.05,0.1,0.2,0.5,1]){
    const falseRejection=nulls.filter(x=>nulls.filter(z=>z>=x).length/64<=alpha).length/64;
    assert.ok(falseRejection<=alpha+1e-12,'superuniform at '+alpha);
  }
});

test('seeded conditional Monte Carlo preserves donor blocks even for orbit too large',()=>{
  const many={...ds, batch:ds.batch.map((_,i)=>'one')};
  assert.equal(conditionalOrbitSize(many.batch,many.y,512,many.subject),513);
  const r=rng(17);
  const source=ds.y.slice();
  for(let t=0;t<80;t++){
    const labels=sampleConditionalLabels(many.batch,many.y,r,many.subject);
    assert.ok(labels.every((v,i)=>v===labels[2*Math.floor(i/2)]));
    assert.equal(labels.filter((v,i)=>i%2===0&&v===1).length,6);
  }
  assert.deepEqual(ds.y,source);
  const p=runControls(many,['within_batch_permutation'],{seed:17,perms:8,k:1});
  assert.equal(p.results.within_batch_permutation.nullSampling,'batch_conditional_fixed_statistic');
  assert.equal(p.results.within_batch_permutation.independentUnits,12);
});

test('cold HTTP CSV route retains grouping and exported evidence warning',async()=>{
  const server=makeServer();await new Promise(done=>server.listen(0,'127.0.0.1',done));
  const base='http://127.0.0.1:'+server.address().port;
  async function post(path,value){return fetch(base+path,
    {method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(value)});}
  try{
    const csv=['sample_id,batch,label,subject_id,f1,f2',
      ...ds.ids.map((id,i)=>[id,ds.batch[i],ds.y[i],ds.subject[i],...ds.X[i]].join(','))].join('\n');
    const loaded=await (await post('/api/dataset',{csv})).json();
    assert.equal(loaded.n,24);assert.equal(loaded.nSubjects,12);
    assert.equal(loaded.subjectMode,'declared_subject_blocks');
    const run=await (await post('/api/run',{datasetId:loaded.datasetId,
      controls:['within_batch_permutation']})).json();
    assert.equal(run.results.within_batch_permutation.orbitSize,64);
    const text=await (await post('/api/report',{datasetId:loaded.datasetId,
      controls:['within_batch_permutation']})).text();
    assert.match(text,/Subject IDs were supplied/);
    assert.match(text,/12 independent allocation units/);
  }finally{await new Promise(done=>server.close(done))}
});
