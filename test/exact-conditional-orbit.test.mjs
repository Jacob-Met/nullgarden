import test from 'node:test';
import assert from 'node:assert/strict';
import { conditionalOrbitSize, conditionalLabelAssignments, fitScore,
  repeatedCv, runControls, toMarkdown } from '../src/core.mjs';
import { makeServer } from '../server.mjs';

const ds = {
  ids: Array.from({length:12}, (_,i)=>'s'+i),
  batch: Array.from({length:12}, (_,i)=>'B'+Math.floor(i/2)),
  y: Array.from({length:12}, (_,i)=>i%2),
  X: Array.from({length:12}, (_,i)=>[(i%2?3:-3)+0.02*Math.floor(i/2)]),
  features:['g1'],
};

// Independent exhaustive 2^6 reference: does not use the production enumerator.
function independentOrbitP(data, {seed=20261030,k=1}={}) {
  const statistic = (y) => repeatedCv({...data,y}, (train,test) =>
    fitScore(train.map(i=>data.X[i]),train.map(i=>y[i]),
      test.map(i=>data.X[i]),k),{seed,repeats:3,folds:5});
  const observed=statistic(data.y);
  assert.ok(Number.isFinite(observed));
  const nulls=[];
  for(let mask=0;mask<64;mask++){
    const labels=data.y.map((_,i)=>((mask >> Math.floor(i/2))&1) ? i%2 : 1-i%2);
    const v=statistic(labels);
    assert.ok(Number.isFinite(v));
    nulls.push(v);
  }
  return {p:nulls.filter(x=>x>=observed).length/64,observed,nulls};
}

test('conditional-space size and lazy enumeration identify all distinct assignments exactly', () => {
  assert.equal(conditionalOrbitSize(ds.batch,ds.y,1024),64);
  const before=ds.y.slice();
  const labels=[...conditionalLabelAssignments(ds.batch,ds.y,{maxAssignments:1024})];
  assert.equal(labels.length,64);
  assert.equal(new Set(labels.map(y=>y.join(''))).size,64);
  assert.equal(labels.filter(y=>y.every((v,i)=>v===ds.y[i])).length,1);
  assert.ok(labels.every(y=>y.every((v,i)=>v===0 || v===1)
    && [...new Set(ds.batch)].every(b=>y.filter((_,i)=>ds.batch[i]===b).reduce((a,v)=>a+v,0)===1)));
  assert.deepEqual(ds.y,before,'enumerator must never mutate source labels');
});

test('complete conditional orbit yields a finite-sample exact p, not a +1 Monte Carlo approximation', () => {
  const ref=independentOrbitP(ds,{seed:20261030,k:1});
  const run=runControls(ds,['within_batch_permutation'],{seed:20261030,perms:100,k:1,stopping:'decision'});
  const p=run.results.within_batch_permutation;
  assert.equal(p.nullSampling,'complete_batch_conditional_orbit');
  assert.equal(p.orbitSize,64);
  assert.equal(p.permsRun,64);
  assert.equal(p.permsBudget,64);
  assert.equal(p.p,ref.p);
  assert.equal(p.pExact,true);
  assert.equal(p.pLower,p.p);
  assert.equal(p.pUpper,p.p);
  assert.equal(p.stopping,'exact');
  assert.equal(p.status,ref.p<=0.05?'signal':'neutral');
  assert.match(toMarkdown('exact conditional',run),/exact conditional|complete.*orbit/i);
});

test('all-tie fixed features have exact p=1 and no evidence of a shortcut or signal', () => {
  const allTie={...ds,X:ds.X.map(()=>[7])};
  const run=runControls(allTie,['within_batch_permutation'],{seed:31,perms:100,k:1});
  const p=run.results.within_batch_permutation;
  assert.equal(p.orbitSize,64);
  assert.equal(p.p,1);
  assert.equal(p.status,'neutral');
});

test('large conditional space falls through to prespecified Monte Carlo bounds', () => {
  const large={
    ...ds,batch:Array.from({length:12},()=> 'one'),
    y:Array.from({length:12},(_,i)=>i%2),
  };
  assert.equal(conditionalOrbitSize(large.batch,large.y,100),101);
  const run=runControls(large,['within_batch_permutation'],{seed:11,perms:20,k:1,stopping:'decision'});
  const p=run.results.within_batch_permutation;
  assert.equal(p.nullSampling,'batch_conditional_fixed_statistic');
  assert.equal(p.permsBudget,20);
  assert.ok(p.permsRun<=20);
  assert.ok(Number.isFinite(p.p));
});

test('cold HTTP automatically upgrades a small stratified CSV to exact inference',async()=>{
  const server=makeServer(); await new Promise(r=>server.listen(0,'127.0.0.1',r));
  const base='http://127.0.0.1:'+server.address().port;
  const post=(path,payload)=>fetch(base+path,{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(payload)});
  try{
    const csv=['sample_id,batch,label,f1',
      ...ds.ids.map((s,i)=>[s,ds.batch[i],ds.y[i],ds.X[i][0]].join(','))].join('\n');
    const loaded=await (await post('/api/dataset',{csv})).json();
    const result=await (await post('/api/run',{datasetId:loaded.datasetId,
      controls:['within_batch_permutation']})).json();
    assert.equal(result.results.within_batch_permutation.orbitSize,64);
    assert.equal(result.results.within_batch_permutation.pExact,true);
    const report=await (await post('/api/report',{datasetId:loaded.datasetId,
      controls:['within_batch_permutation']})).text();
    assert.match(report,/exact conditional|complete.*orbit/i);
  }finally{await new Promise(resolve=>server.close(resolve))}
});
