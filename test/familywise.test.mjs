import test from 'node:test';
import assert from 'node:assert/strict';
import { fitScore, repeatedCv, runControls, conditionalLabelAssignments } from '../src/core.mjs';
import { holmAdjust, runFamilywiseK } from '../src/familywise.mjs';
import { makeServer } from '../server.mjs';

// Ten independently labelled subjects, five immutable exchangeability
// blocks, two repeated measurements per subject. No row is independently
// relabelable. The 2^5 exact null assignments are independently enumerated.
function example() {
  const ids=[],batch=[],subject=[],y=[],X=[];
  for(let u=0;u<10;u++)for(let rep=0;rep<2;rep++){
    const b=Math.floor(u/2);
    ids.push('sample-'+u+'-'+rep);
    batch.push('b'+b);subject.push('person-'+u);y.push(u%2);
    X.push([Math.sin((u+1)*1.73)+rep*0.007,
      Math.cos((u+2)*2.11)+rep*0.005]);
  }
  return {ids,batch,subject,y,X,features:['f1','f2']};
}
function independentLabels(ds, mask) {
  return ds.y.map((_,i)=>{
    const u=Math.floor(i/2), b=Math.floor(u/2);
    return ((mask>>b)&1)?u%2:1-u%2;
  });
}
function statistic(ds,y,k,seed) {
  return repeatedCv({...ds,y},(train,heldout)=>
    fitScore(train.map(i=>ds.X[i]),train.map(i=>y[i]),
      heldout.map(i=>ds.X[i]),k),{seed,repeats:3,folds:5});
}

test('Holm step-down is monotone under arbitrary dependence and rejects invalid inputs',()=>{
  assert.deepEqual(holmAdjust([0.01,0.04,0.03]),[0.03,0.06,0.06]);
  assert.deepEqual(holmAdjust([0.2,0.2,0.2]),Array(3).fill(0.2*3));
  assert.deepEqual(holmAdjust([0.9,0.01]),[0.9,0.02]);
  assert.throws(()=>holmAdjust([0.1,NaN]),/finite/);
  assert.throws(()=>holmAdjust([]),/nonempty/);
});

test('independent complete 32-assignment oracle certifies exact familywise p and type-I envelope',()=>{
  const ds=example(), seed=31, klist=[1,2];
  const allocations=Array.from({length:32},(_,mask)=>independentLabels(ds,mask));
  const independentStats=klist.map(k=>allocations.map(y=>statistic(ds,y,k,seed)));
  assert.ok(independentStats.every(values=>values.every(Number.isFinite)));
  const observed=ds.y.join('');
  const observedIndex=allocations.findIndex(labels=>labels.join('')===observed);
  assert.ok(observedIndex>=0);
  const independentRaw=klist.map((_,i)=>
    independentStats[i].filter(s=>s>=independentStats[i][observedIndex]).length/32);
  const direct=runFamilywiseK(ds,{ks:klist,seed,perms:50,stopping:'decision',alpha:0.05});
  assert.equal(direct.status,'valid');
  assert.deepEqual(direct.tests.map(t=>t.rawP),independentRaw);
  assert.deepEqual(direct.tests.map(t=>t.adjustedP),holmAdjust(independentRaw));
  assert.ok(direct.tests.every(t=>t.nullSampling==='complete_batch_conditional_orbit'));
  assert.ok(direct.tests.every(t=>t.pExact));
  const outcomes=allocations.map((_,obs)=>{
    const raw=independentStats.map(scores=>scores.filter(s=>s>=scores[obs]).length/32);
    return holmAdjust(raw);
  });
  for(const alpha of [0.01,0.05,0.1,0.2,0.4,0.5,1]){
    const fwer=outcomes.filter(adjusted=>adjusted.some(p=>p<=alpha)).length/32;
    assert.ok(fwer<=alpha+1e-12,'complete conditional null exceeds familywise alpha='+alpha);
  }
});

test('family corrections are preregistered, and a missing marginal makes the entire family unavailable',()=>{
  const ds=example();
  assert.throws(()=>runFamilywiseK(ds,{ks:[1,1]}),/DISTINCT/);
  assert.throws(()=>runFamilywiseK(ds,{ks:[1,3]}),/feature count/);
  assert.throws(()=>runFamilywiseK(ds,{ks:[]}),/Predeclare/);
  const unavailable={...ds,y:ds.y.map((_,i)=>i===0||i===1?1:0)};
  const r=runFamilywiseK(unavailable,{ks:[1,2],perms:20});
  assert.equal(r.status,'unavailable');
  assert.ok(r.tests.every(t=>t.adjustedP===null && t.decision==='unavailable'));
  assert.match(r.inferenceLimit,/post-hoc/i);
});

test('cold real HTTP subject_id CSV accepts predeclared family and preserves existing science',async()=>{
  const server=makeServer();await new Promise(done=>server.listen(0,'127.0.0.1',done));
  const base='http://127.0.0.1:'+server.address().port;
  const post=(path,b)=>fetch(base+path,{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(b)});
  try{
    const ds=example();
    const csv=['sample_id,batch,label,subject_id,f1,f2',
      ...ds.ids.map((id,i)=>[id,ds.batch[i],ds.y[i],ds.subject[i],...ds.X[i]].join(','))].join('\n');
    const loaded=await(await post('/api/dataset',{csv,family:{ks:[1,2],seed:31,perms:50}})).json();
    assert.equal(loaded.nSubjects,10);
    assert.equal(loaded.subjectMode,'declared_subject_blocks');
    const body={datasetId:loaded.datasetId};
    assert.equal(loaded.familyRegistration.lockedAt,'dataset_intake');
    assert.deepEqual(loaded.familyRegistration.ks,[1,2]);
    assert.equal(loaded.familyRegistration.seed,31);
    assert.equal(loaded.familyRegistration.perms,50);
    assert.match(loaded.familyRegistration.sha256,/^[0-9a-f]{64}$/);
    const response=await post('/api/family',body);
    assert.equal(response.status,200);
    const result=await response.json();
    assert.equal(result.status,'valid');
    assert.equal(result.tests.length,2);
    assert.ok(result.tests.every(t=>t.adjustedP>=t.rawP));
    assert.deepEqual(result.predeclaredKs,[1,2]);
    assert.equal(result.registrationDigest,loaded.familyRegistration.sha256);
    assert.equal(result.familyLockedAt,'dataset_intake');
    const single=await(await post('/api/run',{datasetId:loaded.datasetId,controls:['within_batch_permutation']})).json();
    assert.equal(single.unitMode,'declared_subject_blocks');
    assert.equal(single.nSubjects,10);
    const report=await(await post('/api/report',{datasetId:loaded.datasetId,controls:['within_batch_permutation']})).text();
    assert.match(report,/Subject IDs were supplied/i);
    const missing=await post('/api/family',{datasetId:'expired'});
    assert.equal(missing.status,404);
    const altered=await post('/api/family',{datasetId:loaded.datasetId,ks:[1]});
    assert.equal(altered.status,400);
    assert.match((await altered.json()).error,/locked.*intake/i);
    assert.equal((await post('/api/family',{datasetId:loaded.datasetId,seed:999})).status,400);
    const invalid=await post('/api/dataset',{csv,family:{ks:[1,1]}});
    assert.equal(invalid.status,400);
    const replay=await(await post('/api/family',body)).json();
    assert.deepEqual(replay.tests,result.tests);
    assert.equal(replay.registrationDigest,result.registrationDigest);
  } finally {await new Promise(done=>server.close(done))}
});

test('registered family retains independently witnessed subject-level positive signal after Holm correction',()=>{
  const ds={ids:[],subject:[],batch:[],y:[],X:[],
    features:Array.from({length:25},(_,i)=>'f'+i)};
  for(let u=0;u<24;u++)for(let r=0;r<3;r++){
    const label=u%2;
    ds.ids.push('p'+u+'-'+r);
    ds.subject.push('p'+u);
    ds.batch.push('batch'+Math.floor(u/4));
    ds.y.push(label);
    const row=Array.from({length:25},(_,i)=>i===u?1:0);
    row[24]=(label?5:-5)+((u*17+r*7)%13)/100;
    ds.X.push(row);
  }
  const result=runFamilywiseK(ds,{ks:[1,25],seed:20261030,perms:199,stopping:'decision'});
  console.log('FROZEN_FAMILY_POSITIVE_RECEIVING='+JSON.stringify({tests:result.tests,status:result.status,refinement:result.sequentialRefinement}));
  assert.equal(result.sequentialRefinement,'completed_censored_suffix_for_holm',
    'The marginal .05 stopping boundary cannot certify Holm at .05 for a two-test family');
  assert.equal(result.status,'valid');
  assert.equal(result.tests.length,2);
  for(const t of result.tests){
    assert.ok(t.rawP<=0.025,'marginal subject signal must survive Holm budget');
    assert.ok(t.adjustedP<=0.05,'familywise adjusted signal must remain detectable');
    assert.equal(t.decision,'reject');
    assert.equal(t.nullSampling,'batch_conditional_fixed_statistic');
  }
});

test('familywise pathwise early stopping and complete-budget decisions agree on changed scientific inputs',()=>{
  const scenarios = [
    {name:'shortcut_null',signal:0},
    {name:'transfer_signal',signal:3.5},
  ];
  for(const scenario of scenarios){
    const ds={ids:[],subject:[],batch:[],y:[],X:[],features:['f1','f2','f3']};
    for(let u=0;u<24;u++)for(let rep=0;rep<2;rep++){
      const y=u%2;
      ds.ids.push('s'+u+'-'+rep);
      ds.subject.push('s'+u);
      ds.batch.push('B'+Math.floor(u/4));
      ds.y.push(y);
      ds.X.push([
        Math.sin(u*1.73)+Math.sin(rep)*0.002,
        Math.cos(u*0.81)+scenario.signal*(y?1:-1),
        Math.sin(u*0.43)+scenario.signal*(y?0.5:-0.5),
      ]);
    }
    for(const seed of [17,20261030]){
      const spec={ks:[1,3],seed,perms:45};
      const fixed=runFamilywiseK(ds,{...spec,stopping:'fixed'});
      const sequential=runFamilywiseK(ds,{...spec,stopping:'decision'});
      assert.equal(sequential.status,fixed.status,scenario.name);
      assert.deepEqual(sequential.tests.map(t=>t.decision),
        fixed.tests.map(t=>t.decision),scenario.name+' seed '+seed);
      for(let i=0;i<spec.ks.length;i++){
        assert.ok(sequential.tests[i].adjustedP>=fixed.tests[i].adjustedP-1e-12,
          'Unseen suffix bound may only make adjusted p conservative');
        assert.ok(sequential.tests[i].preliminaryPermutationsRun<=
          sequential.tests[i].permutationsRun);
      }
    }
  }
});
