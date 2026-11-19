import test from 'node:test';
import assert from 'node:assert/strict';
import { runControls, stratifiedFolds, nestedClusterFolds, repeatedCv, fitScore, rng,
  conditionalClusters, conditionalOrbitSize, conditionalLabelAssignments, sampleConditionalLabels, toMarkdown } from '../src/core.mjs';
import { parseCsv } from '../src/data.mjs';
import { runFamilywiseK } from '../src/familywise.mjs';
import { makeServer } from '../server.mjs';

// Frozen BEFORE modifying the candidate: repeated measurements nested within
// related subjects and households. Binary outcome is a HOUSEHOLD attribute,
// not a transfer-learnable individual biology signal. Only family fingerprint
// features are observed. Each batch has exactly two label-0 and two label-1
// households, so batch-only controls cannot explain family leakage.
export function nestedFamilyFingerprintNull() {
  const ids=[],subject=[],cluster=[],batch=[],y=[],X=[],features=[];
  for(let f=0;f<20;f++)features.push('fingerprint_'+f);
  for(let f=0;f<20;f++)for(let s=0;s<2;s++)for(let rep=0;rep<2;rep++) {
    ids.push('F'+f+'-S'+s+'-R'+rep);
    subject.push('F'+f+'-S'+s);
    cluster.push('F'+f);
    batch.push('B'+Math.floor(f/4));
    y.push((f%4)<2?1:0);
    X.push(Array.from({length:20},(_,j)=>j===f?1:0));
  }
  return {ids,subject,cluster,batch,y,X,features};
}
test('frozen independent nested-family negative control exposes one-level CV and label null breach',()=>{
  const ds=nestedFamilyFingerprintNull();
  const folds=stratifiedFolds(ds.y,5,rng(20261030),ds.subject);
  const leakage=new Set();
  for(const fold of folds){
    const held=new Set(fold.map(i=>ds.cluster[i]));
    const train=new Set(ds.cluster.filter((_,i)=>!fold.includes(i)));
    for(const family of held)if(train.has(family))leakage.add(family);
  }
  assert.ok(leakage.size>=10,'the existing one-level donor CV splits most related families');
  const run=runControls({...ds,cluster:undefined},['within_batch_permutation'],
    {seed:20261030,k:20,perms:199,stopping:'fixed'});
  const p=run.results.within_batch_permutation;
  console.log('FROZEN_NESTED_FAMILY_INCUMBENT='+JSON.stringify({
    familyCount:20,independentFamilies:20,subjectCount:40,rowCount:80,
    leakedFamilies:leakage.size,subjectHeldoutAuc:run.results.naive.auc,
    subjectNullP:p.p,permutations:p.permsRun,nullMethod:p.nullSampling}));
  assert.equal(run.nSubjects,40);
  assert.ok(Number.isFinite(run.results.naive.auc));
  assert.ok(p && Number.isFinite(p.p));
});

test('nested cluster-fold CV eliminates frozen family fingerprint memorization; no exchangeable contrast abstains',()=>{
  const ds=nestedFamilyFingerprintNull();
  const folds=nestedClusterFolds(ds,5,rng(20261030));
  assert.equal(folds.length,5);
  assert.deepEqual(folds.flat().sort((a,b)=>a-b),ds.y.map((_,i)=>i));
  for(const fold of folds){
    const test=new Set(fold), heldFamilies=new Set(fold.map(i=>ds.cluster[i]));
    for(let i=0;i<ds.y.length;i++){
      assert.equal(test.has(i),heldFamilies.has(ds.cluster[i]),
        'one family cannot leak between training and heldout groups');
    }
    assert.ok(fold.some(i=>ds.y[i]===1));
    assert.ok(fold.some(i=>ds.y[i]===0));
  }
  assert.equal(conditionalClusters(ds.batch,ds.y,ds.subject,ds.cluster).length,20);
  assert.equal(conditionalOrbitSize(ds.batch,ds.y,512,ds.subject,ds.cluster),1);
  const corrected=runControls(ds,['within_batch_permutation'],{seed:20261030,k:20,perms:199});
  assert.equal(corrected.nClusters,20);
  assert.equal(corrected.unitMode,'declared_nested_cluster_blocks');
  assert.ok(corrected.results.naive.auc<=0.55,
    'true out-of-family prediction cannot recover unseen household fingerprints');
  const p=corrected.results.within_batch_permutation;
  assert.equal(p.status,'unavailable');
  assert.equal(p.p,null);
  assert.equal(p.orbitSize,1);
  assert.equal(p.nullSampling,'no_nested_cluster_contrast');
  assert.match(toMarkdown('family-null',corrected),/Nested cluster_id/);
  console.log('FROZEN_NESTED_FAMILY_CORRECTED='+JSON.stringify({
    nClusters:20,clusterHeldoutAuc:corrected.results.naive.auc,
    conditionalOrbit:p.orbitSize,p:p.p,status:p.status}));
});

function nestedMixed(families=5,positive=false) {
  const ids=[],subject=[],cluster=[],batch=[],y=[],X=[];
  const nFeatures=positive?families*2+1:2;
  for(let f=0;f<families;f++)for(let s=0;s<2;s++)for(let rep=0;rep<2;rep++){
    ids.push('F'+f+'S'+s+'R'+rep);
    subject.push('F'+f+'S'+s);
    cluster.push('F'+f);batch.push('B'+Math.floor(f/2));y.push(s);
    if(positive){
      const u=f*2+s,row=Array.from({length:nFeatures},(_,j)=>j===u?1:0);
      row[nFeatures-1]=(s?5:-5)+((f*13+s*7+rep*3)%11)/100;
      X.push(row);
    }else{
      X.push([Math.sin(f*3.21+s*1.43)+rep*.002,
        Math.cos(f*1.23+s*2.19)+rep*.003]);
    }
  }
  return {ids,subject,cluster,batch,y,X,
    features:Array.from({length:nFeatures},(_,i)=>'f'+i)};
}
function independentNestedLabels(ds,mask) {
  return ds.y.map((_,i)=>{
    const family=Math.floor(i/4),subj=Math.floor(i/2)%2;
    return ((mask>>family)&1)?1-subj:subj;
  });
}
test('independent exact 32-allocation nested family oracle matches production p and finite-null Type-I',()=>{
  const ds=nestedMixed(),seed=31,k=1;
  const labels=Array.from({length:32},(_,mask)=>independentNestedLabels(ds,mask));
  assert.equal(conditionalOrbitSize(ds.batch,ds.y,512,ds.subject,ds.cluster),32);
  const enumerated=[...conditionalLabelAssignments(ds.batch,ds.y,
    {maxAssignments:512,subject:ds.subject,cluster:ds.cluster})];
  assert.equal(enumerated.length,32);
  assert.deepEqual(new Set(enumerated.map(a=>a.join(''))),
    new Set(labels.map(a=>a.join(''))));
  const statistic=(yy)=>repeatedCv({...ds,y:yy},(train,test)=>
    fitScore(train.map(i=>ds.X[i]),train.map(i=>yy[i]),
      test.map(i=>ds.X[i]),k),{seed,repeats:3,folds:5});
  const scores=labels.map(statistic),observed=statistic(ds.y);
  assert.ok(scores.every(Number.isFinite));
  const expected=scores.filter(x=>x>=observed).length/32;
  const result=runControls(ds,['within_batch_permutation'],{seed,k,perms:60});
  const p=result.results.within_batch_permutation;
  assert.equal(p.pExact,true);
  assert.equal(p.p,expected);
  assert.equal(p.independentClusters,5);
  assert.equal(p.conditioningLevel,'nested_cluster');
  for(const alpha of [.01,.05,.1,.2,.5,1]){
    const falseReject=scores.filter(x=>scores.filter(other=>other>=x).length/32<=alpha).length/32;
    assert.ok(falseReject<=alpha+1e-12);
  }
});
test('nested metadata fails closed on family/site contradictions at intake and direct API',()=>{
  const ds=nestedMixed();
  const wrongCluster=ds.cluster.slice();wrongCluster[1]='other';
  assert.throws(()=>conditionalClusters(ds.batch,ds.y,ds.subject,wrongCluster),
    e=>e.code==='SUBJECT_SPANS_CLUSTER');
  const across=ds.cluster.slice();
  for(let i=8;i<12;i++)across[i]=ds.cluster[0];
  assert.throws(()=>conditionalClusters(ds.batch,ds.y,ds.subject,across),
    e=>e.code==='CLUSTER_SPANS_BATCH');
  const shuffled=sampleConditionalLabels(ds.batch,ds.y,rng(19),ds.subject,ds.cluster);
  for(let i=0;i<shuffled.length;i+=2)assert.equal(shuffled[i],shuffled[i+1]);
  const csv=['sample_id,batch,label,subject_id,cluster_id,f1,f2',
    ...ds.ids.map((id,i)=>[id,ds.batch[i],ds.y[i],ds.subject[i],ds.cluster[i],...ds.X[i]].join(','))].join('\n');
  assert.equal(parseCsv(csv).cluster.length,ds.cluster.length);
  assert.throws(()=>parseCsv(csv.replace('F0S0R1,B0,0,F0S0,F0','F0S0R1,B0,0,F0S0,F9')),
    e=>e.code==='SUBJECT_SPANS_CLUSTER');
  assert.throws(()=>parseCsv(csv.replace('subject_id,cluster_id','cluster_id,subject_id')),
    /Nested cluster_id requires subject_id/);
});
test('transfer-learnable population signal survives fully cluster-heldout CV and immutable Holm family',()=>{
  const ds=nestedMixed(20,true);
  const family=runFamilywiseK(ds,{ks:[1,41],seed:20261030,perms:199,stopping:'decision'});
  assert.equal(family.status,'valid');
  assert.ok(family.tests.every(x=>x.adjustedP<=.05 && x.decision==='reject'));
  assert.ok(family.tests.every(x=>x.nullSampling==='batch_conditional_fixed_statistic'));
  const score=runControls(ds,['within_batch_permutation'],{seed:20261030,k:41,perms:199});
  assert.ok(score.results.naive.auc>.95);
  console.log('FROZEN_NESTED_FAMILY_POSITIVE='+JSON.stringify({
    nFamilies:20,nSubjects:40,groupHeldoutAuc:score.results.naive.auc,
    family:family.tests.map(t=>({k:t.k,p:t.rawP,adjusted:t.adjustedP}))}));
});
test('cold HTTP nested CSV registration fingerprints family nesting and refuses exchangeability violations',async()=>{
  const server=makeServer();await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
  const base='http://127.0.0.1:'+server.address().port;
  const post=(path,b)=>fetch(base+path,{method:'POST',
    headers:{'content-type':'application/json'},body:JSON.stringify(b)});
  try{
    const ds=nestedMixed();
    const csv=['sample_id,batch,label,subject_id,cluster_id,f1,f2',
      ...ds.ids.map((id,i)=>[id,ds.batch[i],ds.y[i],ds.subject[i],ds.cluster[i],...ds.X[i]].join(','))].join('\n');
    const response=await post('/api/dataset',{csv,family:{ks:[1,2],seed:31,perms:60}});
    assert.equal(response.status,200);
    const loaded=await response.json();
    assert.equal(loaded.nSubjects,10);
    assert.equal(loaded.nClusters,5);
    assert.equal(loaded.nestedClusterMode,'cluster_disjoint_within_cluster_permutation');
    const run=await(await post('/api/run',{datasetId:loaded.datasetId,
      controls:['within_batch_permutation']})).json();
    assert.equal(run.unitMode,'declared_nested_cluster_blocks');
    assert.equal(run.results.within_batch_permutation.orbitSize,32);
    const family=await(await post('/api/family',{datasetId:loaded.datasetId})).json();
    assert.equal(family.registrationDigest,loaded.familyRegistration.sha256);
    const report=await(await post('/api/report',{datasetId:loaded.datasetId,
      controls:['within_batch_permutation']})).text();
    assert.match(report,/Nested cluster_id/);
    const changed=csv.replace(/,F0,/g,',family_0,');
    const second=await(await post('/api/dataset',{csv:changed,
      family:{ks:[1,2],seed:31,perms:60}})).json();
    assert.notEqual(second.familyRegistration.sha256,loaded.familyRegistration.sha256);
    const invalid=csv.replace('F0S0R1,B0,0,F0S0,F0','F0S0R1,B0,0,F0S0,F9');
    const bad=await post('/api/dataset',{csv:invalid});
    assert.equal(bad.status,400);
    assert.match((await bad.json()).error,/Subject spans clusters/);
  }finally{await new Promise(resolve=>server.close(resolve))}
});
