import test from 'node:test';
import assert from 'node:assert/strict';
import {conditionalOrbitSize, conditionalLabelAssignments, sampleConditionalLabels,
  runControls, repeatedCv, fitScore, rng, toMarkdown} from '../src/core.mjs';
import {runFamilywiseK} from '../src/familywise.mjs';
import {makeServer} from '../server.mjs';

// LOCKED BEFORE candidate evaluation: six randomized strata, two whole
// treatment clusters per stratum, two related subjects with two acquisitions
// each per cluster. All CV folds must exclude complete clusters. The feature
// signals are arbitrary fixed measurements under the sharp randomization null;
// finite-sample Type-I validity is tested over ALL 2^6 assignments.
function trial() {
  const ids=[],subject=[],cluster=[],batch=[],y=[],X=[];
  for(let b=0;b<6;b++)for(let c=0;c<2;c++)for(let s=0;s<2;s++)for(let r=0;r<2;r++){
    const id='B'+b+'C'+c,sub=id+'S'+s;
    ids.push(sub+'R'+r);
    subject.push(sub);cluster.push(id);batch.push('B'+b);y.push(c);
    X.push([(c?3.8:-3.8)+b*.013+s*.021+r*.002,
      Math.sin(b*0.713+c*0.41+s*0.37)+r*.001]);
  }
  return {ids,subject,cluster,batch,y,X,features:['signal','fingerprint'],
    randomization:{unit:'cluster',mechanism:'uniform_within_batch',protocolId:'TRIAL-LOCKED-001'}};
}

// An INDEPENDENT oracle: bit m flips which cluster gets treatment inside
// stratum m. It never calls the production enumerator or sampler.
function oracleLabels(ds,mask) {
  return ds.y.map((_,i) => (Math.floor(i/4)%2) ^ ((mask >> Math.floor(i/8))&1));
}
function statistic(ds,labels,k=1,seed=31){
  return repeatedCv({...ds,y:labels},(tr,te)=>
    fitScore(tr.map(i=>ds.X[i]),tr.map(i=>labels[i]),
      te.map(i=>ds.X[i]),k),{seed,repeats:3,folds:5});
}
function csv(ds) {
  return ['sample_id,batch,label,subject_id,cluster_id,signal,fingerprint',
    ...ds.ids.map((id,i)=>[id,ds.batch[i],ds.y[i],ds.subject[i],ds.cluster[i],...ds.X[i]].join(','))].join('\n');
}

test('whole-cluster randomized orbit matches independent 64-vertex oracle and finite-sample Type-I envelope',()=>{
  const ds=trial(),before=ds.y.slice();
  assert.equal(conditionalOrbitSize(ds.batch,ds.y,512,ds.subject,ds.cluster),1,
    'dependency clusters alone must NEVER imply experimental randomization');
  assert.equal(conditionalOrbitSize(ds.batch,ds.y,512,ds.subject,ds.cluster,'cluster'),64);
  const legal=[...conditionalLabelAssignments(ds.batch,ds.y,
    {maxAssignments:512,subject:ds.subject,cluster:ds.cluster,assignmentUnit:'cluster'})];
  const independently=[...Array(64).keys()].map(mask=>oracleLabels(ds,mask));
  assert.equal(legal.length,64);
  assert.deepEqual(new Set(legal.map(x=>x.join(''))),
    new Set(independently.map(x=>x.join(''))));
  assert.equal(legal.filter(x=>x.every((v,i)=>v===ds.y[i])).length,1);
  assert.deepEqual(ds.y,before);
  const scores=independently.map(x=>statistic(ds,x));
  assert.ok(scores.every(Number.isFinite),'complete 64-orbit reference statistic must be identified');
  const actual=statistic(ds,ds.y),p=scores.filter(x=>x>=actual).length/64;
  const result=runControls(ds,['within_batch_permutation'],{seed:31,k:1,perms:127});
  const got=result.results.within_batch_permutation;
  assert.equal(result.unitMode,'registered_cluster_randomization');
  assert.equal(result.nClusters,12);
  assert.equal(got.conditioningLevel,'cluster_randomization');
  assert.equal(got.nullSampling,'complete_cluster_randomization_orbit');
  assert.equal(got.orbitSize,64);
  assert.equal(got.pExact,true);
  assert.equal(got.p,p);
  assert.equal(got.permsRun,64);
  for(const alpha of [.01,.05,.1,.2,.5,1]){
    const falseRejections=scores.filter(x=>
      scores.filter(v=>v>=x).length/64<=alpha).length/64;
    assert.ok(falseRejections<=alpha+1e-12,
      'enumerated randomization exact p must be superuniform');
  }
  const report=toMarkdown('trial',result);
  assert.match(report,/Registered uniform whole-cluster treatment assignment/);
  assert.match(report,/not independent proof/i);
});

test('independent clusters never split in either exact or sampled assignments',()=>{
  const ds=trial(),sample=sampleConditionalLabels(ds.batch,ds.y,rng(19),
    ds.subject,ds.cluster,'cluster');
  for(let i=0;i<sample.length;i+=4){
    assert.deepEqual(sample.slice(i,i+4),Array(4).fill(sample[i]));
  }
  for(let b=0;b<6;b++){
    const first=sample[b*8],second=sample[b*8+4];
    assert.notEqual(first,second,'stratum treatment cluster count is preserved');
  }
  const observations=runControls({...ds,randomization:undefined},
    ['within_batch_permutation'],{seed:31,k:1,perms:127});
  const p=observations.results.within_batch_permutation;
  assert.equal(p.nullSampling,'no_nested_cluster_contrast');
  assert.equal(p.status,'unavailable');
  assert.equal(p.p,null);
});

test('refuse unsupported experimental mechanism, absent metadata, and mixed-label treatment clusters',()=>{
  const ds=trial();
  assert.throws(()=>runControls({...ds,randomization:{unit:'cluster'}},[],{seed:31}),
    e=>e.code==='RANDOMIZATION_NOT_REGISTERED');
  assert.throws(()=>runControls({...ds,cluster:undefined},[],{seed:31}),
    e=>e.code==='CLUSTER_ASSIGNMENT_METADATA_MISSING');
  const mixed={...ds,y:ds.y.slice()};
  mixed.y[2]=mixed.y[3]=1;
  assert.throws(()=>conditionalOrbitSize(mixed.batch,mixed.y,512,
    mixed.subject,mixed.cluster,'cluster'),e=>e.code==='CLUSTER_LABEL_MIXED');
  assert.throws(()=>runControls(mixed,[],{seed:31}),e=>e.code==='CLUSTER_LABEL_MIXED');
});

test('immutable HTTP trial registration, familywise inference and no post-hoc promotion',async()=>{
  const server=makeServer();await new Promise(r=>server.listen(0,'127.0.0.1',r));
  const root='http://127.0.0.1:'+server.address().port;
  const post=async(path,b)=>{
    const res=await fetch(root+path,{method:'POST',
      headers:{'content-type':'application/json'},body:JSON.stringify(b)});
    return {code:res.status,data:await res.json()};
  };
  try {
    const ds=trial(),body={csv:csv(ds),name:'locked cluster trial',
      randomization:ds.randomization,family:{ks:[1,2],seed:31,perms:127}};
    const loaded=await post('/api/dataset',body);
    assert.equal(loaded.code,200);
    assert.deepEqual(loaded.data.randomization,ds.randomization);
    assert.equal(loaded.data.nestedClusterMode,'cluster_disjoint_whole_cluster_randomization');
    const ordinary=await post('/api/dataset',{csv:csv(ds),
      family:{ks:[1,2],seed:31,perms:127}});
    assert.equal(ordinary.code,200);
    assert.notEqual(ordinary.data.familyRegistration.sha256,
      loaded.data.familyRegistration.sha256,'scientific assumption must be locked into digest');
    const run=await post('/api/run',{datasetId:loaded.data.datasetId,
      controls:['within_batch_permutation']});
    assert.equal(run.code,200);
    assert.equal(run.data.results.within_batch_permutation.orbitSize,64);
    assert.equal(run.data.results.within_batch_permutation.pExact,true);
    const family=await post('/api/family',{datasetId:loaded.data.datasetId});
    assert.equal(family.code,200);
    assert.equal(family.data.registrationDigest,loaded.data.familyRegistration.sha256);
    assert.equal(family.data.status,'valid');
    assert.equal(family.data.tests.length,2);
    assert.ok(family.data.tests.every(t=>t.adjustedP>=t.rawP));
    const unregistered=await post('/api/dataset',{csv:csv(ds),
      randomization:{unit:'cluster'}});
    assert.equal(unregistered.code,400);
    const mixed={...ds,y:ds.y.slice()};
    mixed.y[2]=mixed.y[3]=1;
    const bad=await post('/api/dataset',{csv:csv(mixed),randomization:ds.randomization});
    assert.equal(bad.code,400);
    assert.match(bad.data.error,/mixed treatment labels/i);
  } finally {await new Promise(r=>server.close(r));}
});
