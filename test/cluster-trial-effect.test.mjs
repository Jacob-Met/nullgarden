import test from 'node:test';
import assert from 'node:assert/strict';
import {runClusterTrialOutcome,planClusterTrialOutcome,trialOutcomeMarkdown} from '../src/cluster-trial.mjs';
import {makeServer} from '../server.mjs';

// Prelocked design: 6 strata, 2 independent treatment clusters per stratum,
// 2 subjects per cluster, 2 repeated outcomes per subject. Under the sharp
// null, each cluster's fixed measured outcome is unchanged when its treatment
// is reassigned. A separate 2^6 bitmask independently enumerates every
// randomized treatment assignment, without using the candidate enumerator.
function trial(strata=6,effect=0) {
  const ids=[],subject=[],cluster=[],batch=[],y=[],X=[];
  for(let b=0;b<strata;b++)for(let c=0;c<2;c++)for(let s=0;s<2;s++)for(let r=0;r<2;r++) {
    const group='B'+b+'C'+c,sub=group+'S'+s;
    ids.push(sub+'R'+r);subject.push(sub);cluster.push(group);batch.push('B'+b);y.push(c);
    const baseline=5*Math.sin(b*.3)+Math.cos(b*.718+c*1.37)+s*.045+r*.007;
    X.push([baseline+effect*c,Math.cos(b*2.3+c*.83+s*.4)+r*.006]);
  }
  return {ids,subject,cluster,batch,y,X,features:['response','fingerprint'],
    randomization:{unit:'cluster',mechanism:'uniform_within_batch',protocolId:'RANDOMIZED-CLUSTER-TRIAL-001'},
    trialOutcome:{feature:'response',seed:21,perms:129}};
}
const plan={feature:'response',seed:21,perms:129};
function flippedLabels(ds,mask){
  return ds.y.map((_,i)=>((Math.floor(i/4)%2)^((mask>>Math.floor(i/8))&1)));
}
function independentStatistic(ds,labels){
  // A deliberately independent raw-row reducer: group by batch + cluster +
  // subject, average technical repeats inside subject, average subjects inside
  // cluster, then equal-weight treated/control cluster means within each batch.
  const strata=new Map();
  for(let i=0;i<ds.ids.length;i++){
    const key=ds.batch[i],cluster=ds.cluster[i],subject=ds.subject[i];
    if(!strata.has(key))strata.set(key,new Map());
    if(!strata.get(key).has(cluster))strata.get(key).set(cluster,new Map());
    const subjects=strata.get(key).get(cluster);
    if(!subjects.has(subject))subjects.set(subject,[]);
    subjects.get(subject).push({outcome:ds.X[i][0],label:labels[i]});
  }
  let total=0;
  for(const groups of strata.values()){
    const yes=[],no=[];
    for(const subjects of groups.values()){
      const first=[...subjects.values()][0][0].label;
      const means=[...subjects.values()].map(rows=>{
        assert.ok(rows.every(r=>r.label===first));
        return rows.reduce((s,r)=>s+r.outcome,0)/rows.length;
      });
      const clusterMean=means.reduce((a,b)=>a+b,0)/means.length;
      (first?yes:no).push(clusterMean);
    }
    assert.ok(yes.length&&no.length);
    const avg=a=>a.reduce((x,y)=>x+y,0)/a.length;
    total+=avg(yes)-avg(no);
  }
  return total/strata.size;
}
function csv(ds){
  return ['sample_id,batch,label,subject_id,cluster_id,response,fingerprint',
    ...ds.ids.map((id,i)=>[id,ds.batch[i],ds.y[i],ds.subject[i],ds.cluster[i],...ds.X[i]].join(','))].join('\n');
}
function equal(a,b){return Math.abs(a-b)<=1e-10*Math.max(1,Math.abs(a),Math.abs(b));}

test('exact six-stratum Fisher test agrees with independent 64-orbit treatment-effect oracle',()=>{
  const ds=trial(),reference=[...Array(64).keys()].map(mask=>
    independentStatistic(ds,flippedLabels(ds,mask)));
  const observed=independentStatistic(ds,ds.y);
  const independentP=reference.filter(t=>Math.abs(t)+1e-12>=Math.abs(observed)).length/64;
  const result=runClusterTrialOutcome(ds);
  assert.equal(result.method,'Fisher cluster-randomization test of sharp no-effect null');
  assert.ok(equal(result.observedContrast,observed));
  assert.equal(result.nullSampling,'complete_cluster_randomization_orbit');
  assert.equal(result.orbitSize,64);
  assert.equal(result.assignmentsEvaluated,64);
  assert.equal(result.pExact,true);
  assert.equal(result.p,independentP);
  assert.equal(result.nClusters,12);
  assert.equal(result.nSubjects,24);
  assert.equal(result.nStrata,6);
  assert.match(result.causalLimit,/SHARP null/);
  assert.match(trialOutcomeMarkdown(result),/Primary outcome: response/);
  // Enumerate EVERY possible null-observed assignment and exact p under
  // immutable outcomes; no stochastic Type-I estimate or weak p-value claim.
  for(const alpha of [.01,.05,.1,.2,.5,1]){
    const type1=reference.filter(t=>reference.filter(other=>
      Math.abs(other)+1e-12>=Math.abs(t)).length/64<=alpha).length/64;
    assert.ok(type1<=alpha+1e-12,'full 64-case null Type-I envelope must be superuniform');
  }
});

test('actual shifted outcome in independently randomized clusters gives a detectable Fisher contrast',()=>{
  const ds=trial(6,9),result=runClusterTrialOutcome(ds);
  assert.ok(result.observedContrast>8);
  assert.ok(result.p<=.05,'6-stratum positive control should exceed finite randomization null');
  assert.equal(result.status,'reject_sharp_null');
  assert.equal(result.pExact,true);
  assert.ok(result.p>=2/64,'two-sided symmetry cannot claim impossible p<2/64');
});

test('unequal cluster size and unequal technical repeats do not weight treatment allocations by row count',()=>{
  const ds=trial();
  // One subject gets three additional highly shifted technical replicates;
  // outcome changes within that subject, but its cluster remains one unit.
  const i=0;
  for(let j=0;j<3;j++){
    ds.ids.push('repeat-'+j);
    ds.subject.push(ds.subject[i]);ds.cluster.push(ds.cluster[i]);
    ds.batch.push(ds.batch[i]);ds.y.push(ds.y[i]);
    ds.X.push([ds.X[i][0]+2+j,ds.X[i][1]]);
  }
  const result=runClusterTrialOutcome(ds);
  const reference=independentStatistic(ds,ds.y);
  assert.ok(equal(result.observedContrast,reference));
  assert.equal(result.orbitSize,64,'extra rows are NOT randomization units');
  assert.equal(result.nSubjects,24);
  assert.equal(result.nClusters,12);
  assert.equal(result.assignmentsEvaluated,64);
});

test('1024-allocation trial follows frozen Monte Carlo +1, never pretends it is exact enumeration',()=>{
  const ds=trial(10,4);
  const a=runClusterTrialOutcome(ds),b=runClusterTrialOutcome(ds);
  assert.deepEqual(a,b);
  assert.equal(a.nullSampling,'predeclared_monte_carlo_plus_one');
  assert.equal(a.orbitSizeLowerBound,513);
  assert.equal(a.orbitSizeExact,false);
  assert.equal(a.pExact,false);
  assert.equal(a.assignmentsEvaluated,129);
  assert.ok(a.p>=1/130&&a.p<=1);
});

test('huge finite outcomes never overflow summation into fabricated p-values',()=>{
  const same=trial();
  same.X.forEach(row=>{row[0]=1e308;});
  const result=runClusterTrialOutcome(same);
  assert.equal(result.observedContrast,0);
  assert.equal(result.p,1);
  assert.equal(result.orbitSize,64);
  const outside=trial();
  outside.X.forEach((row,i)=>{row[0]=outside.y[i]?1e308:-1e308;});
  assert.throws(()=>runClusterTrialOutcome(outside),
    error=>error.code==='TRIAL_NUMERIC_RANGE_UNSUPPORTED');
});

test('registration refuses post-hoc or invalid outcome, mechanism, and one-class strata without silent selection',()=>{
  const ds=trial();
  assert.throws(()=>planClusterTrialOutcome({...ds,randomization:null},plan),/preregistered/);
  assert.throws(()=>planClusterTrialOutcome(ds,{feature:'missing',seed:21,perms:129}),/exactly one/);
  assert.throws(()=>planClusterTrialOutcome(ds,{feature:'response',seed:21,perms:0}),/budget/);
  const oneClass={...ds,y:ds.y.slice()};
  for(let i=0;i<8;i++)oneClass.y[i]=1;
  const result=runClusterTrialOutcome(oneClass);
  assert.equal(result.status,'unavailable');
  assert.equal(result.observedContrast,null);
  assert.equal(result.p,null);
  assert.equal(result.nullSampling,'unidentifiable_stratum');
});

test('HTTP intake locks primary outcome with digest; separate result and report survive cold',async()=>{
  const server=makeServer();await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
  const root='http://127.0.0.1:'+server.address().port;
  const post=async(path,body,accept='json')=>{
    const response=await fetch(root+path,{method:'POST',
      headers:{'content-type':'application/json'},body:JSON.stringify(body)});
    return {code:response.status,data:accept==='text'?await response.text():await response.json()};
  };
  try{
    const ds=trial();
    const input={csv:csv(ds),randomization:ds.randomization,trialOutcome:ds.trialOutcome,
      family:{ks:[1,2],seed:31,perms:40}};
    const loaded=await post('/api/dataset',input);
    assert.equal(loaded.code,200);
    assert.deepEqual(loaded.data.trialOutcomeRegistration,ds.trialOutcome);
    const noOutcome=await post('/api/dataset',{...input,trialOutcome:undefined});
    assert.equal(noOutcome.code,200);
    assert.notEqual(loaded.data.familyRegistration.sha256,noOutcome.data.familyRegistration.sha256);
    const result=await post('/api/trial-outcome',{datasetId:loaded.data.datasetId});
    assert.equal(result.code,200);
    assert.equal(result.data.pExact,true);
    assert.equal(result.data.orbitSize,64);
    assert.equal(result.data.registrationDigest,loaded.data.familyRegistration.sha256);
    const missing=await post('/api/trial-outcome',{datasetId:noOutcome.data.datasetId});
    assert.equal(missing.code,400);
    const changed=await post('/api/trial-outcome',{datasetId:loaded.data.datasetId,
      trialOutcome:{feature:'fingerprint',seed:21,perms:129}});
    assert.equal(changed.code,400);
    const upgrade=await post('/api/run',{datasetId:noOutcome.data.datasetId,
      trialOutcome:ds.trialOutcome});
    assert.equal(upgrade.code,400);
    const text=await post('/api/report',{datasetId:loaded.data.datasetId,
      controls:['within_batch_permutation']},'text');
    assert.equal(text.code,200);
    assert.match(text.data,/Preregistered cluster-trial primary outcome/);
    assert.match(text.data,/SHARP null/);
    const invalid=await post('/api/dataset',{...input,trialOutcome:{feature:'unknown',seed:21,perms:129}});
    assert.equal(invalid.code,400);
    assert.match(invalid.data.error,/primary outcome/i);
  }finally{await new Promise(resolve=>server.close(resolve));}
});
