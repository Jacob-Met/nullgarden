import test from 'node:test';
import assert from 'node:assert/strict';
import {runTrialConfidenceSet,invertFisherLines,rationalFisherP} from '../src/fisher-invert.mjs';
import {R0,R1,rint,rat,rfromNumber,rwireToRat,rcmp} from '../src/exact-rational.mjs';
import {makeServer} from '../server.mjs';

// Independent *potential-outcomes* oracle. Assignments are six independent
// matched cluster pairs. Subject outcomes are quarter-integer binary64 values,
// so a separate raw-row brute-force test never shares a floating tie heuristic
// with the candidate rational coefficient/sweep algorithm.
function trial(mask=0,effect=0,strata=6){
  const ids=[],subject=[],cluster=[],batch=[],y=[],X=[];
  for(let b=0;b<strata;b++)for(let c=0;c<2;c++)for(let s=0;s<2;s++)for(let r=0;r<2;r++){
    const cl='B'+b+'C'+c,sub=cl+'S'+s;
    const treatment=c ^ ((mask>>b)&1);
    ids.push(sub+'R'+r);subject.push(sub);cluster.push(cl);
    batch.push('B'+b);y.push(treatment);
    X.push([b*3+c+s*.5+r*.25+effect*treatment,
      b+c*.25+s*.125]);
  }
  return {ids,subject,cluster,batch,y,X,
    features:['response','fingerprint'],
    randomization:{unit:'cluster',mechanism:'uniform_within_batch',
      protocolId:'FROZEN-MATCHED-CLUSTER-TRIAL-001'},
    trialOutcome:{feature:'response',seed:911,perms:79},
    effectModel:'constant_additive'};
}
function legal(ds,mask){
  return ds.y.map((_,i)=>(Math.floor(i/4)%2)^((mask>>Math.floor(i/8))&1));
}
function bruteStatistic(ds,assignment,candidateEffect){
  const strata=new Map();
  for(let i=0;i<ds.ids.length;i++){
    const batch=ds.batch[i],cl=ds.cluster[i],subject=ds.subject[i];
    if(!strata.has(batch))strata.set(batch,new Map());
    if(!strata.get(batch).has(cl))strata.get(batch).set(cl,new Map());
    const subjects=strata.get(batch).get(cl);
    if(!subjects.has(subject))subjects.set(subject,[]);
    subjects.get(subject).push(ds.X[i][0]-candidateEffect*ds.y[i]);
  }
  let stratumTotal=0;
  for(const clusters of strata.values()){
    const yes=[],no=[];
    for(const [cl,subjects] of clusters){
      let value=0;
      for(const readings of subjects.values())
        value+=readings.reduce((a,b)=>a+b,0)/readings.length;
      value/=subjects.size;
      const i=ds.cluster.indexOf(cl);
      (assignment[i]===1?yes:no).push(value);
    }
    assert.ok(yes.length&&no.length);
    const mean=v=>v.reduce((a,b)=>a+b,0)/v.length;
    stratumTotal+=mean(yes)-mean(no);
  }
  return stratumTotal/strata.size;
}
function bruteP(ds,candidateEffect){
  const obs=Math.abs(bruteStatistic(ds,ds.y,candidateEffect));
  let count=0;
  for(let mask=0;mask<64;mask++){
    const stat=Math.abs(bruteStatistic(ds,legal(ds,mask),candidateEffect));
    if(stat>=obs)count++;
  }
  return {count,p:count/64};
}
function includesWire(intervals,effect){
  const n=BigInt(effect*4),d=4n; // quarter-integer held-out effects.
  return intervals.some(v=>{
    if(v.lower){
      const delta=n*BigInt(v.lower.denominator)-BigInt(v.lower.numerator)*d;
      if(delta<0n||(delta===0n&&!v.includeLower))return false;
    }
    if(v.upper){
      const delta=n*BigInt(v.upper.denominator)-BigInt(v.upper.numerator)*d;
      if(delta>0n||(delta===0n&&!v.includeUpper))return false;
    }
    return true;
  });
}
function fracEquals(w,n,d=1n){
  return w!==null&&BigInt(w.numerator)*d===n*BigInt(w.denominator);
}
function csv(ds){
  return ['sample_id,batch,label,subject_id,cluster_id,response,fingerprint',
    ...ds.ids.map((id,i)=>[id,ds.batch[i],ds.y[i],
      ds.subject[i],ds.cluster[i],...ds.X[i]].join(','))].join('\n');
}

test('rational substrate exactly represents IEEE decimals, subnormals, signed-zero and arbitrary fractions',()=>{
  assert.deepEqual(rfromNumber(0.1),rat(3602879701896397n,36028797018963968n));
  assert.deepEqual(rfromNumber(-0),R0);
  assert.deepEqual(rfromNumber(Number.MIN_VALUE),rat(1n,1n<<1074n));
  assert.deepEqual(rfromNumber(1.5),rat(3n,2n));
  const h=rat(-41n,13n);
  assert.equal(rcmp(rwireToRat({numerator:'-41',denominator:'13'}),h),0);
  assert.throws(()=>rfromNumber(Infinity),/non-finite/);
});

test('algebraic event sweep retains signed line ties at closed edges, isolated points and disconnected sets',()=>{
  const constant=(a,b)=>({a:rint(a),b:rint(b)});
  const observed=constant(0,1),complement=constant(0,-1);
  const lines=[observed,complement,...Array.from({length:62},()=>constant(1,0))];
  const interval=invertFisherLines(lines,0);
  assert.equal(interval.status,'exact');
  assert.equal(interval.orbitSize,64);
  assert.equal(interval.nComponents,1);
  assert.ok(fracEquals(interval.intervals[0].lower,-1n));
  assert.ok(fracEquals(interval.intervals[0].upper,1n));
  assert.equal(interval.intervals[0].includeLower,true);
  assert.equal(interval.intervals[0].includeUpper,true);
  assert.equal(rationalFisherP(lines,observed,rint(2)).p,2/64);
  assert.equal(rationalFisherP(lines,observed,rint(1)).p,1);

  const isolated=invertFisherLines([observed,complement,
    ...Array.from({length:62},()=>constant(0,0))],0);
  assert.equal(isolated.nComponents,1);
  assert.ok(fracEquals(isolated.intervals[0].lower,0n));
  assert.ok(fracEquals(isolated.intervals[0].upper,0n));
  assert.equal(isolated.intervals[0].includeLower,true);
  assert.equal(isolated.intervals[0].includeUpper,true);

  // This synthetic coefficient b=2 lies outside legal cluster-trial [-1,1]
  // but is an adversarial algebraic solver control: accepting only a single
  // min/max interval would silently fill an unsupported gap.
  const disconnected=invertFisherLines([observed,complement,
    ...Array.from({length:62},()=>constant(10,2))],0);
  assert.equal(disconnected.nComponents,2);
  assert.equal(disconnected.intervals[0].lower,null);
  assert.ok(fracEquals(disconnected.intervals[0].upper,10n,3n));
  assert.ok(fracEquals(disconnected.intervals[1].lower,10n));
  assert.equal(disconnected.intervals[1].upper,null);
  assert.ok(!includesWire(disconnected.intervals,7));
  assert.ok(includesWire(disconnected.intervals,0));
  assert.ok(includesWire(disconnected.intervals,12));
});

test('independent 64-mask brute-force p agrees with exact inverted confidence-set membership across a frozen effect grid',()=>{
  const ds=trial(0,5),run=runTrialConfidenceSet(ds);
  assert.equal(run.status,'exact');
  assert.equal(run.confidenceLevel,.95);
  assert.equal(run.orbitSize,64);
  assert.ok(run.nBreakpoints>0);
  assert.ok(run.nComponents>=1);
  assert.ok(includesWire(run.intervals,5));
  assert.ok(!includesWire(run.intervals,0),
    'predeclared strong treatment response must exclude no-effect sharp null');
  const oracleAtZero=bruteP(ds,0);
  assert.equal(run.pAtZero.exceedances,oracleAtZero.count);
  assert.equal(run.pAtZero.p,oracleAtZero.p);
  for(let candidate=-14;candidate<=18;candidate+=.5){
    const direct=bruteP(ds,candidate);
    const retained=direct.p>.05;
    assert.equal(includesWire(run.intervals,candidate),retained,
      'exact inversion and independent full randomization disagree at candidate effect '+candidate);
  }
});

test('complete finite randomization coverage for constant additive effect across ALL 64 observed assignments',()=>{
  const trueEffect=3;
  let nonCoverage=0;
  for(let observedMask=0;observedMask<64;observedMask++){
    const ds=trial(observedMask,trueEffect),res=runTrialConfidenceSet(ds);
    const oracle=bruteP(ds,trueEffect);
    assert.equal(res.status,'exact');
    if(!includesWire(res.intervals,trueEffect))nonCoverage++;
    assert.equal(includesWire(res.intervals,trueEffect),oracle.p>.05);
  }
  assert.ok(nonCoverage<=3,
    '95% finite conditional coverage: no more than 3 of 64 legal assignments exclude the truth');
});

test('large orbit, unbalanced/nonidentified strata and unsupported model abstain without fake exact CI',()=>{
  const big=trial(0,2,10),unavailable=runTrialConfidenceSet(big);
  assert.equal(unavailable.status,'unavailable');
  assert.equal(unavailable.orbitSizeLowerBound,513);
  assert.equal(unavailable.intervals,null);
  const obs=trial();
  assert.throws(()=>runTrialConfidenceSet({...obs,effectModel:undefined}),
    /explicitly opted in/);
  const oneClass=trial();
  for(let i=0;i<8;i++)oneClass.y[i]=1;
  const none=runTrialConfidenceSet(oneClass);
  assert.equal(none.status,'unavailable');
  assert.equal(none.intervals,null);
  assert.match(none.reason,/not present in every/);
});

test('cold HTTP locks model and confidence level at dataset intake; report includes exact rational set',async()=>{
  const server=makeServer();await new Promise(r=>server.listen(0,'127.0.0.1',r));
  const root='http://127.0.0.1:'+server.address().port;
  const post=async(path,body,asText=false)=>{
    const res=await fetch(root+path,{method:'POST',
      headers:{'content-type':'application/json'},body:JSON.stringify(body)});
    return {status:res.status,data:asText?await res.text():await res.json()};
  };
  try{
    const ds=trial(0,6);
    const request={csv:csv(ds),randomization:ds.randomization,
      trialOutcome:ds.trialOutcome,effectModel:'constant_additive',
      family:{ks:[1,2],seed:21,perms:40}};
    const accepted=await post('/api/dataset',request);
    assert.equal(accepted.status,200);
    assert.equal(accepted.data.effectModel,'constant_additive');
    const result=await post('/api/trial-confidence',{datasetId:accepted.data.datasetId});
    assert.equal(result.status,200);
    assert.equal(result.data.status,'exact');
    assert.equal(result.data.confidenceLevel,.95);
    assert.equal(result.data.registrationDigest,accepted.data.familyRegistration.sha256);
    assert.ok(includesWire(result.data.intervals,6));
    const noModel=await post('/api/dataset',{...request,effectModel:undefined});
    assert.equal(noModel.status,200);
    assert.notEqual(noModel.data.familyRegistration.sha256,accepted.data.familyRegistration.sha256);
    const forbidden=await post('/api/trial-confidence',{datasetId:noModel.data.datasetId});
    assert.equal(forbidden.status,400);
    const posthoc=await post('/api/trial-confidence',{
      datasetId:accepted.data.datasetId,effectModel:'constant_additive',confidenceLevel:.99});
    assert.equal(posthoc.status,400);
    const posthocRun=await post('/api/run',{
      datasetId:noModel.data.datasetId,effectModel:'constant_additive'});
    assert.equal(posthocRun.status,400);
    const invalid=await post('/api/dataset',{...request,effectModel:'average_effect'});
    assert.equal(invalid.status,400);
    const invalidNoOutcome=await post('/api/dataset',
      {...request,trialOutcome:undefined});
    assert.equal(invalidNoOutcome.status,400);
    const report=await post('/api/report',{datasetId:accepted.data.datasetId,
      controls:['within_batch_permutation']},true);
    assert.equal(report.status,200);
    assert.ok(report.data.includes('## Preregistered constant-additive Fisher confidence set'));
    assert.ok(report.data.includes('95% confidence SET across 64 full randomized assignments'));
  }finally{await new Promise(r=>server.close(r));}
});
