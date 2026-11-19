import test from 'node:test';
import assert from 'node:assert/strict';
import {spawnSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import {runClusterTrialOutcome,planClusterTrialOutcome,trialOutcomeMarkdown} from '../src/cluster-trial.mjs';
import {invertAffineRandomization,confidenceSetNotation} from '../src/cluster-confidence.mjs';
import {makeServer} from '../server.mjs';

// Frozen before the implementation: integer/dyadic raw outcomes, unequal
// subject counts and technical repetitions, complete allocation coverage.
function fixture({strata=6,clusters=2,mask=0,effect=0,alpha=.05,constant=false}={}) {
  const ds={ids:[],subject:[],cluster:[],batch:[],y:[],X:[],features:['response','other'],
    randomization:{unit:'cluster',mechanism:'uniform_within_batch',protocolId:'CONSTANT-EFFECT-LOCK-001'},
    trialOutcome:{feature:'response',seed:21,perms:129,
      confidenceSet:{model:'constant_additive',alpha}}};
  for(let b=0;b<strata;b++)for(let c=0;c<clusters;c++) {
    const label=c===((mask>>b)&1)?1:0;
    for(let s=0;s<1+(b+c)%3;s++)for(let r=0;r<1+(b*2+c+s)%3;r++) {
      const cluster='B'+b+'C'+c,subject=cluster+'S'+s;
      ds.ids.push(subject+'R'+r);ds.subject.push(subject);ds.cluster.push(cluster);
      ds.batch.push('B'+b);ds.y.push(label);
      ds.X.push([(constant?10:(11*b+7*c+2*s+3*r)/8)+effect*label,(b+c+s+r)%5]);
    }
  }
  return ds;
}
const csv=ds=>['sample_id,batch,label,subject_id,cluster_id,response,other',
  ...ds.ids.map((id,i)=>[id,ds.batch[i],ds.y[i],ds.subject[i],ds.cluster[i],...ds.X[i]].join(','))].join('\n');

test('independent raw-row Fraction oracle certifies every boundary/cell and all-allocation coverage',()=>{
  const cases=[];
  for(const effect of [0,8])for(const alpha of [.05,.2])for(let mask=0;mask<64;mask++) {
    const ds=fixture({effect,alpha,mask});
    cases.push({id:`paired-${effect}-${alpha}-${mask}`,dataset:ds,result:runClusterTrialOutcome(ds),
      trueTau:effect,checkCells:mask===11,coverageGroup:`effect=${effect},alpha=${alpha}`});
  }
  for(const effect of [0,8]) {
    const ds=fixture({strata:2,clusters:4,effect});
    cases.push({id:'unbalanced-'+effect,dataset:ds,result:runClusterTrialOutcome(ds),
      trueTau:effect,checkCells:true});
  }
  const receiver=spawnSync('python3',[fileURLToPath(new URL('../research/constant-effect-confidence-oracle.py',import.meta.url))],
    {input:JSON.stringify(cases),encoding:'utf8',maxBuffer:2**20,timeout:120000});
  assert.equal(receiver.error,undefined,receiver.error?.message);
  assert.equal(receiver.status,0,receiver.stderr);
  const proof=JSON.parse(receiver.stdout);
  assert.equal(proof.cases,258);assert.ok(proof.cells>100);assert.ok(proof.membershipChecks>600);
  for(const group of Object.values(proof.coverage))assert.equal(group.cases,64);
  console.log('Independent confidence receiver:',JSON.stringify(proof));
});

test('exact affine sweep preserves disconnected ranges, isolated roots, degenerate ties and infinity',()=>{
  const disconnected=invertAffineRandomization([{a:0,b:1},{a:1,b:2}],0,.5);
  assert.equal(confidenceSetNotation({status:'available',...disconnected}),'(-infinity, 1/3] union [1, +infinity)');
  assert.equal(disconnected.components.length,2);
  assert.deepEqual(disconnected.cells.map(c=>c.exceedances),[2,2,1,2,2]);
  const point=invertAffineRandomization([{a:0,b:1},{a:0,b:0}],0,.5);
  assert.equal(confidenceSetNotation({status:'available',...point}),'{0}');
  const all=invertAffineRandomization([{a:0,b:1},{a:0,b:-1}],0,.5);
  assert.equal(confidenceSetNotation({status:'available',...all}),'(-infinity, +infinity)');
  assert.equal(all.breakpoints,0);
  assert.throws(()=>invertAffineRandomization([{a:1,b:1}],0,.05),/observed assignment/);
  assert.throws(()=>invertAffineRandomization([{a:0,b:1},{a:Infinity,b:2}],0,.05),/Nonfinite/);
});

test('near-coincident roots remain distinct exact values and decimal-alpha ties reject',()=>{
  const result=invertAffineRandomization([{a:0,b:1},{a:1,b:2},{a:1+Number.EPSILON,b:2}],0,.5);
  assert.equal(result.breakpoints,4,'Do not merge distinct exact roots with a tolerance');
  const tiny=invertAffineRandomization([{a:0,b:1},{a:Number.MIN_VALUE,b:0}],0,.5);
  assert.equal(tiny.breakpoints,2,'Subnormal measurements still have exact distinct signed roots');
  const lines=[...Array(3)].map(()=>({a:0,b:1})).concat([...Array(7)].map(()=>({a:0,b:0})));
  const atPoint=invertAffineRandomization(lines,0,.3);
  assert.equal(confidenceSetNotation({status:'available',...atPoint}),'{0}',
    'p=3/10 must reject at declared alpha=0.3 despite its binary64 rounding');
});

test('constant positive shift can give an isolated exact effect with zero excluded',()=>{
  const ds=fixture({effect:8,constant:true}),result=runClusterTrialOutcome(ds);
  assert.equal(result.p,2/64);
  assert.equal(result.status,'reject_sharp_null');
  assert.equal(confidenceSetNotation(result.confidenceSet),'{8}');
  assert.equal(result.confidenceSet.observedContrast,8);
  assert.match(result.confidenceSet.interpretation,/NOT a confidence interval for an average/);
  assert.match(trialOutcomeMarkdown(result),/Exact set: \{8\}/);
});

test('registration is explicit, nested-frozen, and cannot be added or retuned after intake',()=>{
  const ds=fixture(),locked=planClusterTrialOutcome(ds,ds.trialOutcome);
  assert.ok(Object.isFrozen(locked)&&Object.isFrozen(locked.confidenceSet));
  for(const model of ['average_effect','heterogeneous',''])
    assert.throws(()=>planClusterTrialOutcome(ds,{...ds.trialOutcome,confidenceSet:{model,alpha:.05}}),/SAME additive effect/);
  for(const alpha of [0,1,-.1,NaN,Infinity,'0.05'])
    assert.throws(()=>planClusterTrialOutcome(ds,{...ds.trialOutcome,confidenceSet:{model:'constant_additive',alpha}}),/alpha/);
  assert.throws(()=>planClusterTrialOutcome(ds,{...ds.trialOutcome,confidenceSet:{model:'constant_additive',alpha:.05,grid:100}}),/Preregister/);
  const unregistered=structuredClone(ds);delete unregistered.trialOutcome.confidenceSet;
  assert.equal(runClusterTrialOutcome(unregistered).confidenceSet.status,'not_registered');
  assert.throws(()=>runClusterTrialOutcome(unregistered,ds.trialOutcome),/post-hoc/);
  assert.throws(()=>runClusterTrialOutcome(ds,{...ds.trialOutcome,confidenceSet:{model:'constant_additive',alpha:.2}}),/post-hoc/);
  assert.throws(()=>runClusterTrialOutcome({...ds,randomization:{...ds.randomization,mechanism:'nonuniform'}}),/uniform/);
});

test('large or unidentified orbits never become fabricated exact confidence sets',()=>{
  const large=runClusterTrialOutcome(fixture({strata:10,effect:8}));
  assert.equal(large.pExact,false);assert.equal(large.confidenceSet.exact,false);
  assert.equal(large.confidenceSet.reason,'exact_orbit_exceeds_budget');
  assert.equal(large.confidenceSet.components,undefined);
  const oneClass=fixture();oneClass.y=oneClass.y.map((y,i)=>oneClass.batch[i]==='B0'?1:y);
  const unavailable=runClusterTrialOutcome(oneClass);
  assert.equal(unavailable.status,'unavailable');
  assert.equal(unavailable.confidenceSet.reason,'unidentifiable_stratum');
  const huge=fixture();huge.X.forEach((row,i)=>{row[0]=huge.y[i]?1e308:-1e308;});
  assert.throws(()=>runClusterTrialOutcome(huge),e=>e.code==='TRIAL_NUMERIC_RANGE_UNSUPPORTED');
});

test('HTTP intake locks the effect model and alpha into the digest; JSON and Markdown agree',async()=>{
  const server=makeServer();await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
  const root='http://127.0.0.1:'+server.address().port;
  const post=async(path,data,text=false)=>{
    const r=await fetch(root+path,{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(data)});
    return {code:r.status,data:text?await r.text():await r.json()};
  };
  try {
    const ds=fixture({effect:8,constant:true});
    const input={csv:csv(ds),randomization:ds.randomization,trialOutcome:ds.trialOutcome,family:{ks:[1],seed:31,perms:20}};
    const loaded=await post('/api/dataset',input);assert.equal(loaded.code,200);
    assert.deepEqual(loaded.data.trialOutcomeRegistration.confidenceSet,ds.trialOutcome.confidenceSet);
    const noModel=structuredClone(input);delete noModel.trialOutcome.confidenceSet;
    const basic=await post('/api/dataset',noModel);assert.equal(basic.code,200);
    assert.notEqual(basic.data.familyRegistration.sha256,loaded.data.familyRegistration.sha256);
    const changed=structuredClone(input);changed.trialOutcome.confidenceSet.alpha=.2;
    const alternate=await post('/api/dataset',changed);assert.equal(alternate.code,200);
    assert.notEqual(alternate.data.familyRegistration.sha256,loaded.data.familyRegistration.sha256);
    const result=await post('/api/trial-outcome',{datasetId:loaded.data.datasetId});
    assert.equal(result.code,200);assert.equal(result.data.confidenceSet.status,'available');
    assert.equal(result.data.registrationDigest,loaded.data.familyRegistration.sha256);
    assert.equal(confidenceSetNotation(result.data.confidenceSet),'{8}');
    const after=await post('/api/trial-outcome',{datasetId:basic.data.datasetId,confidenceSet:{model:'constant_additive',alpha:.05}});
    assert.equal(after.code,400);
    const report=await post('/api/report',{datasetId:loaded.data.datasetId,controls:[]},true);
    assert.equal(report.code,200);assert.match(report.data,/### Constant-additive effect confidence set/);
    assert.match(report.data,/Exact set: \{8\}/);assert.match(report.data,/SAME constant additive/);
    const html=await(await fetch(root+'/')).text(),app=await(await fetch(root+'/app.js')).text();
    assert.match(html,/id="trialConstantEffect" disabled/);
    assert.match(html,/same additive effect for every unit/);
    assert.match(app,/Exact endpoints and acceptance rule/);
    assert.match(app,/confidenceSet:\{model:'constant_additive',alpha:0.05\}/);
    const bad=await post('/api/dataset',{...input,trialOutcome:{...input.trialOutcome,confidenceSet:{model:'constant_additive',alpha:0}}});
    assert.equal(bad.code,400);
  } finally {await new Promise(resolve=>server.close(resolve));}
});
