import test from 'node:test';
import assert from 'node:assert/strict';
import {stateOperatorWorlds,stateOperatorPlan,observeStateOperatorWorld,outsideStateOperatorWorlds} from '../src/state-operator-course.mjs';
test('twenty distinct operator laws and four genuine omitted mechanisms have a fixed roster',()=>{
  const roster=stateOperatorWorlds();assert.equal(roster.length,20);assert.equal(new Set(roster.map(w=>w.id)).size,20);
  assert.equal(roster.filter(w=>w.kind==='periodic').length,8);
  assert.deepEqual(outsideStateOperatorWorlds().map(w=>w.id),['outside_delay','outside_projected_lorenz','outside_nonlocal','outside_dispersion']);
});
test('independent heat reference matches its exact discrete Fourier evolution and refines',()=>{
  const world=stateOperatorWorlds('development').find(w=>w.id==='dev_heat');
  const plan=stateOperatorPlan('periodic','train0'),errors=[];
  for(const substeps of [12,24]) {
    const {reference}=observeStateOperatorWorld(world,plan,40,11,{substeps});
    let worst=0;
    for(let i=0;i<reference.states.length;i++)for(let j=0;j<32;j++) {
      const t=i*plan.step,k=2*Math.PI/32,h=plan.spacing;
      const a=Math.exp(-.13*4*Math.sin(k/2)**2*t/(h*h));
      const b=Math.exp(-.13*4*Math.sin(k)**2*t/(h*h));
      const exact=.65+.25*a*Math.sin(k*j+.1)+.12*b*Math.cos(2*k*j);
      worst=Math.max(worst,Math.abs(reference.states[i][j]-exact));
    }
    errors.push(worst);
  }
  assert.ok(errors[1]<1e-8&&errors[1]<errors[0]/3.9,JSON.stringify(errors));
});
test('state observations expose no exact derivative or clean initial state, and noise reproduces',()=>{
  const w=stateOperatorWorlds('development')[0],p=stateOperatorPlan(w.kind);
  const a=observeStateOperatorWorld(w,p,25,531),b=observeStateOperatorWorld(w,p,25,531);
  assert.deepEqual(a,b);assert.notDeepEqual(a.observation.samples[0].state,p.initial);
  for(const s of a.observation.samples)assert.deepEqual(Object.keys(s),['time','state','command']);
  assert.deepEqual(Object.keys(a.observation),['schema','id','kind','samples']);
  let error=0,signal=0;
  for(let i=0;i<a.reference.states.length;i++)for(let j=0;j<2;j++){error+=(a.observation.samples[i].state[j]-a.reference.states[i][j])**2;signal+=a.reference.states[i][j]**2;}
  assert.ok(Math.abs(10*Math.log10(signal/error)-25)<1);
});
test('delay and hidden-state observations keep their complete internal dynamics outside fitting input',()=>{
  for(const w of outsideStateOperatorWorlds().slice(0,2)) {
    const p={...stateOperatorPlan(w.kind),steps:40};
    const r=observeStateOperatorWorld(w,p,25,17);
    assert.equal(r.observation.samples.length,41);
    assert.ok(r.observation.samples.every(s=>s.state.length===2&&s.state.every(Number.isFinite)));
    assert.equal(Object.hasOwn(r.observation,'rhs'),false);
  }
});
