import {pathToFileURL} from 'node:url';

// Independent simulator: no Worldlaw imports, no derivative labels in observations.
export const STATE_OPERATOR_PROTOCOL=Object.freeze({
  schema:'nullgarden.state-operator-course/v1',snrDb:[40,25,15],
  referenceIntegrator:'explicit midpoint, 24 substeps per observation',
  trainingQueries:2,selectionQueries:1,additionalQueries:1,testQueries:1,
  noise:'independent Gaussian per state channel; RMS signal SNR',
  testUse:'frozen scoring only; model selection uses a different whole recording',
  failurePenalty:10,relativeErrorFloor:1e-4,
  qualification:'synthetic semidiscrete laws; no physical or calibrated OOD certification'
});
function rng(seed){let x=seed>>>0;return()=>{x^=x<<13;x^=x>>>17;x^=x<<5;return((x>>>0)+.5)/4294967296;};}
function normal(random){return Math.sqrt(-2*Math.log(random()))*Math.cos(2*Math.PI*random());}
const mul=(a,b)=>a.map((v,j)=>v*b[j]);
const dx=(a,h)=>a.map((_,j)=>(a[(j+1)%a.length]-a[(j+a.length-1)%a.length])/(2*h));
const dxx=(a,h)=>a.map((v,j)=>(a[(j+1)%a.length]-2*v+a[(j+a.length-1)%a.length])/(h*h));
const combine=(...parts)=>parts[0].map((_,j)=>parts.reduce((s,a)=>s+a[j],0));
const scale=(a,c)=>a.map(v=>v*c);

const oscillator=[
 ['duffing_double_well',(x,v,u)=>.8*x-.55*x**3-.2*v+.65*u],
 ['van_der_pol',(x,v,u)=>.45*v-.45*x*x*v-x+.7*u],
 ['rayleigh',(x,v,u)=>.35*v-.2*v**3-x+.7*u],
 ['cubic_drag',(x,v,u)=>-x-.25*v-.15*v**3+.7*u],
 ['quadratic_drag_pendulum',(x,v,u)=>-Math.sin(x)-.2*Math.abs(v)*v+.7*u],
 ['saturating_restoring',(x,v,u)=>-Math.tanh(x)-.15*v+.7*u],
 ['saturating_drag',(x,v,u)=>-x-.35*Math.tanh(v)+.7*u],
 ['rational_restoring',(x,v,u)=>-1.5*x/(1+x*x)-.2*v+.6*u],
 ['parametric_stiffness',(x,v,u)=>-x-.2*v+.35*u*x+.3*u],
 ['phase_modulated_forcing',(x,v,u)=>-Math.sin(x)-.2*v+.7*u*Math.cos(x)],
 ['position_dependent_drag',(x,v,u)=>-x-.15*v-.25*x*x*v+.6*u],
 ['asymmetric_trigonometric',(x,v,u)=>-Math.sin(x)-.2*x*Math.sin(x)-.15*v+.6*u]
];
const periodic=[
 ['linear_reaction_diffusion',(a,u,h)=>combine(scale(a,-.3),scale(dxx(a,h),.1),scale(u,.2))],
 ['fisher_kpp',(a,u,h)=>combine(scale(a,.35),scale(mul(a,a),-.35),scale(dxx(a,h),.08))],
 ['allen_cahn',(a,u,h)=>combine(scale(a,.5),scale(a.map(v=>v**3),-.5),scale(dxx(a,h),.1))],
 ['cubic_conservation',(a,u,h)=>combine(scale(dx(a.map(v=>v**3),h),-.2),scale(dxx(a,h),.08))],
 ['forced_viscous_burgers',(a,u,h)=>combine(scale(dx(mul(a,a),h),-.5),scale(dxx(a,h),.1),scale(u,.15))],
 ['porous_diffusion',(a,u,h)=>combine(scale(dxx(a.map(v=>v**3),h),.04),scale(dxx(a,h),.05))],
 ['sine_reaction_diffusion',(a,u,h)=>combine(scale(a.map(Math.sin),-.45),scale(dxx(a,h),.08),scale(u,.2))],
 ['saturating_flux',(a,u,h)=>combine(scale(dx(a.map(Math.tanh),h),-.4),scale(dxx(a,h),.1),scale(u,.2))]
];
const development=[
 {id:'dev_harmonic',kind:'oscillator',rhs:(a,u)=>[a[1],-1.3*a[0]-.17*a[1]+.6*u]},
 {id:'dev_input_product',kind:'oscillator',rhs:(a,u)=>[a[1],u*Math.cos(a[0])-.2*a[1]]},
 {id:'dev_heat',kind:'periodic',rhs:(a,u,h)=>scale(dxx(a,h),.13)},
 {id:'dev_advection',kind:'periodic',rhs:(a,u,h)=>combine(scale(dx(a,h),-.45),scale(dxx(a,h),.07),scale(u,.15))}
];

export function stateOperatorWorlds(split='evaluation',seed=927341) {
  if(split==='development')return development;
  if(split!=='evaluation')throw new RangeError('Unknown course split');
  const random=rng(seed);
  return [...oscillator.map(([id,f])=>({id,kind:'oscillator',rhs:(a,u)=>[a[1],f(a[0],a[1],u)]})),
    ...periodic.map(([id,rhs])=>({id,kind:'periodic',rhs}))]
    .map((world,i)=>{const rate=.85+.3*random();return {...world,index:i,rate,rhs:(a,u,h)=>{const y=world.rhs(a,u,h);return world.kind==='oscillator'?[y[0],rate*y[1]]:scale(y,rate);}};});
}

export function outsideStateOperatorWorlds() {
  return [
    {id:'outside_delay',kind:'oscillator',delay:.35,
      history:(plan,t)=>[plan.initial[0]+.7*Math.sin(4*t),plan.initial[1]],
      rhs:(a,u,_h,t,past)=>[a[1],-.5*a[0]-.12*a[1]+.7*u-.8*past(t-.35)[0]]},
    {id:'outside_projected_lorenz',kind:'oscillator',
      initialize:plan=>[plan.initial[0],plan.initial[0]+plan.initial[1]/10,24+2*plan.initial[0]],
      project:a=>[a[0],10*(a[1]-a[0])],
      rhs:(a,u)=>[10*(a[1]-a[0]),a[0]*(28-a[2])-a[1]+.1*u,a[0]*a[1]-(8/3)*a[2]]},
    {id:'outside_nonlocal',kind:'periodic',
      initialize:plan=>plan.initial.map((v,j)=>v+.15*Math.sin(6*Math.PI*j/plan.initial.length)),
      rhs:(a,u,h)=>a.map((v,j)=>.08*(a[(j+1)%a.length]-2*v+a[(j+a.length-1)%a.length])/(h*h)+.3*(a[(j+a.length/2)%a.length]-v)+.2*u[j])},
    {id:'outside_dispersion',kind:'periodic',
      initialize:plan=>plan.initial.map((v,j)=>v+.15*Math.sin(6*Math.PI*j/plan.initial.length)),
      rhs:(a,u,h)=>combine(scale(dx(dxx(a,h),h),-.06),scale(dx(mul(a,a),h),-.3),scale(u,.2))}
  ];
}

export function stateOperatorPlan(kind,label='train0',query=0) {
  const role=label==='selection'?2:label==='test'?3:label==='query'?4:label==='train1'?1:0;
  const amplitudes=[.2,.35,.8,1.1,[.25,.8,1.4][query]],phases=[.1,1.3,.7,2.1,[.2,1.2,2.4][query]];
  const initial=kind==='oscillator'?[[.15,.05],[-.4,.1],[.8,-.4],[-1.1,.5],[1.2,.3]][role]:
    Array.from({length:32},(_,j)=>.65+(.25+.06*role)*Math.sin(2*Math.PI*j/32+phases[role])+.12*Math.cos(4*Math.PI*j/32-.4*role));
  return {id:label+(label==='query'?query:''),initial,step:kind==='oscillator'?.02:.01,steps:400,
    ...(kind==='periodic'?{spacing:.25}:{}),forcing:{offset:0,amplitude:amplitudes[role],startFrequency:label==='query'?[.15,.3,.5][query]:.3+.07*role,endFrequency:label==='query'?[.6,1.1,1.8][query]:.9+.11*role,...(kind==='periodic'?{spatialMode:1}:{})}};
}

export function observeStateOperatorWorld(world,plan,snrDb,seed=27181,{substeps=24}={}) {
  if(!Number.isFinite(snrDb)||snrDb<0||!Number.isInteger(substeps)||substeps<1)throw new RangeError('Invalid observation protocol');
  const {initial,step,steps,spacing}=plan,states=[],commands=[],h=step/substeps,total=step*steps;
  function input(t) {
    const f=plan.forcing,phase=2*Math.PI*(f.startFrequency*t+.5*(f.endFrequency-f.startFrequency)*t*t/total);
    const u=f.offset+f.amplitude*Math.sin(phase);
    return world.kind==='oscillator'?u:initial.map((_,j)=>u*Math.cos(2*Math.PI*j/initial.length));
  }
  let x=world.initialize?world.initialize(plan):initial.slice();
  const history=world.delay?[x.slice()]:null;
  const past=time=>{
    if(time<=0)return world.history(plan,time);
    const coordinate=time/h,j=Math.floor(coordinate),f=coordinate-j;
    if(j>=history.length-1)return history[history.length-1];
    return history[j].map((v,k)=>v+f*(history[j+1][k]-v));
  };
  for(let i=0;i<=steps;i++) {
    const t=i*step;states.push(world.project?world.project(x):x.slice());commands.push(input(t));if(i===steps)break;
    for(let k=0;k<substeps;k++) {
      const at=t+k*h,a=world.rhs(x,input(at),spacing,at,past);
      const middle=x.map((v,j)=>v+h*a[j]/2),b=world.rhs(middle,input(at+h/2),spacing,at+h/2,past);
      x=x.map((v,j)=>v+h*b[j]);
      if(x.some(v=>!Number.isFinite(v)||Math.abs(v)>1e5))throw new RangeError('Reference trajectory diverged: '+world.id);
      if(history)history.push(x.slice());
    }
  }
  const random=rng(seed),sigma=initial.map((_,j)=>Math.sqrt(states.reduce((s,a)=>s+a[j]**2,0)/states.length)*10**(-snrDb/20));
  const samples=states.map((a,i)=>({time:i*step,state:a.map((v,j)=>v+sigma[j]*normal(random)),command:commands[i]}));
  return {observation:{schema:'worldlaw.state-trajectory/v1',id:world.id+':'+plan.id+':'+seed,kind:world.kind,...(world.kind==='periodic'?{spacing}:{}),samples},
    reference:{states,noiseSigma:sigma,snrDb,plan,world:world.id,integratorSubsteps:substeps}};
}

if(process.argv[1]&&pathToFileURL(process.argv[1]).href===import.meta.url) {
  const worlds=stateOperatorWorlds(process.argv[2]||'development');
  const output=worlds.map((w,i)=>observeStateOperatorWorld(w,stateOperatorPlan(w.kind,'train0'),25,27181+i));
  process.stdout.write(JSON.stringify({protocol:STATE_OPERATOR_PROTOCOL,cases:output})+'\n');
}
