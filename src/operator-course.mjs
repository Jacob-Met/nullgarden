import {pathToFileURL} from 'node:url';

// Independent fixed synthetic course. No fitting code or Worldlaw imports.
const settings=Object.freeze({trainingTolerance:1e-9,heldDerivativeTolerance:1e-9,heldStateTolerance:2e-6,
  resolutionTolerance:2e-6,permutationOffset:7});
function course(id,initial,step,steps,amplitude,startFrequency,endFrequency,periodic=false){
  return {id,initial,step,steps,...(periodic?{spacing:0.5}:{}),
    forcing:{offset:0,amplitude,startFrequency,endFrequency,...(periodic?{spatialMode:1}:{})}};
}
function input(c,t,j){
  const f=c.forcing,total=c.step*c.steps;
  const phase=2*Math.PI*(f.startFrequency*t+0.5*(f.endFrequency-f.startFrequency)*t*t/total);
  return f.offset+f.amplitude*Math.sin(phase)*(j===undefined?1:Math.cos(2*Math.PI*j/c.initial.length));
}
function physical(kind,c,t,state){
  if(kind==='oscillator'){
    const [position,rate]=state,command=input(c,t),change=command*Math.cos(position);
    return {rhs:[rate,change],rows:[{position,rate,command,change}]};
  }
  const rows=state.map((field,j)=>{
    const left=state[(j+state.length-1)%state.length],right=state[(j+1)%state.length],h=c.spacing;
    const change=-field*(right-left)/(2*h)+0.2*(right-2*field+left)/(h*h);
    return {field,left,right,spacing:h,command:input(c,t,j),change};
  });
  return {rhs:rows.map(r=>r.change),rows};
}
function observe(kind,spec) {
  const sub=32,h=spec.step/sub,states=[],blocks=[];
  let x=[...spec.initial];
  for(let i=0;i<=spec.steps;i++) {
    const t=i*spec.step;
    states.push([...x]);blocks.push(physical(kind,spec,t,x).rows);
    if(i===spec.steps)break;
    for(let j=0;j<sub;j++){
      const at=t+j*h,k=physical(kind,spec,at,x).rhs;
      const middle=x.map((v,n)=>v+0.5*h*k[n]);
      const slope=physical(kind,spec,at+h/2,middle).rhs;
      x=x.map((v,n)=>v+h*slope[n]);
    }
  }
  const shuffled=blocks.flatMap((rows,t)=>rows.map((r,j)=>({...r,change:blocks[(t+settings.permutationOffset)%blocks.length][j].change})));
  return {spec,states,rows:blocks.flat(),shuffled};
}
function field(n,phase,scale){
  return Array.from({length:n},(_,j)=>scale*(Math.sin(2*Math.PI*j/n+phase)+0.35*Math.cos(4*Math.PI*j/n-2*phase)));
}
function controls(){
  const delay=[-1,1,-0.5,0.5].map(previous=>({position:0,rate:0,command:0,change:previous}));
  // x'' = sigma * (x*(rho-z) - y - x') in the original Lorenz system.
  const lorenz=[25,29,27,31].map(z=>({position:1,rate:0,command:0,change:10*(1*(28-z)-1-0)}));
  // Centered third difference for [u_-2,0,0,0,0] with unit spacing.
  const third=[-2,2,-1,1].map(farLeft=>({field:0,left:0,right:0,spacing:1,command:0,change:-farLeft/2}));
  return [{id:'hidden_delay',kind:'oscillator',rows:delay},{id:'hidden_lorenz_state',kind:'oscillator',rows:lorenz},
    {id:'unseen_second_neighbour',kind:'periodic',rows:third}];
}
export function buildOperatorCourse(){
  const oscillator=[course('osc-train-low',[-2.4,0.3],0.015,40,0.8,0.8,2),
    course('osc-train-middle',[0.5,-0.6],0.015,40,1.5,1.2,3),course('osc-train-high',[2.2,0.7],0.015,40,2.2,0.5,2.5)];
  const periodic=[course('field-train-first',field(12,0.2,0.75),0.01,20,0.3,1,2,true),
    course('field-train-second',field(12,1.1,1.1),0.01,20,0.2,0.5,1.6,true)];
  return {schema:'nullgarden.operator-course/v1',settings:{...settings},cases:[
    {kind:'oscillator',train:oscillator.map(s=>observe('oscillator',s)),
     held:observe('oscillator',course('osc-held',[1.45,-1.2],0.0125,64,2.7,0.3,3.7))},
    {kind:'periodic',train:periodic.map(s=>observe('periodic',s)),
     held:observe('periodic',course('field-held',field(15,0.7,0.9),0.009,28,0.6,0.7,2.2,true))}
  ],outside:controls()};
}
if(process.argv[1]&&pathToFileURL(process.argv[1]).href===import.meta.url)
  process.stdout.write(JSON.stringify(buildOperatorCourse())+'\n');
