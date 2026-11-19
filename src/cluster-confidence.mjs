import {conditionalClusters, conditionalOrbitSize, conditionalLabelAssignments} from './exact-orbit.mjs';

export const EXACT_TRIAL_CAP = 512;
const abs = n => n < 0n ? -n : n;
function gcd(a,b) { a=abs(a);b=abs(b);while(b){[a,b]=[b,a%b];}return a; }
function q(n,d=1n) {
  if(!d)throw new RangeError('Undefined exact confidence-set arithmetic');
  if(d<0n){n=-n;d=-d;}
  const g=gcd(n,d);return {n:n/g,d:d/g};
}
const ZERO=q(0n), ONE=q(1n);
const add=(a,b)=>q(a.n*b.d+b.n*a.d,a.d*b.d);
const neg=a=>({n:-a.n,d:a.d});
const sub=(a,b)=>add(a,neg(b));
const div=(a,b)=>q(a.n*b.d,a.d*b.n);
const cmp=(a,b)=>{const n=a.n*b.d-b.n*a.d;return n<0n?-1:n>0n?1:0;};
const key=a=>a.n+'/'+a.d;
const mean=values=>div(values.reduce(add,ZERO),q(BigInt(values.length)));

// Preserve the actual finite binary64 measurement, including subnormals.
// Root ordering and ties never depend on a floating epsilon or a search grid.
function measured(value) {
  if(!Number.isFinite(value))throw Object.assign(new RangeError('Nonfinite trial outcome'),
    {code:'TRIAL_NUMERIC_RANGE_UNSUPPORTED'});
  if(value===0)return ZERO;
  const bytes=new DataView(new ArrayBuffer(8));bytes.setFloat64(0,value);
  const bits=bytes.getBigUint64(0), exponent=Number((bits>>52n)&2047n);
  let n=(bits&((1n<<52n)-1n))+(exponent?1n<<52n:0n);
  if(bits>>63n)n=-n;
  const power=exponent?exponent-1075:-1074;
  return power>=0?q(n<<BigInt(power)):q(n,1n<<BigInt(-power));
}

// Alpha is the declared decimal probability (0.3 means 3/10), not the
// slightly different binary64 value that would misclassify a p=3/10 tie.
function level(value) {
  if(typeof value!=='number'||!Number.isFinite(value)||value<=0||value>=1)
    throw new RangeError('Confidence-set alpha must be a preregistered number strictly between 0 and 1');
  const [coefficient,exp='0']=String(value).toLowerCase().split('e');
  const digits=coefficient.replace('.',''), decimals=(coefficient.split('.')[1]||'').length;
  const power=Number(exp)-decimals;
  return power>=0?q(BigInt(digits)*10n**BigInt(power)):q(BigInt(digits),10n**BigInt(-power));
}

export function planConstantEffect(proposal) {
  if(!proposal||typeof proposal!=='object'||Array.isArray(proposal)
      ||Object.keys(proposal).sort().join(',')!=='alpha,model'
      ||proposal.model!=='constant_additive') {
    throw new Error('Preregister confidenceSet {model:"constant_additive",alpha:0.05} at dataset intake; this assumes the SAME additive effect for every measured unit');
  }
  level(proposal.alpha);
  return Object.freeze({...proposal});
}

function approximate(a) {
  if(a.n===0n)return 0;
  const n=abs(a.n), ns=Math.max(0,n.toString(2).length-54), ds=Math.max(0,a.d.toString(2).length-54);
  const value=Number(n>>BigInt(ns))/Number(a.d>>BigInt(ds))*2**(ns-ds)*(a.n<0n?-1:1);
  return Number.isFinite(value)&&value!==0?value:null;
}
function point(a) {
  return {numerator:String(a.n),denominator:String(a.d),approximate:approximate(a)};
}
const sign=n=>n<0n?-1:n>0n?1:0;
function linear(intercept,slope) {
  return {c:intercept.n*slope.d,s:slope.n*intercept.d};
}

// Invert |a-b*tau| >= |observed-tau| for every legal allocation.
// The two signed differences are affine. Their exact rational roots partition
// the real line into constant-p open cells and separately evaluated tie points.
// A sweep counts both sides and every root in O(M log M), not O(M^2).
function invert(lines,observed,alpha) {
  const threshold=level(alpha),events=new Map();let count=0,zeroCount=0;
  for(const {a,b} of lines) {
    const factors=[linear(sub(a,observed),sub(ONE,b)),linear(add(a,observed),neg(add(b,ONE)))];
    const [u,v]=factors;
    if(sign(u.c)*sign(v.c)>=0)zeroCount++;
    if(factors.some(f=>f.c===0n&&f.s===0n)){count++;continue;}
    let left=sign(u.s?-u.s:u.c)*sign(v.s?-v.s:v.c)>=0;
    if(left)count++;
    const own=new Map();
    for(const f of factors)if(f.s){
      const root=q(-f.c,f.s),k=key(root),entry=own.get(k)||{root,multiplicity:0};
      entry.multiplicity++;own.set(k,entry);
    }
    for(const {root,multiplicity} of [...own.values()].sort((a,b)=>cmp(a.root,b.root))) {
      const right=multiplicity%2?!left:left,k=key(root);
      const event=events.get(k)||{root,at:0,after:0};
      // Equality is INCLUDED, even when neither neighboring cell is included.
      event.at+=left?0:1;event.after+=right?0:-1;events.set(k,event);left=right;
    }
  }
  const accepted=n=>BigInt(n)*threshold.d>threshold.n*BigInt(lines.length);
  const cells=[],components=[];
  const append=(kind,lower,upper,n)=>{
    if(n<1||n>lines.length)throw new Error('Complete-orbit confidence count invariant failed');
    const cell={kind,lower,upper,lowerClosed:kind==='point',upperClosed:kind==='point',
      exceedances:n,p:n/lines.length,accepted:accepted(n)};
    cells.push(cell);
  };
  let previous=null;
  for(const event of [...events.values()].sort((a,b)=>cmp(a.root,b.root))) {
    const root=point(event.root);
    append('interval',previous,root,count);
    count+=event.at;append('point',root,root,count);count+=event.after;previous=root;
  }
  append('interval',previous,null,count);
  let active=null;
  for(const cell of cells) {
    if(!cell.accepted){active=null;continue;}
    if(active){active.upper=cell.upper;active.upperClosed=cell.upperClosed;active.kind='interval';}
    else {active={kind:cell.kind,lower:cell.lower,upper:cell.upper,
      lowerClosed:cell.lowerClosed,upperClosed:cell.upperClosed};components.push(active);}
  }
  return {components,cells,breakpoints:events.size,assignmentsEvaluated:lines.length,
    zeroExceedances:zeroCount,pAtZero:zeroCount/lines.length};
}

// A mathematical entry point used to verify disconnected and singleton sets.
// The HTTP trial API never accepts arbitrary coefficients or alternative models.
export function invertAffineRandomization(lines,observed,alpha) {
  if(!Array.isArray(lines)||!lines.length||lines.length>EXACT_TRIAL_CAP)
    throw new RangeError('Affine orbit must contain 1..'+EXACT_TRIAL_CAP+' assignments');
  const o=measured(observed),exact=lines.map(({a,b})=>({a:measured(a),b:measured(b)}));
  if(!exact.some(({a,b})=>cmp(a,o)===0&&cmp(b,ONE)===0))
    throw new Error('The complete orbit must include its observed assignment');
  return invert(exact,o,alpha);
}

export function runConstantEffectConfidenceSet(ds,plan) {
  if(ds.randomization?.unit!=='cluster'||ds.randomization?.mechanism!=='uniform_within_batch')
    throw new Error('Constant-effect confidence sets require preregistered uniform whole-cluster randomization');
  const registered=planConstantEffect(plan.confidenceSet);
  const common={model:registered.model,alpha:registered.alpha,confidenceLevel:1-registered.alpha,
    exact:false,assumption:'Every measured unit has the SAME constant additive treatment effect, with no interference; the registered uniform whole-cluster allocation protocol was followed.',
    interpretation:'A randomization confidence set for a common constant effect under this strong model. It is NOT a confidence interval for an average treatment effect under heterogeneous effects.',
    rule:'Keep exactly tau with full-orbit two-sided p(tau) > alpha; ties in absolute statistics count as exceedances.',
    arithmetic:'Exact rational arithmetic on stored binary64 outcomes and decimal alpha; exact rational endpoints are authoritative.'};
  const groups=conditionalClusters(ds.batch,ds.y,ds.subject,ds.cluster),strata=new Map();
  const feature=ds.features.indexOf(plan.feature);
  for(const group of groups) {
    const label=group.units[0].label;
    if(group.units.some(u=>u.label!==label))throw new Error('Confidence set requires complete uniformly labeled clusters');
    const outcome=mean(group.units.map(u=>mean(u.indices.map(i=>measured(ds.X[i][feature])))));
    if(!strata.has(group.batch))strata.set(group.batch,[]);
    strata.get(group.batch).push({index:group.units[0].indices[0],label,outcome});
  }
  if([...strata.values()].some(units=>!units.some(u=>u.label===0)||!units.some(u=>u.label===1)))
    return {...common,status:'unavailable',reason:'unidentifiable_stratum',
      finding:'Every registered stratum needs both treatment arms; no strata were removed.'};
  const orbit=conditionalOrbitSize(ds.batch,ds.y,EXACT_TRIAL_CAP,ds.subject,ds.cluster,'cluster');
  if(orbit>EXACT_TRIAL_CAP)return {...common,status:'unavailable',reason:'exact_orbit_exceeds_budget',
    enumerationBudget:EXACT_TRIAL_CAP,orbitSizeLowerBound:EXACT_TRIAL_CAP+1,
    finding:'The complete allocation orbit exceeds '+EXACT_TRIAL_CAP+' assignments. No exact confidence set is reported; Monte Carlo no-effect p is a separate result.'};
  const coefficients=labels=>{
    const aa=[],bb=[];
    for(const units of strata.values()) {
      const yes=units.filter(u=>labels[u.index]===1),no=units.filter(u=>labels[u.index]===0);
      aa.push(sub(mean(yes.map(u=>u.outcome)),mean(no.map(u=>u.outcome))));
      bb.push(sub(mean(yes.map(u=>q(BigInt(u.label)))),mean(no.map(u=>q(BigInt(u.label))))));
    }
    return {a:mean(aa),b:mean(bb)};
  };
  const observed=coefficients(ds.y);
  if(cmp(observed.b,ONE)!==0)throw new Error('Observed constant-effect coefficient must equal one');
  const lines=[...conditionalLabelAssignments(ds.batch,ds.y,{maxAssignments:EXACT_TRIAL_CAP,
    subject:ds.subject,cluster:ds.cluster,assignmentUnit:'cluster'})].map(coefficients);
  if(lines.length!==orbit)throw new Error('Confidence-set orbit is incomplete');
  const result=invert(lines,observed.a,registered.alpha);
  const estimate=approximate(observed.a);
  if(estimate===null)throw Object.assign(new RangeError('Trial contrast is outside representable numeric range'),
    {code:'TRIAL_NUMERIC_RANGE_UNSUPPORTED'});
  return {...common,...result,status:'available',exact:true,orbitSize:orbit,
    observedContrast:estimate,observedContrastExact:point(observed.a),
    finding:'Complete '+orbit+'-allocation inversion with '+result.breakpoints+' exact breakpoints; '
      +result.components.length+' connected component'+(result.components.length===1?'':'s')+'.'};
}

export function confidenceSetNotation(result) {
  if(result.status!=='available')return result.finding;
  const exact=p=>p.denominator==='1'?p.numerator:p.numerator+'/'+p.denominator;
  return result.components.map(c=>c.kind==='point'?'{'+exact(c.lower)+'}':
    (c.lowerClosed?'[':'(')+(c.lower?exact(c.lower):'-infinity')+', '
      +(c.upper?exact(c.upper):'+infinity')+(c.upperClosed?']':')')).join(' union ')||'empty set';
}
