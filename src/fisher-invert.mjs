import {conditionalOrbitSize, conditionalLabelAssignments} from './exact-orbit.mjs';
import {trialClusterBlocks} from './cluster-trial.mjs';
import {R0,R1,rint,rat,rneg,rabs,radd,rsub,rmul,rdiv,rmean,rcmp,req,
  rkey,rfromNumber,rwire} from './exact-rational.mjs';

const LEVEL=0.95;
const EXACT_CAP=512;
const exactAccept=(count,total)=>BigInt(count)*20n>BigInt(total);

function comparison(line,observed,tau){
  const alt=rsub(line.a,rmul(line.b,tau));
  const obs=rsub(observed.a,tau);
  return rcmp(rabs(alt),rabs(obs))>=0;
}
export function rationalFisherP(lines,observed,tau){
  let exceedances=0;
  for(const line of lines)if(comparison(line,observed,tau))exceedances++;
  return {exceedances,total:lines.length,p:exceedances/lines.length};
}

// Any two-sided Fisher contrast |a_j - b_j*tau| >= |a_obs - tau|
// changes truth value only where one of the signed linear factors vanishes.
// Sweep those algebraic crossings instead of scanning a user-chosen grid:
// no skipped slivers, disconnected sets, isolated points, or arbitrary range.
export function invertFisherLines(lines,observedIndex,{confidenceLevel=LEVEL}={}){
  if(confidenceLevel!==LEVEL)throw new Error('Fisher inversion fixes 95% confidence at preregistration');
  if(!Array.isArray(lines)||lines.length<2||!lines[observedIndex])
    throw new Error('Exact Fisher inversion needs a complete legal allocation orbit');
  const observed=lines[observedIndex];
  if(!req(observed.b,R1))
    throw new Error('Observed trial treatment assignment must have coefficient exactly +1');
  const events=new Map();
  function register(root,index){
    const key=rkey(root);
    let event=events.get(key);
    if(!event){event={root,multiplicity:new Map()};events.set(key,event);}
    event.multiplicity.set(index,(event.multiplicity.get(index)||0)+1);
  }
  for(let i=0;i<lines.length;i++){
    const line=lines[i];
    // Signed difference factors:
    // (a-a0)+(1-b)*tau and (a+a0)-(b+1)*tau.
    const c1=rsub(line.a,observed.a),s1=rsub(R1,line.b);
    const c2=radd(line.a,observed.a),s2=rneg(radd(line.b,R1));
    if((req(c1,R0)&&req(s1,R0))||(req(c2,R0)&&req(s2,R0))){
      // Identically equal absolute scores for EVERY treatment effect.
      continue;
    }
    if(!req(s1,R0))register(rneg(rdiv(c1,s1)),i);
    if(!req(s2,R0))register(rneg(rdiv(c2,s2)),i);
  }
  const roots=[...events.values()].sort((a,b)=>rcmp(a.root,b.root));
  const left=roots.length?rsub(roots[0].root,R1):R0;
  const state=lines.map(line=>comparison(line,observed,left));
  let current=state.reduce((sum,value)=>sum+(value?1:0),0);
  const counts=[current];
  for(const event of roots){
    let at=current;
    for(const [i] of event.multiplicity)if(!state[i])at++;
    counts.push(at);
    for(const [i,mult] of event.multiplicity){
      if(mult%2){
        current+=state[i]?-1:1;
        state[i]=!state[i];
      }
    }
    counts.push(current);
  }
  if(counts.length!==2*roots.length+1)throw new Error('Fisher sweep partition length invariant');
  const retained=counts.map(count=>exactAccept(count,lines.length));
  const intervals=[];
  let start=-1;
  for(let i=0;i<=retained.length;i++){
    if(i<retained.length&&retained[i]){if(start<0)start=i;continue;}
    if(start<0)continue;
    const last=i-1,first=start;
    const lo=first%2?roots[(first-1)/2].root
      :first===0?null:roots[first/2-1].root;
    const hi=last%2?roots[(last-1)/2].root
      :last===2*roots.length?null:roots[last/2].root;
    intervals.push({lower:lo===null?null:rwire(lo),
      upper:hi===null?null:rwire(hi),
      includeLower:first%2===1,includeUpper:last%2===1});
    start=-1;
  }
  if(!intervals.length)throw new Error('Exact Fisher confidence set excludes observed contrast: impossible invariant');
  const atZero=rationalFisherP(lines,observed,R0);
  const atEstimate=rationalFisherP(lines,observed,observed.a);
  if(atEstimate.exceedances!==lines.length)
    throw new Error('Observed point contrast must always be retained with p=1');
  return {status:'exact',confidenceLevel:LEVEL,
    model:'constant_additive_effect_for_every_subject',
    method:'exact algebraic Fisher inversion, all signed crossings and rational sweep',
    orbitSize:lines.length,nBreakpoints:roots.length,nComponents:intervals.length,
    intervals,observedContrastExact:rwire(observed.a),pAtZero:atZero,
    pAtObservedContrast:atEstimate,
    arithmetic:'exact rational arithmetic on ingested binary64 cluster-mean outcomes; boundaries serialized as reduced fractions',
    scientificLimit:'95% randomization confidence SET only under a constant additive response effect on every subject, uniform registered cluster treatment, fixed numeric outcome measurement and no between-cluster interference. It is not a general average-treatment-effect interval and cannot establish the historical randomization mechanism.'};
}

function coefficientLines(ds, blocks){
  const prepared=blocks.blocksByStratum.map(({batch,units})=>({
    batch,units:units.map(unit=>({
      indices:unit.indices,outcome:rfromNumber(unit.outcome),
      originalTreatment:rint(unit.label)
    }))
  }));
  const makeLine=allocation=>{
    const outcomes=[],baselineTreatments=[];
    for(const {batch,units} of prepared){
      const treatment=[],control=[],treatmentOrig=[],controlOrig=[];
      for(const unit of units){
        const z=allocation[unit.indices[0]];
        if(unit.indices.some(i=>allocation[i]!==z))
          throw new Error('Inversion relabeling split an indivisible randomized cluster in '+batch);
        (z===1?treatment:control).push(unit.outcome);
        (z===1?treatmentOrig:controlOrig).push(unit.originalTreatment);
      }
      if(!treatment.length||!control.length)
        throw new Error('Inversion has unidentifiable treatment stratum '+batch);
      outcomes.push(rsub(rmean(treatment),rmean(control)));
      baselineTreatments.push(rsub(rmean(treatmentOrig),rmean(controlOrig)));
    }
    return {a:rmean(outcomes),b:rmean(baselineTreatments)};
  };
  const observed=makeLine(ds.y),lines=[];
  let observedIndex=-1;
  for(const assignment of conditionalLabelAssignments(ds.batch,ds.y,{
    maxAssignments:EXACT_CAP,subject:ds.subject,cluster:ds.cluster,assignmentUnit:'cluster'
  })){
    if(assignment.every((z,i)=>z===ds.y[i])){
      if(observedIndex!==-1)throw new Error('Observed cluster allocation appeared twice');
      observedIndex=lines.length;
    }
    lines.push(makeLine(assignment));
  }
  if(observedIndex<0||!req(lines[observedIndex].a,observed.a)
    ||!req(lines[observedIndex].b,observed.b))
    throw new Error('Observed assignment coefficient not independently recovered from complete orbit');
  return {lines,observedIndex};
}

export function runTrialConfidenceSet(ds){
  if(ds.effectModel!=='constant_additive')
    throw new Error('A constant-additive-effect confidence set must be explicitly opted in at dataset intake');
  if(!ds.trialOutcome||!ds.randomization||ds.randomization.unit!=='cluster'
    ||ds.randomization.mechanism!=='uniform_within_batch')
    throw new Error('Confidence inversion requires preregistered numeric outcome and uniform cluster assignment');
  const blocks=trialClusterBlocks(ds);
  const common={protocolId:ds.randomization.protocolId,primaryOutcome:ds.trialOutcome.feature,
    nClusters:blocks.nClusters,nSubjects:blocks.nSubjects,
    nStrata:blocks.summary.length,effectModel:ds.effectModel,
    confidenceLevel:LEVEL};
  if(blocks.unsupported.length){
    return {...common,status:'unavailable',intervals:null,pAtZero:null,
      reason:'Treatment and control are not present in every preregistered randomization stratum; no design-valid additive confidence inversion.'};
  }
  const n=conditionalOrbitSize(ds.batch,ds.y,EXACT_CAP,ds.subject,ds.cluster,'cluster');
  if(n>EXACT_CAP){
    return {...common,status:'unavailable',intervals:null,pAtZero:null,
      orbitSizeLowerBound:EXACT_CAP+1,
      reason:'Complete exact orbit exceeds '+EXACT_CAP+' allocations. No Monte Carlo estimate is silently relabeled an exact 95% confidence set.'};
  }
  const {lines,observedIndex}=coefficientLines(ds,blocks);
  if(lines.length!==n)throw new Error('Exact cluster-orbit enumeration mismatch');
  const result=invertFisherLines(lines,observedIndex);
  return {...common,...result};
}

export function trialConfidenceMarkdown(result){
  const lines=['## Preregistered constant-additive Fisher confidence set','',
    'Outcome: '+result.primaryOutcome+'; protocol: '+result.protocolId,
    'Model: all subjects share a constant additive effect (asserted; not empirically proven).',
    'Status: '+result.status,
    result.status==='exact'?'95% confidence SET across '+result.orbitSize+' full randomized assignments:'
      :'Confidence set unavailable: '+result.reason];
  if(result.intervals)for(const interval of result.intervals){
    const lo=interval.lower?interval.lower.numerator+'/'+interval.lower.denominator:'-infinity';
    const hi=interval.upper?interval.upper.numerator+'/'+interval.upper.denominator:'+infinity';
    lines.push((interval.includeLower?'[':'(')+lo+', '+hi+(interval.includeUpper?']':')'));
  }
  lines.push('Precision: '+(result.arithmetic||'not evaluated'));
  lines.push('Scientific boundary: '+(result.scientificLimit||'No supported confidence-set inversion.'));
  return lines.join('\n')+'\n';
}
