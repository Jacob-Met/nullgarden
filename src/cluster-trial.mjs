import {conditionalClusters, conditionalOrbitSize, conditionalLabelAssignments,
  sampleConditionalLabels} from './exact-orbit.mjs';
import {rng} from './core.mjs';
import {EXACT_TRIAL_CAP,planConstantEffect,runConstantEffectConfidenceSet,
  confidenceSetNotation} from './cluster-confidence.mjs';

function numericFailure(where){
  const error=new RangeError('Trial outcome numerical range unsupported in '+where
    +'; rescale measurements or choose a representable outcome contrast');
  error.code='TRIAL_NUMERIC_RANGE_UNSUPPORTED';throw error;
}
function stableMean(values,where){
  if(!values.length)throw new Error('Empty experimental unit at '+where);
  let scale=0;
  for(const value of values){
    if(!Number.isFinite(value))numericFailure(where);
    scale=Math.max(scale,Math.abs(value));
  }
  if(scale===0)return 0;
  let normalized=0;
  for(const value of values)normalized+=value/scale;
  const result=(normalized/values.length)*scale;
  if(!Number.isFinite(result))numericFailure(where);
  return result;
}

// Design-based Fisher randomization: outcomes are held fixed under the SHARP
// null; treatment is exchanged between whole randomized clusters, never rows.
// The statistic is an equal-stratum mean of differences in equal-cluster mean
// outcomes. Within each cluster, subjects are equally weighted, regardless of
// the number of technical repetitions per subject.
export function planClusterTrialOutcome(ds, proposal) {
  if (!ds?.randomization || ds.randomization.unit!=='cluster'
      || ds.randomization.mechanism!=='uniform_within_batch') {
    throw new Error('A causal trial outcome requires preregistered uniform whole-cluster treatment assignment');
  }
  if (!proposal || typeof proposal!=='object' || Array.isArray(proposal)
      || !['feature,perms,seed','confidenceSet,feature,perms,seed'].includes(Object.keys(proposal).sort().join(','))) {
    throw new Error('Preregister the exact primary trialOutcome {feature,seed,perms} with optional confidenceSet at dataset intake');
  }
  if(typeof proposal.feature!=='string'||!proposal.feature.trim()
      ||ds.features.filter(x=>x===proposal.feature).length!==1){
    throw new Error('The primary outcome must name exactly one existing numeric feature');
  }
  if(!Number.isSafeInteger(proposal.seed)||proposal.seed<0||proposal.seed>4294967295){
    throw new RangeError('Trial outcome seed must be an unsigned 32-bit preregistered integer');
  }
  if(!Number.isSafeInteger(proposal.perms)||proposal.perms<1||proposal.perms>1000){
    throw new RangeError('Trial outcome Monte Carlo budget must be a preregistered integer 1..1000');
  }
  const groups=conditionalClusters(ds.batch,ds.y,ds.subject,ds.cluster);
  const byBatch=new Map();
  for(const group of groups){
    if(group.units.some(u=>u.label!==group.units[0].label)){
      const e=new Error('Trial treatment must be identical within each randomized cluster '+group.id);
      e.code='CLUSTER_LABEL_MIXED';throw e;
    }
    const count=byBatch.get(group.batch)||{treated:0,control:0};
    if(group.units[0].label)count.treated++;else count.control++;
    byBatch.set(group.batch,count);
  }
  if(!byBatch.size)throw new Error('Trial outcome requires at least one randomized stratum');
  const confidenceSet=Object.hasOwn(proposal,'confidenceSet')?planConstantEffect(proposal.confidenceSet):undefined;
  return Object.freeze({...proposal,...(confidenceSet?{confidenceSet}:{})});
}

function prepare(ds,plan){
  const featureIndex=ds.features.indexOf(plan.feature);
  if(featureIndex<0)throw new Error('Registered outcome feature is no longer available');
  const clusterGroups=conditionalClusters(ds.batch,ds.y,ds.subject,ds.cluster);
  const blocks=new Map(),strata=new Map();
  for(const group of clusterGroups){
    const meanSubject=group.units.map(unit=>stableMean(unit.indices.map(i=>
      ds.X[i][featureIndex]),'subject '+unit.id));
    const outcome=stableMean(meanSubject,'cluster '+group.id);
    const block={id:group.id,batch:group.batch,label:group.units[0].label,
      indices:group.units.flatMap(u=>u.indices),subjects:group.units.length,outcome};
    if(group.units.some(u=>u.label!==block.label)){
      const e=new Error('Mixed trial treatment labels inside randomized cluster '+group.id);
      e.code='CLUSTER_LABEL_MIXED';throw e;
    }
    blocks.set(group.id,block);
    if(!strata.has(group.batch))strata.set(group.batch,[]);
    strata.get(group.batch).push(block);
  }
  const summary=[...strata].map(([batch,units])=>({
    batch,clusters:units.length,treated:units.filter(u=>u.label===1).length,
    control:units.filter(u=>u.label===0).length,subjects:units.reduce((s,u)=>s+u.subjects,0)
  }));
  const unsupported=summary.filter(s=>s.treated===0||s.control===0);
  const statistic=(labels)=>{
    const contrasts=[];
    for(const [stratum,units] of strata){
      const yes=[],no=[];
      for(const unit of units){
        const label=labels[unit.indices[0]];
        if(label!==0&&label!==1)throw new TypeError('Randomization labels must be binary');
        if(unit.indices.some(i=>labels[i]!==label))throw new Error('Randomization broke a complete cluster block');
        (label===1?yes:no).push(unit.outcome);
      }
      if(!yes.length||!no.length)throw new Error('Stratum has no randomized treatment contrast');
      const difference=stableMean(yes,'treated stratum '+stratum)-stableMean(no,'control stratum '+stratum);
      if(!Number.isFinite(difference))numericFailure('stratum treatment contrast '+stratum);
      contrasts.push(difference);
    }
    return stableMean(contrasts,'equal-stratum treatment contrast');
  };
  return {statistic,summary,unsupported,nClusters:clusterGroups.length,
    nSubjects:clusterGroups.reduce((n,g)=>n+g.units.length,0)};
}

function exceeds(candidate,observed){
  // Count ties as exceedances. Conservative floating guard protects equal
  // absolute contrasts from differently ordered sum accumulation.
  const scale=Math.max(1,Math.abs(candidate),Math.abs(observed));
  return Math.abs(candidate)+1e-12*scale>=Math.abs(observed);
}

export function runClusterTrialOutcome(ds,plan=ds.trialOutcome){
  const frozen=planClusterTrialOutcome(ds,plan);
  if(frozen.confidenceSet){
    const registered=ds.trialOutcome;
    if(!registered?.confidenceSet||registered.feature!==frozen.feature
        ||registered.seed!==frozen.seed||registered.perms!==frozen.perms
        ||registered.confidenceSet.model!==frozen.confidenceSet.model
        ||registered.confidenceSet.alpha!==frozen.confidenceSet.alpha)
      throw new Error('The constant-effect model and alpha must be registered in the dataset trialOutcome before analysis; no post-hoc confidence-set upgrade');
  }
  const prepared=prepare(ds,frozen), {summary,unsupported}=prepared;
  const confidenceSet=frozen.confidenceSet?runConstantEffectConfidenceSet(ds,frozen):
    {status:'not_registered',exact:false,finding:'No constant-additive effect model was registered at dataset intake; no effect confidence set is reported.'};
  const common={method:'Fisher cluster-randomization test of sharp no-effect null',
    design:'predeclared uniform whole-cluster assignment conditioned on each stratum treatment count',
    primaryOutcome:frozen.feature,seed:frozen.seed,permutationBudget:frozen.perms,
    protocolId:ds.randomization.protocolId,nClusters:prepared.nClusters,
    nSubjects:prepared.nSubjects,nStrata:summary.length,strata:summary,
    estimator:'mean of equal-cluster treatment-control differences within each stratum; equal stratum weights',
    unit:'cluster-level treatment allocation; equal subject contribution inside each cluster',
    confidenceSet,
    causalLimit:'Tests a SHARP null that treatment changes no measured outcome for any subject under registered uniform cluster assignment and no interference. It does not independently verify that allocation protocol was followed; observational treatment, unequal assignment probabilities, unregistered restrictions, carryover and between-cluster interference are unsupported. The point contrast alone is not a confidence interval. Any separately registered constant-effect confidence set relies on a SAME-effect model for every unit and does not give a confidence interval for a heterogeneous average treatment effect.'};
  if(unsupported.length){
    return {...common,status:'unavailable',observedContrast:null,p:null,pExact:false,
      nullSampling:'unidentifiable_stratum',
      finding:'No treatment contrast is identified in strata '+unsupported.map(s=>s.batch).join(', ')
        +'. Refusing to silently drop selected strata or infer a causal effect.'};
  }
  const observed=prepared.statistic(ds.y);
  const exactCap=EXACT_TRIAL_CAP;
  const orbit=conditionalOrbitSize(ds.batch,ds.y,exactCap,ds.subject,ds.cluster,'cluster');
  if(orbit===1){
    return {...common,status:'unavailable',observedContrast:observed,p:null,pExact:false,
      orbitSize:1,nullSampling:'no_cluster_randomization_contrast',
      finding:'Only the observed cluster allocation is legal after conditioning on preregistered strata.'};
  }
  let exceedances=0,draws=0;
  if(orbit<=exactCap){
    if(confidenceSet.status==='available'){
      const p=confidenceSet.pAtZero;
      return {...common,status:p<=0.05?'reject_sharp_null':'retain_sharp_null',
        observedContrast:confidenceSet.observedContrast,
        absoluteStatistic:Math.abs(confidenceSet.observedContrast),p,
        pExact:true,orbitSize:orbit,nullSampling:'complete_cluster_randomization_orbit',
        assignmentsEvaluated:orbit,exceedances:confidenceSet.zeroExceedances,
        finding:'Exact two-sided Fisher randomization p='+p.toFixed(5)
          +' across all '+orbit+' legal whole-cluster treatment assignments (exact rational ties included).'};
    }
    for(const labels of conditionalLabelAssignments(ds.batch,ds.y,
      {maxAssignments:exactCap,subject:ds.subject,cluster:ds.cluster,assignmentUnit:'cluster'})){
      const value=prepared.statistic(labels);
      if(exceeds(value,observed))exceedances++;
      draws++;
    }
    if(draws!==orbit||exceedances<1)throw new Error('Fisher sharp-null complete orbit invariant failed');
    const p=exceedances/orbit;
    return {...common,status:p<=0.05?'reject_sharp_null':'retain_sharp_null',
      observedContrast:observed,absoluteStatistic:Math.abs(observed),p,
      pExact:true,orbitSize:orbit,nullSampling:'complete_cluster_randomization_orbit',
      assignmentsEvaluated:draws,exceedances,
      finding:'Exact two-sided Fisher randomization p='+p.toFixed(5)
        +' across all '+orbit+' legal whole-cluster treatment assignments (ties included).'};
  }
  const random=rng(frozen.seed),B=frozen.perms;
  for(let i=0;i<B;i++){
    const labels=sampleConditionalLabels(ds.batch,ds.y,random,
      ds.subject,ds.cluster,'cluster');
    const value=prepared.statistic(labels);
    if(exceeds(value,observed))exceedances++;
    draws++;
  }
  const p=(1+exceedances)/(B+1);
  return {...common,status:p<=0.05?'reject_sharp_null':'retain_sharp_null',
    observedContrast:observed,absoluteStatistic:Math.abs(observed),p,pExact:false,
    orbitSizeLowerBound:exactCap+1,orbitSizeExact:false,
    nullSampling:'predeclared_monte_carlo_plus_one',assignmentsEvaluated:draws,
    sampledExceedances:exceedances,
    finding:'Two-sided '+B+'-draw Monte Carlo Fisher p estimate with +1 correction='
      +p.toFixed(5)+'; not a complete conditional-orbit enumeration.'};
}

export function trialOutcomeMarkdown(run){
  const fmt=v=>v===null?'unavailable':Number(v).toPrecision(6);
  const lines=['## Preregistered cluster-trial primary outcome','',
    'Protocol: '+run.protocolId+' (asserted; historical randomization NOT independently verified).',
    'Primary outcome: '+run.primaryOutcome,
    'Allocation: '+run.design,
    'Unit weighting: '+run.unit,
    'Point contrast: '+fmt(run.observedContrast),
    'Randomization p: '+fmt(run.p)+' ('+run.nullSampling+')',
    'Independent clusters: '+run.nClusters+'; strata: '+run.nStrata,
    'Outcome: '+run.status+'. '+run.finding,'',
    'Interpretation limit: '+run.causalLimit,
    ''];
  const confidence=run.confidenceSet;
  if(confidence){
    lines.push('### Constant-additive effect confidence set','',
      'Status: '+confidence.status+'. '+confidence.finding);
    if(confidence.status==='available')lines.push(
      'Confidence level: '+((1-confidence.alpha)*100).toPrecision(6)+'%; alpha: '+confidence.alpha,
      'Exact set: '+confidenceSetNotation(confidence),
      'Endpoint convention: square brackets include a boundary; braces are an isolated value; parentheses exclude a boundary or infinity.',
      'Assumption: '+confidence.assumption,
      'Interpretation: '+confidence.interpretation,
      'Acceptance rule: '+confidence.rule,
      'Arithmetic: '+confidence.arithmetic,
      'Certificate: '+confidence.cells.length+' exact open-interval / boundary cells; '+confidence.assignmentsEvaluated+' allocations.');
    else if(confidence.assumption)lines.push('Assumption: '+confidence.assumption,'Interpretation: '+confidence.interpretation);
    lines.push('');
  }
  return lines.join('\n');
}
