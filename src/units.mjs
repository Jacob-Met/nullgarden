// Exchangeability units: one scientific independent subject may contribute
// multiple technical/longitudinal measurements. Never split those rows in
// CV or shuffle their labels independently. A unit cannot straddle batches
// or carry multiple labels in this one-level blocked design.
export function unitGroups(ds) {
  const { y, batch } = ds, unit = ds.unit;
  if (!Array.isArray(y) || !Array.isArray(batch) || y.length !== batch.length
      || (unit !== undefined && (!Array.isArray(unit) || unit.length !== y.length))) {
    throw new TypeError('Unit grouping requires aligned sample, batch, label, and optional unit arrays');
  }
  const map = new Map();
  y.forEach((label, i) => {
    if (label !== 0 && label !== 1) throw new TypeError('Unit labels must be binary 0 or 1');
    const id = unit ? unit[i] : i;
    if (unit && (typeof id !== 'string' || !id.trim())) {
      throw new TypeError('unit_id must be a nonempty string for every row');
    }
    const group = map.get(id);
    if (group) {
      if (group.label !== label) {
        const e = new Error('unit_id ' + id + ' has inconsistent labels; one independent unit must retain one binary label');
        e.code = 'UNIT_LABEL_INCONSISTENT';
        throw e;
      }
      if (group.batch !== batch[i]) {
        const e = new Error('unit_id ' + id + ' crosses batches; this one-level exchangeability model requires each unit in one batch');
        e.code = 'UNIT_CROSSES_BATCH';
        throw e;
      }
      group.indices.push(i);
    } else {
      map.set(id, {id, batch:batch[i], label, indices:[i]});
    }
  });
  return [...map.values()];
}

function shuffleUnits(rows,r) {
  const copy=rows.slice();
  for(let j=copy.length-1;j>0;j--) {
    const i=Math.floor(r()*(j+1));
    [copy[i],copy[j]]=[copy[j],copy[i]];
  }
  return copy;
}

export function stratifiedUnitFolds(y, unit, folds, random) {
  if (!Number.isInteger(folds) || folds < 2 || typeof random !== 'function') {
    throw new RangeError('Unit folds need k>=2 and a deterministic RNG callback');
  }
  const groups=unitGroups({y,batch:y.map(()=> 'same'),unit});
  const out=Array.from({length:folds},()=>[]);
  for (const cls of [0,1]) {
    const selected=shuffleUnits(groups.filter(g=>g.label===cls),random);
    selected.forEach((g,j)=>out[j%folds].push(...g.indices));
  }
  return out;
}

export function unitPermutationDesign(ds) {
  const groups=unitGroups(ds);
  return {
    groups,
    batch:groups.map(g=>g.batch),
    y:groups.map(g=>g.label),
    expand:(labels)=>{
      if(!Array.isArray(labels)||labels.length!==groups.length)
        throw new RangeError('Independent-unit label assignment length must match unit count');
      const out=Array(ds.y.length);
      groups.forEach((g,j)=>{
        if(labels[j]!==0 && labels[j]!==1)throw new TypeError('Independent-unit permutation labels must be binary');
        for(const i of g.indices)out[i]=labels[j];
      });
      return out;
    },
  };
}

/** Training-only independent-unit feature representation.
 * Each independent unit contributes exactly one feature vector regardless
 * of how many technical replicates it carries. Per-coordinate max scaling
 * prevents intermediate sums from overflowing on finite extreme inputs.
 */
export function independentFeatureRows(X, y, unit, batch, indices) {
  if (!unit) return indices.map(i => ({
    id:i, label:y[i], batch:batch?.[i], x:X[i], indices:[i],
  }));
  const map = new Map();
  for(const i of indices){
    const key=unit[i];
    let group=map.get(key);
    if(!group){
      group={id:key,label:y[i],batch:batch?.[i],indices:[],x:null};
      map.set(key,group);
    } else if(group.label!==y[i] || group.batch!==batch?.[i]){
      throw new Error('One independent unit has inconsistent labels or batches in training features: '+key);
    }
    group.indices.push(i);
  }
  const p=X[indices[0]]?.length;
  if (!Number.isInteger(p) || p < 1) throw new TypeError('Independent-unit feature matrix is empty');
  for (const group of map.values()) {
    group.x=new Array(p);
    for(let j=0;j<p;j++){
      let scale=0;
      for(const i of group.indices) {
        const v=X[i][j];
        if(!Number.isFinite(v)) throw new RangeError('Independent-unit feature must be finite at '+group.id+' feature '+j);
        scale=Math.max(scale,Math.abs(v));
      }
      if(scale===0){group.x[j]=0;continue;}
      let sum=0;
      for(const i of group.indices) sum+=X[i][j]/scale;
      const value=(sum/group.indices.length)*scale;
      if(!Number.isFinite(value))throw new RangeError('Independent-unit feature mean unrepresentable at '+group.id+' feature '+j);
      group.x[j]=value;
    }
  }
  return [...map.values()];
}
