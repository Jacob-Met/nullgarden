// Conditional randomization operates on independent allocation units, not
// blindly on rows. With declared subject IDs, every repeated measurement of a
// subject is one indivisible label unit nested in one batch. Without IDs, each
// row is *assumed* to be its own unit; the assumption cannot be verified.
export function conditionalUnits(batch, labels, subject = null) {
  if (!Array.isArray(batch) || !Array.isArray(labels)
      || !batch.length || batch.length !== labels.length
      || (subject !== null && (!Array.isArray(subject) || subject.length !== labels.length))) {
    throw new TypeError('Conditional units require aligned nonempty batch, binary label and optional subject arrays');
  }
  const byKey = new Map();
  for (let i = 0; i < labels.length; i++) {
    if (labels[i] !== 0 && labels[i] !== 1) {
      throw new TypeError('Conditional units require binary labels');
    }
    const key = subject === null ? i : subject[i];
    if (subject !== null && (typeof key !== 'string' || !key.trim())) {
      throw new TypeError('Subject IDs must be nonempty strings');
    }
    let unit = byKey.get(key);
    if (!unit) {
      unit = {id: key, batch: batch[i], label: labels[i], indices: []};
      byKey.set(key, unit);
    } else {
      if (unit.batch !== batch[i]) {
        const error = new Error('Subject spans batches: ' + key + '. Cannot shuffle it independently inside batches.');
        error.code = 'SUBJECT_SPANS_BATCH';
        throw error;
      }
      if (unit.label !== labels[i]) {
        const error = new Error('Subject has conflicting labels: ' + key + '. A single binary subject-level allocation is not defined.');
        error.code = 'SUBJECT_LABEL_CONFLICT';
        throw error;
      }
    }
    unit.indices.push(i);
  }
  return [...byKey.values()];
}

// Nested whole-subject exchangeability: family/site clusters are uniquely
// nested in batches, and each subject is in exactly one cluster. A cluster
// is the out-of-sample CV unit. Subject labels exchange only WITHIN its
// declared cluster, never across potentially dependent families/sites.
export function conditionalClusters(batch,labels,subject,cluster) {
  if(!Array.isArray(subject)||!Array.isArray(cluster)
      ||subject.length!==labels.length||cluster.length!==labels.length) {
    throw new TypeError('Nested cluster_id requires aligned subject_id and cluster_id arrays');
  }
  const units=conditionalUnits(batch,labels,subject);
  const byCluster=new Map();
  for(const unit of units) {
    const id=cluster[unit.indices[0]];
    if(typeof id!=='string'||!id.trim())throw new TypeError('cluster_id must be a nonempty string');
    for(const row of unit.indices)if(cluster[row]!==id) {
      const e=new Error('Subject spans clusters: '+unit.id);
      e.code='SUBJECT_SPANS_CLUSTER';throw e;
    }
    if(!byCluster.has(id))byCluster.set(id,{id,batch:unit.batch,units:[]});
    const group=byCluster.get(id);
    if(group.batch!==unit.batch){
      const e=new Error('Cluster spans batches: '+id);
      e.code='CLUSTER_SPANS_BATCH';throw e;
    }
    group.units.push(unit);
  }
  return [...byCluster.values()];
}

function groupsOf(batch,labels,subject,cluster=null) {
  if(cluster!==null)return conditionalClusters(batch,labels,subject,cluster).map(g=>g.units);
  const byBatch=new Map();
  for(const unit of conditionalUnits(batch,labels,subject)){
    if(!byBatch.has(unit.batch))byBatch.set(unit.batch,[]);
    byBatch.get(unit.batch).push(unit);
  }
  return [...byBatch.values()];
}

// Orbit cardinality is the number of allocations of unit labels. A subject
// with twenty observations contributes ONE slot, not twenty pseudo-replicates.
export function conditionalOrbitSize(batch, labels, cap = 4096, subject = null, cluster = null) {
  if (!Number.isSafeInteger(cap) || cap < 1) {
    throw new RangeError('Conditional-orbit cap must be a positive safe integer');
  }
  let size = 1n;
  const limit = BigInt(cap);
  for (const group of groupsOf(batch, labels, subject, cluster)) {
    const n = group.length, positives = group.filter((u) => u.label === 1).length;
    const k = Math.min(positives, n - positives);
    let ways = 1n;
    for (let j = 1; j <= k; j++) {
      ways = ways * BigInt(n - k + j) / BigInt(j);
      if (ways > limit) return cap + 1;
    }
    size *= ways;
    if (size > limit) return cap + 1;
  }
  return Number(size);
}

function* choose(n, count, start = 0, selected = []) {
  if (count === 0) {
    yield selected.slice();
    return;
  }
  for (let i = start; i <= n - count; i++) {
    selected.push(i);
    yield* choose(n, count - 1, i + 1, selected);
    selected.pop();
  }
}

export function* conditionalLabelAssignments(batch, labels,
  {maxAssignments = 4096, subject = null, cluster = null} = {}) {
  const size = conditionalOrbitSize(batch, labels, maxAssignments, subject, cluster);
  if (size > maxAssignments) {
    const error = new RangeError('Conditional label orbit exceeds exact enumeration budget; choose bounded Monte Carlo instead');
    error.code = 'CONDITIONAL_ORBIT_TOO_LARGE';
    throw error;
  }
  const groups = groupsOf(batch, labels, subject, cluster);
  const current = new Array(labels.length).fill(0);
  function* walk(groupIndex) {
    if (groupIndex === groups.length) {
      yield current.slice();
      return;
    }
    const group = groups[groupIndex];
    const positives = group.filter((u) => u.label === 1).length;
    for (const chosen of choose(group.length, positives)) {
      const yes = new Set(chosen);
      group.forEach((unit, j) => {
        for (const i of unit.indices) current[i] = yes.has(j) ? 1 : 0;
      });
      yield* walk(groupIndex + 1);
    }
  }
  yield* walk(0);
}

// Uniform within-batch Fisher-Yates of unit labels; repeated rows can never
// be separated and observed per-batch unit-label counts are fixed.
export function sampleConditionalLabels(batch, labels, random, subject = null, cluster = null) {
  if (typeof random !== 'function') throw new TypeError('A seeded random generator is required');
  const result = labels.slice();
  for (const group of groupsOf(batch, labels, subject, cluster)) {
    const shuffled = group.map((u) => u.label);
    for (let i = shuffled.length - 1; i > 0; i--) {
      const j = Math.floor(random() * (i + 1));
      [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
    }
    group.forEach((unit, j) => {
      for (const i of unit.indices) result[i] = shuffled[j];
    });
  }
  return result;
}
