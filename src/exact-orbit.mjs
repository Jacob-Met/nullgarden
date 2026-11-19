// Exact conditional randomization over label allocations within immutable batches.
// Generated lazily: never store an exponential permutation table in memory.
// The observed assignment is part of the orbit exactly once. Ties count.
// No model or random number generator is used in enumeration.
function groupsOf(batch, labels) {
  if (!Array.isArray(batch) || !Array.isArray(labels)
      || !batch.length || batch.length !== labels.length) {
    throw new TypeError('Exact conditional orbit requires aligned nonempty batch and binary label arrays');
  }
  const map = new Map();
  batch.forEach((id, i) => {
    if (labels[i] !== 0 && labels[i] !== 1) {
      throw new TypeError('Exact conditional orbit requires binary labels');
    }
    if (!map.has(id)) map.set(id, {id, indices:[], positives:0});
    const g = map.get(id);
    g.indices.push(i);
    g.positives += labels[i];
  });
  return [...map.values()];
}

// Distinct permutations are combinations, not factorial permutations of
// indistinguishable labels. Clamp once the product exceeds a caller budget.
export function conditionalOrbitSize(batch, labels, cap = 4096) {
  if (!Number.isSafeInteger(cap) || cap < 1) {
    throw new RangeError('Conditional-orbit cap must be a positive safe integer');
  }
  let size = 1n;
  const limit = BigInt(cap);
  for (const g of groupsOf(batch, labels)) {
    const n = g.indices.length, k = Math.min(g.positives, n - g.positives);
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

function* choose(ids, count, start = 0, selected = []) {
  if (count === 0) {
    yield selected.slice();
    return;
  }
  for (let i = start; i <= ids.length - count; i++) {
    selected.push(ids[i]);
    yield* choose(ids, count - 1, i + 1, selected);
    selected.pop();
  }
}

export function* conditionalLabelAssignments(batch, labels, {maxAssignments = 4096} = {}) {
  const size = conditionalOrbitSize(batch, labels, maxAssignments);
  if (size > maxAssignments) {
    const error = new RangeError('Conditional label orbit exceeds exact enumeration budget; choose bounded Monte Carlo instead');
    error.code = 'CONDITIONAL_ORBIT_TOO_LARGE';
    throw error;
  }
  const groups = groupsOf(batch, labels);
  const current = new Array(labels.length).fill(0);
  function* walk(groupIndex) {
    if (groupIndex === groups.length) {
      yield current.slice();
      return;
    }
    const g = groups[groupIndex];
    for (const positive of choose(g.indices, g.positives)) {
      for (const i of g.indices) current[i] = 0;
      for (const i of positive) current[i] = 1;
      yield* walk(groupIndex + 1);
    }
  }
  yield* walk(0);
}
