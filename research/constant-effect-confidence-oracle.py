#!/usr/bin/env python3
"""Independent exhaustive receiver for the cluster constant-effect confidence set.

Reads cases containing raw datasets and product JSON on stdin. Python Fraction,
raw subject/cluster reduction, and bitmask allocation enumeration share no code
with the JavaScript product or its affine sweep. No stochastic coverage claim.
"""
import itertools
import json
import sys
from fractions import Fraction as F


def rational(endpoint):
    return F(int(endpoint['numerator']), int(endpoint['denominator']))


def prepare(ds):
    subjects = {}
    for i, subject in enumerate(ds['subject']):
        subjects.setdefault(subject, []).append(i)
    clusters = {}
    for rows in subjects.values():
        i = rows[0]
        assert all(ds['cluster'][j] == ds['cluster'][i] and ds['y'][j] == ds['y'][i] for j in rows)
        value = sum((F(ds['X'][j][0]) for j in rows), F(0)) / len(rows)
        clusters.setdefault(ds['cluster'][i], []).append((value, ds['batch'][i], ds['y'][i]))
    units = []
    for group in clusters.values():
        assert all(g[1:] == group[0][1:] for g in group)
        units.append((sum((g[0] for g in group), F(0)) / len(group), group[0][1], group[0][2]))
    strata = {}
    for i, (_, batch, _) in enumerate(units):
        strata.setdefault(batch, []).append(i)
    assignments = []
    # Independent full bitmask filter, not production's combinations generator.
    for bits in itertools.product((0, 1), repeat=len(units)):
        if all(sum(bits[i] for i in indices) == sum(units[i][2] for i in indices)
               for indices in strata.values()):
            assignments.append(bits)
    observed = tuple(u[2] for u in units)

    def statistic(adjusted, allocation):
        contrasts = []
        for indices in strata.values():
            treated = [adjusted[i] for i in indices if allocation[i]]
            control = [adjusted[i] for i in indices if not allocation[i]]
            contrasts.append(sum(treated, F(0)) / len(treated) - sum(control, F(0)) / len(control))
        return sum(contrasts, F(0)) / len(contrasts)

    def count(tau):
        adjusted = [value - tau * label for value, _, label in units]
        threshold = abs(statistic(adjusted, observed))
        return sum(abs(statistic(adjusted, labels)) >= threshold for labels in assignments)

    return count, len(assignments)


def contains(component, tau):
    if component['lower'] is not None:
        lower = rational(component['lower'])
        if tau < lower or (tau == lower and not component['lowerClosed']):
            return False
    if component['upper'] is not None:
        upper = rational(component['upper'])
        if tau > upper or (tau == upper and not component['upperClosed']):
            return False
    return True


def main(cases):
    checked_cells = checked_values = 0
    coverage = {}
    for case in cases:
        result = case['result']['confidenceSet']
        assert result['status'] == 'available' and result['exact']
        count, orbit = prepare(case['dataset'])
        assert orbit == result['orbitSize'] == result['assignmentsEvaluated']
        alpha = F(str(result['alpha']))
        checks = [F(0), F(case['trueTau'])]
        if case.get('checkCells'):
            for cell in result['cells']:
                if cell['kind'] == 'point':
                    tau = rational(cell['lower'])
                elif cell['lower'] is None and cell['upper'] is None:
                    tau = F(0)
                elif cell['lower'] is None:
                    tau = rational(cell['upper']) - 1
                elif cell['upper'] is None:
                    tau = rational(cell['lower']) + 1
                else:
                    tau = (rational(cell['lower']) + rational(cell['upper'])) / 2
                n = count(tau)
                assert n == cell['exceedances'], (case['id'], tau, n, cell)
                assert cell['accepted'] == (F(n, orbit) > alpha), (case['id'], cell)
                checks.append(tau)
                checked_cells += 1
        for tau in checks:
            expected = F(count(tau), orbit) > alpha
            actual = any(contains(component, tau) for component in result['components'])
            assert actual == expected, (case['id'], str(tau), actual, expected)
            checked_values += 1
        assert result['zeroExceedances'] == count(F(0))
        assert case['result']['p'] == count(F(0)) / orbit
        if case.get('coverageGroup'):
            entry = coverage.setdefault(case['coverageGroup'], {'cases': 0, 'covered': 0, 'alpha': alpha})
            entry['cases'] += 1
            entry['covered'] += any(contains(c, F(case['trueTau'])) for c in result['components'])
    for group, result in coverage.items():
        assert F(result['covered'], result['cases']) >= 1 - result['alpha'], (group, result)
        result['alpha'] = float(result['alpha'])
    return {'oracle': 'independent raw-row Fraction / whole-cluster bitmask', 'cases': len(cases),
            'cells': checked_cells, 'membershipChecks': checked_values, 'coverage': coverage}


if __name__ == '__main__':
    print(json.dumps(main(json.load(sys.stdin)), sort_keys=True))
