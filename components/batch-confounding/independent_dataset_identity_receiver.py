"""Consumer regression for descriptive datasets with large conditional support.

No schedule internals are inspected; identities follow the documented canonical
format and small-input admission is compared with the existing public API.
"""
import copy
import hashlib
import json
import sys
import traceback
import numpy as np
import batch_label_schedule as schedule
import batch_confounding as diagnostic

RESULTS = []
GUARD = sys.get_int_max_str_digits()


def case(name, fn):
    try:
        detail = fn()
        assert sys.get_int_max_str_digits() == GUARD, 'Receiver changed process integer serialization guard'
        RESULTS.append({'name': name, 'passed': True, 'detail': detail})
    except Exception:
        RESULTS.append({'name': name, 'passed': False, 'traceback': traceback.format_exc()})


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode('utf-8')


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def dataset(n=6):
    return {'X': np.arange(n * 2, dtype=np.float64).reshape(n, 2) / 8,
            'y': [0, 1] * (n // 2), 'groups': ['single'] * n,
            'sample_ids': ['s' + str(i) for i in range(n)],
            'feature_names': ['left', 'right'],
            'subject_ids': ['subject' + str(i) for i in range(n)]}


def expected_identity(data):
    metadata = {name: list(data[name]) for name in ['sample_ids', 'groups', 'y', 'feature_names']}
    metadata['subject_ids'] = None if data.get('subject_ids') is None else list(data['subject_ids'])
    binding = {'shape': list(data['X'].shape), 'matrix_convention': '<f8 C',
               'matrix_sha256': hashlib.sha256(np.asarray(data['X'], dtype='<f8', order='C').tobytes(order='C')).hexdigest(),
               'ordered_metadata_sha256': digest(metadata)}
    binding['dataset_sha256'] = digest(binding)
    return {'binding': binding, 'ordered_metadata': metadata}


def balanced(n):
    data = dataset(n)
    original = copy.deepcopy(data)
    identity = schedule.validate_dataset_identity(data)
    assert identity == expected_identity(data)
    report = diagnostic.describe_batch_confounding(data)
    assert report['dataset_binding'] == identity['binding']
    assert report['totals']['n'] == n
    assert report['totals']['positive'] == report['totals']['negative'] == n // 2
    assert report['score']['auc'] == {'status': 'defined', 'wins': 0, 'ties': (n // 2) ** 2,
                                    'losses': 0, 'pairs': (n // 2) ** 2,
                                    'auc_fraction': {'numerator': 1, 'denominator': 2}, 'auc_decimal': 0.5}
    assert report['design'] == {'batch_rank': 1, 'batch_plus_target_rank': 2, 'target_in_batch_indicator_span': False}
    assert report['subjects']['unique_count'] == n
    # The consumer's output must actually serialize with the existing guard.
    encoded = canonical(report)
    assert json.loads(encoded)['dataset_binding'] == identity['binding']
    np.testing.assert_array_equal(data['X'], original['X'])
    assert all(data[k] == original[k] for k in data if k != 'X')
    if n == 10_000:
        legacy = schedule.make_schedule(data, seed=0, draws=0)
        assert identity == {k: legacy[k] for k in ['binding', 'ordered_metadata']}
        assert schedule.validate_schedule(data, legacy, strict_replay=True)['valid'] is True
    return {'samples': n, 'ties': (n // 2) ** 2, 'auc': '1/2', 'report_bytes': len(encoded), 'dataset_sha256': identity['binding']['dataset_sha256']}


def rejected(fn):
    try:
        fn()
    except (ValueError, KeyError):
        return True
    return False


def admission():
    mutations = [
        ('nonfinite matrix', 'X', np.full((6, 2), np.nan)),
        ('complex matrix', 'X', np.ones((6, 2), dtype=complex)),
        ('Boolean matrix', 'X', np.ones((6, 2), dtype=bool)),
        ('object matrix', 'X', np.ones((6, 2), dtype=object)),
        ('wrong dimensions', 'X', np.ones(6)),
        ('Boolean label', 'y', [False, 1, 0, 1, 0, 1]),
        ('out of range label', 'y', [0, 2, 0, 1, 0, 1]),
        ('short labels', 'y', [0, 1]),
        ('duplicate samples', 'sample_ids', ['same'] * 6),
        ('padded sample', 'sample_ids', [' s0', 's1', 's2', 's3', 's4', 's5']),
        ('duplicate features', 'feature_names', ['same', 'same']),
        ('mixed group type', 'groups', ['A', 'A', 'A', 'A', 'A', 1]),
        ('short subjects', 'subject_ids', ['subject']),
        ('invalid subjects', 'subject_ids', ['p', 'q', 'r', 's', 't', '']),
    ]
    for name, field, value in mutations:
        data = dataset(); data[field] = value
        assert rejected(lambda: schedule.make_schedule(data, seed=0, draws=0)), name + ' legacy admitted'
        assert rejected(lambda: schedule.validate_dataset_identity(data)), name + ' identity admitted'
        assert rejected(lambda: diagnostic.describe_batch_confounding(data)), name + ' diagnostic admitted'
    for budget in [11, 0, -1, True, 12.0, '12']:
        data = dataset()
        assert rejected(lambda: schedule.make_schedule(data, seed=0, draws=0, max_schedule_cells=budget)), repr(budget)
        assert rejected(lambda: schedule.validate_dataset_identity(data, max_schedule_cells=budget)), repr(budget)
        assert rejected(lambda: diagnostic.describe_batch_confounding(data, max_schedule_cells=budget)), repr(budget)
    data = dataset()
    legacy = schedule.make_schedule(data, seed=0, draws=0, max_schedule_cells=12)
    assert schedule.validate_schedule(data, legacy, strict_replay=True, max_schedule_cells=12)['valid'] is True
    assert schedule.validate_dataset_identity(data, max_schedule_cells=12) == {k: legacy[k] for k in ['binding', 'ordered_metadata']}
    assert diagnostic.describe_batch_confounding(data, max_schedule_cells=12)['dataset_binding'] == legacy['binding']
    return {'invalid_datasets_rejected': len(mutations), 'invalid_budgets_rejected': 6, 'exact_baseline_budget': 12}


def detached():
    data = dataset()
    identity = schedule.validate_dataset_identity(data)
    original = expected_identity(data)
    identity['binding']['shape'][0] = 99
    identity['binding']['matrix_sha256'] = 'changed'
    for name in ['sample_ids', 'groups', 'y', 'feature_names', 'subject_ids']:
        identity['ordered_metadata'][name][0] = 'changed'
    assert schedule.validate_dataset_identity(data) == original
    received = schedule.validate_dataset_identity(data)
    data['X'][0, 0] = 99
    for name in ['sample_ids', 'groups', 'feature_names', 'subject_ids']:
        data[name][0] = 'input-edited'
    data['y'][0] = 1
    assert received == original
    return 'Both result-to-input and input-to-result mutations remain detached'


if __name__ == '__main__':
    case('10000-row balanced control retains public schedule identities', lambda: balanced(10_000))
    case('20000-row balanced descriptive report survives default integer guard', lambda: balanced(20_000))
    case('Malformed input and exact/refused baseline budgets retain admission policy', admission)
    case('Returned identity and metadata are detached from input state', detached)
    passed = all(row['passed'] for row in RESULTS) and sys.get_int_max_str_digits() == GUARD
    print(json.dumps({'schema': 'nullgarden.independent-dataset-identity-receiver.v1', 'passed': passed,
                      'integer_guard_before': GUARD, 'integer_guard_after': sys.get_int_max_str_digits(),
                      'cases': RESULTS, 'limits': ['Descriptive dataset validation only; no fitted model or inference.',
                                                   'Large-support schedule artifact serialization is unchanged.']}, indent=2, allow_nan=False))
    raise SystemExit(0 if passed else 1)
