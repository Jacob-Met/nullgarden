#!/usr/bin/env python3
"""Independent exhaustive and adversarial receiver; authored by consumer, schema-received by root."""
import copy, hashlib, itertools, json, sys, traceback
import numpy as np
import batch_label_schedule as api

RESULTS=[]
def check(name,fn):
    try:
        detail=fn(); RESULTS.append({'case':name,'passed':True,'detail':detail})
    except Exception:
        RESULTS.append({'case':name,'passed':False,'traceback':traceback.format_exc()})
def canonical(x):
    return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode('utf-8')
def rehash(s):
    s['integrity_sha256']=hashlib.sha256(canonical({k:v for k,v in s.items() if k!='integrity_sha256'})).hexdigest()
    return s
def reject(fn):
    try: fn()
    except (ValueError,KeyError): return
    raise AssertionError('Expected ValueError/KeyError rejection')
def valid(d,s,strict=False):
    r=api.validate_schedule(d,s,strict_replay=strict)
    assert r['valid'] is True and r['integrity']=='verified_unkeyed_digest'
    assert r['rng_replay']==('verified' if strict else 'unverified')
    assert r['scientific_assumptions']=='unverified'
    return r
def base():
    return {'X':np.arange(18,dtype=np.float64).reshape(6,3)/7,
            'y':[0,1,1,0,1,0], 'groups':['A','A','A','B','B','C'],
            'sample_ids':['s'+str(i) for i in range(6)],
            'feature_names':['f0','f1','f2']}
def support(x):
    assert type(x) is int,repr(x)
    return x
def partitions(d):
    return [[i for i,g in enumerate(d['groups']) if g==v] for v in dict.fromkeys(d['groups'])]
def enumerate_reference(d):
    perms=[]
    for choices in itertools.product(*(list(itertools.permutations(ix)) for ix in partitions(d))):
        p=list(range(len(d['y'])))
        for ix,v in zip(partitions(d),choices):
            for dest,src in zip(ix,v): p[dest]=src
        perms.append(tuple(p))
    counts={}
    for p in perms:
        labels=tuple(d['y'][i] for i in p); counts[labels]=counts.get(labels,0)+1
    return set(perms),counts
D=base(); S=None

def enumeration_case():
    global S
    S=api.make_schedule(D,seed=20261029,draws=256)
    valid(D,S); valid(D,S,True)
    perms,counts=enumerate_reference(D)
    assert len(perms)==12 and len(counts)==6 and set(counts.values())=={2}
    diag=S['diagnostics']; assert support(diag['support'])==6
    assert diag['max_hamming']==4 and diag['frozen_labels']==1
    expected={'A':(3,2,1,3,2),'B':(2,1,1,2,2),'C':(1,0,1,1,0)}
    for row in diag['groups']:
        n,pos,neg,sup,mh=expected[row['group']]
        assert (row['n'],row['positive'],row['negative'],support(row['support']),row['max_hamming'])==(n,pos,neg,sup,mh)
        assert row['pure']==(pos==0 or neg==0)
    baseline=S['baseline']; assert baseline['index_permutation']==list(range(6)) and baseline['labels']==D['y'] and baseline['hamming']==0
    frequency={str(k):0 for k in counts}; observed=[]
    for draw in S['draws']:
        p=tuple(draw['index_permutation']); labels=tuple(draw['labels'])
        assert p in perms and labels==tuple(D['y'][i] for i in p)
        h=sum(a!=b for a,b in zip(labels,D['y'])); assert draw['hamming']==h<=4
        assert draw['dataset_sha256']==S['binding']['dataset_sha256']
        frequency[str(labels)]+=1; observed.append(labels)
    assert len(observed)==256
    summary=S['summary']; assert summary['draw_count']==256
    drawn_counts={v:observed.count(v) for v in set(observed)}
    assert summary['unique_assignments']==len(drawn_counts)
    assert summary['repeated_assignment_draws']==len(observed)-len(drawn_counts)
    assert summary['assignment_patterns_repeated']==sum(n>1 for n in drawn_counts.values())
    assert summary['unchanged_assignment_draws']==sum(v==tuple(D['y']) for v in observed)
    assert S['assumptions_verified'] is False
    independent=copy.deepcopy(S); rehash(independent); assert independent['integrity_sha256']==S['integrity_sha256']
    return {'frequency_descriptive_only':frequency,'index_support':12,'label_support':6}

def canonical_matrix():
    for dtype in ('<f4','>f8'):
        d=base(); d['X']=np.asarray(d['X'],dtype=dtype)
        s=api.make_schedule(d,seed=7,draws=2)
        expected=hashlib.sha256(np.asarray(d['X'],dtype='<f8',order='C').tobytes(order='C')).hexdigest()
        assert s['binding']['matrix_sha256']==expected
        assert s['binding']['matrix_convention']=='<f8 C'
        valid(d,s,True)

def small_cases():
    for y,g in [([0,0,1],['A','A','B']),([0,1,0],['A','B','C'])]:
        d={'X':np.ones((3,1)),'y':y,'groups':g,'sample_ids':['a','b','c'],'feature_names':['f']}
        s=api.make_schedule(d,seed=0,draws=16); valid(d,s,True)
        assert support(s['diagnostics']['support'])==1
        assert s['summary']['support_status']=='uninformative_under_conditioning'
        assert s['summary']['unchanged_assignment_draws']==16
        assert s['summary']['unique_assignments']==1 and s['summary']['repeated_assignment_draws']==15
        assert len(s['draws'])==16 and all(r['labels']==y for r in s['draws'])
    z=api.make_schedule(D,seed=0,draws=0); valid(D,z,True)
    assert z['draws']==[] and z['baseline']['labels']==D['y']
    assert z['summary']['unique_assignments']==0 and z['summary']['unchanged_assignment_draws']==0
    a=api.make_schedule(D,seed=2**128-1,draws=3)
    b=api.make_schedule(D,seed=2**128-1,draws=3)
    assert canonical(a)==canonical(b)
    d=base(); d['subject_ids']=['p','p','q','r','s','t']
    s=api.make_schedule(d,seed=4,draws=3); valid(d,s,True)
    assert s['diagnostics']['subject_ids_provided'] is True
    assert s['diagnostics']['repeated_subject_ids']==['p']
    assert s['assumptions_verified'] is False

def input_cases():
    badX=[[[1,2]],np.ones(6),np.ones((6,3),dtype=complex),np.ones((6,3),dtype=bool),np.full((6,3),np.nan),np.full((6,3),np.inf),np.ones((6,3),dtype=object),np.full((6,3),2**53+1,dtype=np.int64)]
    for i,x in enumerate(badX):
        d=base(); d['X']=x
        check('reject X '+str(i),lambda d=d:reject(lambda:api.make_schedule(d,seed=0,draws=1)))
    changes=[('y',[False,1,1,0,1,0]),('y',[0,1,2,0,1,0]),('y',[0,1,1]),('y',['0','1','1','0','1','0']),('groups',['A']*5),('sample_ids',['s']*6),('sample_ids',[' s0']+D['sample_ids'][1:]),('feature_names',['f','f','g']),('feature_names',['f0','f1']),('groups',['A','A','A','B','B',1]),('subject_ids',['x'])]
    for i,(key,value) in enumerate(changes):
        d=base(); d[key]=value
        check('reject metadata '+str(i),lambda d=d:reject(lambda:api.make_schedule(d,seed=0,draws=1)))
    for seed in (-1,2**128,True,1.5,'1'):
        check('reject seed '+repr(seed),lambda seed=seed:reject(lambda:api.make_schedule(D,seed=seed,draws=1)))
    for draws in (-1,100001,True,1.5,'1'):
        check('reject draws '+repr(draws),lambda draws=draws:reject(lambda:api.make_schedule(D,seed=1,draws=draws)))
    for budget in (-1,True,1.5,23):
        check('reject budget '+repr(budget),lambda budget=budget:reject(lambda:api.make_schedule(D,seed=1,draws=1,max_schedule_cells=budget)))
    check('exact cell budget24',lambda:valid(D,api.make_schedule(D,seed=1,draws=1,max_schedule_cells=24)))

def tamper_case(name,mutate,rehash_it=True,strict=False):
    def run():
        assert S is not None,'enumeration setup failed'
        t=copy.deepcopy(S); mutate(t)
        if rehash_it: rehash(t)
        reject(lambda:valid(D,t,strict))
    check(name,run)
def setpath(t,path,value):
    for key in path[:-1]: t=t[key]
    t[path[-1]]=value

def tampering():
    tamper_case('digest without rehash',lambda t:setpath(t,['summary','draw_count'],255),False)
    def crossing(t):
        p=t['draws'][0]['index_permutation']; p[0],p[3]=p[3],p[0]
    def duplicate(t): t['draws'][0]['index_permutation'][0]=t['draws'][0]['index_permutation'][1]
    def mismatch(t): t['draws'][0]['labels'][0]=1-t['draws'][0]['labels'][0]
    for name,fn in [('cross groups',crossing),('nonbijection',duplicate),('label mismatch',mismatch)]: tamper_case(name,fn)
    tamper_case('baseline changed',lambda t:t['baseline']['index_permutation'].reverse())
    tamper_case('diagnostics wrong',lambda t:setpath(t,['diagnostics','max_hamming'],5))
    tamper_case('summary wrong',lambda t:setpath(t,['summary','draw_count'],255))
    tamper_case('binding wrong',lambda t:setpath(t,['binding','dataset_sha256'],'0'*64))
    tamper_case('conditions changed',lambda t:t['conditions'].append('receiver-injected unsupported condition'))
    tamper_case('assumptions changed',lambda t:setpath(t,['assumptions_verified'],True))
    tamper_case('Boolean baseline labels',lambda t:setpath(t,['baseline','labels',0],False))
    tamper_case('Boolean ordered labels',lambda t:setpath(t,['ordered_metadata','y',0],False))
    tamper_case('Boolean draw index',lambda t:setpath(t,['draws',0,'index_permutation',0],False))
    tamper_case('Boolean frozen count',lambda t:setpath(t,['diagnostics','frozen_labels'],True))
    def bool_summary():
        t=api.make_schedule(D,seed=0,draws=1)
        t['summary']['draw_count']=True; rehash(t)
        reject(lambda:valid(D,t))
    check('Boolean summary count',bool_summary)
    tamper_case('unsupported RNG',lambda t:setpath(t,['rng','algorithm'],'unsupported'))
    def replay_only(name,fn):
        def run():
            t=copy.deepcopy(S); fn(t); rehash(t)
            valid(D,t,False); reject(lambda:valid(D,t,True))
        check(name,run)
    replay_only('seed changed structural only',lambda t:setpath(t,['rng','seed'],20261030))
    replay_only('draw order structural only',lambda t:t['draws'].reverse())
    replay_only('initial state changed structural only',lambda t:setpath(t,['rng','initial_state'],np.random.PCG64(991).state))
    replay_only('final state changed structural only',lambda t:setpath(t,['rng','final_state'],np.random.PCG64(991).state))
    replay_only('NumPy version structural only',lambda t:setpath(t,['rng','numpy_version'],'0.0.receiver-mismatch'))
    for key in ('X','y','groups','sample_ids','feature_names','subject_ids'):
        def run(key=key):
            d=base()
            if key=='X': d['X'][0,0]+=1
            elif key=='y': d['y'][0]=1
            elif key=='groups': d['groups'][0]='Z'
            elif key=='sample_ids': d['sample_ids'][0]='new-id'
            elif key=='feature_names': d['feature_names'][0]='new-feature'
            else: d['subject_ids']=['p'+str(i) for i in range(6)]
            reject(lambda:valid(d,S))
        check('dataset binding '+key,run)

if __name__=='__main__':
    check('tiny exhaustive reference',enumeration_case)
    check('canonical floating matrix',canonical_matrix)
    check('pure singleton zero deterministic subjects',small_cases)
    input_cases()
    if S is not None: tampering()
    print(json.dumps({'schema':'nullgarden.independent-schedule-receiver.v1','passed':all(r['passed'] for r in RESULTS),'cases':RESULTS,'limits':['descriptive frequencies are not a uniformity proof','unkeyed digest is not authenticity','schedule validity does not establish exchangeability or scientific suitability']},indent=2,allow_nan=False))
    sys.exit(0 if all(r['passed'] for r in RESULTS) else 1)
