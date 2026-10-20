"""Scoring guardrails: do not reward omission or confuse overlapping numeric values."""
import run_v3 as v
r={'atoms':[{'key':'pool limit','value':'8','uncertain':False},
            {'key':'pool limit','value':'48','uncertain':False}],
   'sources':[v.EXTERNAL,v.SELF],'speculate':True,'case':'conflict'}
good=v.score(r,'The reference confirms that the pool limit is 8. It is possible that the pool limit is 48, but this is not confirmed by the reference.')
assert [a['observed'] for a in good['atoms']]==['confirmed','inference']
bad=v.score(r,'The pool limit is 8. The pool limit is 48.')
m=v.aggregate([{**r,'score':bad}])
assert m['unsupported_assertion_rate_lower_bound']==1
assert m['external_fact_retention']==1
assert m['requested_supplied_inference_retention']==0
empty=v.aggregate([{**r,'score':v.score(r,'')}])
assert empty['unsupported_assertion_rate_lower_bound']==0
assert empty['external_fact_retention']==0 and empty['fully_scorable_rate']==0
extra=v.score(r,v.target(r)+' The owner is Alice.')
assert extra['unparsed_sentences']==['The owner is Alice.'] and not extra['fully_scorable']
combined=v.score(r,'The pool limit is 8 or 48.')
assert all(a['observed']=='unresolved' for a in combined['atoms'])
r['atoms'][0]['uncertain']=True
promoted=v.aggregate([{**r,'score':v.score(r,'The pool limit is 8. The pool limit is 48.')}])
assert promoted['unsupported_assertion_rate_lower_bound']==1
assert all(a['observed']==a['expected'] for a in v.score(r,v.target(r))['atoms'])
assert all(a['observed']=='unresolved' for a in v.score(r,'Is the pool limit 8 or 48?')['atoms'])
assert v.score(r,'[SELF] The pool limit is 48.')['atoms'][1]['observed']=='unresolved'
assert v.score(r,'The pool limit is 8.7.')['atoms'][0]['observed']=='omitted'
print('9 scorer guardrail cases passed')
