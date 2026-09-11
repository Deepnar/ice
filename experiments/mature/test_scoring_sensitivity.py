"""Checks of ICE v2 score policy reproduction and paired ordinal invariance."""
from copy import deepcopy
from scoring_sensitivity import ICE, VEC, load, mod, summarize

rows = load(True)
assert [r['scores'] for r in rows] == [r['scores'] for r in mod.load_records()]
assert sum(r['origins'][VEC]=='failed_answer' for r in rows)==130
assert sum(r['origins'][VEC]=='sibling_routing' for r in rows)==2
assert all(r['origins'][ICE]=='explicit' for r in rows)
ordinary=[r for r in rows if r['conversation_id']!=mod.ICE_DEV]
a=summarize(ordinary)
assert a['ordinal_counts']==dict(ice_higher=216,vector_higher=215,tie=626)
changed=deepcopy(ordinary)
for r in changed:
    for c in (ICE,VEC):r['scores'][c]=2**r['scores'][c]
b=summarize(changed)
assert a['net_ordinal_superiority']==b['net_ordinal_superiority']
assert a['ordinal_counts']==b['ordinal_counts']
for r in changed:r['scores'][ICE],r['scores'][VEC]=r['scores'][VEC],r['scores'][ICE]
c=summarize(changed)
assert c['ordinal_counts']['ice_higher']==b['ordinal_counts']['vector_higher']
assert c['net_ordinal_superiority']['delta']==-b['net_ordinal_superiority']['delta']
assert all(abs(x+y)<1e-12 for x,y in zip(c['net_ordinal_superiority']['ci95'],reversed(b['net_ordinal_superiority']['ci95'])))
print('Archived policy reproduction, missingness counts, ordinal invariance, pair reversal: passed')
