"""Offline artifact checks: no model loading or inference."""
from pathlib import Path
import json,hashlib,ast,collections
R=Path(__file__).resolve().parents[1]
def lines(p):return [json.loads(x) for x in p.read_text().splitlines()]
def main():
 ledger=json.loads((R/'metric-ledger.json').read_text())
 for x in ledger:
  value=json.loads((R/x['source']).read_text())
  for k in x['json_path']:value=value[k]
  assert value==x['value'],x['id']
  assert (R/x['original_report']).is_file()
 for p in R.rglob('*.py'):ast.parse(p.read_text(),filename=str(p.relative_to(R)))
 for name in ['origin_counterfactual_v2','origin_generation_v3']:
  E=R/'evidence'/name;results=json.loads((E/'results.json').read_text())
  data=results.get('data_validation',results.get('data'))['splits']
  for split,m in data.items(): assert hashlib.sha256((E/'data'/f'{split}.jsonl').read_bytes()).hexdigest()==m['sha256']
  rows=lines(E/'predictions.jsonl');groups=collections.defaultdict(list)
  for x in rows:
   key=f"{x['variant']}/{x['condition']}/{x['split']}" if name.endswith('v2') else x['condition']
   groups[key].append(x)
  assert set(groups)==set(results['metrics'])
  if name.endswith('v2'):
   for key,rs in groups.items():
    pairs=collections.defaultdict(list)
    for x in rs:pairs[x['pair_id']].append(x)
    assert all(len(p)==2 for p in pairs.values())
    sample=sum(x['raw'].strip()==x['answer'] for x in rs)/len(rs)
    paired=sum(all(x['raw'].strip()==x['answer'] for x in p) for p in pairs.values())/len(pairs)
    assert sample==results['metrics'][key]['sample_accuracy']
    assert paired==results['metrics'][key]['pair_accuracy']
 print(json.dumps({'ledger_entries_checked':len(ledger),'data_hashes':'PASS','v2_accuracy_recomputed_from_raw':'PASS','python_syntax':'PASS','model_inference':False}))
if __name__=='__main__':main()
