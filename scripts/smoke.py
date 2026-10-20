"""Tokenizer/data/scorer smoke test. Never loads model weights."""
import sys,os,json,tempfile,hashlib,runpy
from pathlib import Path
R=Path(__file__).resolve().parents[1]
sys.dont_write_bytecode=True
sys.path.insert(0,str(R/'experiments/origin_generation_v3'))
import run_v3 as v3
v2=v3.v2
validation=R/'.validation';validation.mkdir(exist_ok=True)
with tempfile.TemporaryDirectory(dir=validation) as d:
 for mod,name in [(v2,'origin_counterfactual_v2'),(v3,'origin_generation_v3')]:
  original=mod.HERE;mod.HERE=Path(d)/name;mod.HERE.mkdir()
  try:
   mod.make_data()
   for p in (mod.HERE/'data').glob('*.jsonl'):
    expected=R/'evidence'/name/'data'/p.name
    assert p.read_bytes()==expected.read_bytes(),name+'/'+p.name
  finally:mod.HERE=original
runpy.run_path(str(R/'experiments/origin_generation_v3/test_scorer.py'))
# Rescore existing V3 text only; write nothing to evidence.
rows=[json.loads(x) for x in (R/'evidence/origin_generation_v3/predictions.jsonl').read_text().splitlines()]
for r in rows: assert v3.score(r,r['raw'],swap='/swap/' in r['condition'])==r['score']
print(json.dumps({'data_generators':'byte-identical to archived splits','v3_saved_scores_checked':len(rows),'training':False,'model_loading':False,'generation':False}))
