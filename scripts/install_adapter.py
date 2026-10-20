"""Copy a reviewed local adapter into a fresh run directory. No model execution."""
from pathlib import Path
import argparse,shutil,json
p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('--stage',choices=['v2','v3'],required=True);p.add_argument('--variant',choices=['text','origin'],required=True);a=p.parse_args()
r=Path(__file__).resolve().parents[1];name={'v2':'origin_counterfactual_v2','v3':'origin_generation_v3'}[a.stage];dest=r/'experiments'/name/'checkpoints'/a.variant
if dest.exists():raise SystemExit('Destination exists; refusing overwrite')
for name in ['lora/adapter_config.json','lora/adapter_model.safetensors']+(['origin_embedding.pt'] if a.variant=='origin' else []):
 if not (a.source/name).is_file():raise SystemExit('Missing '+name)
config=json.loads((a.source/'lora/adapter_config.json').read_text());assert config['base_model_name_or_path']=='Qwen/Qwen3-0.6B'
dest.mkdir(parents=True)
shutil.copytree(a.source/'lora',dest/'lora')
if a.variant=='origin':shutil.copyfile(a.source/'origin_embedding.pt',dest/'origin_embedding.pt')
print('Installed into',dest.relative_to(r))
