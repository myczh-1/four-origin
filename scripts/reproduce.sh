#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1
case "${1:-check}" in
 check) python3 scripts/check.py ;;
 smoke) python3 scripts/smoke.py ;;
 data)
  python3 experiments/origin_counterfactual_v2/run_experiment.py make-data
  python3 experiments/origin_generation_v3/run_v3.py make-data ;;
 train-v2)
  python3 experiments/origin_counterfactual_v2/run_experiment.py train --variant origin --max-steps 600
  steps=$(python3 -c 'import json; print(json.load(open("experiments/origin_counterfactual_v2/training_origin.json"))["steps"])')
  python3 experiments/origin_counterfactual_v2/run_experiment.py train --variant text --max-steps "$steps" ;;
 train-v3)
  python3 experiments/origin_generation_v3/run_v3.py train --variant text
  python3 experiments/origin_generation_v3/run_v3.py train --variant origin ;;
 eval-v2) python3 experiments/origin_counterfactual_v2/run_experiment.py eval ;;
 eval-v3) python3 experiments/origin_generation_v3/run_v3.py eval ;;
 transfer) python3 experiments/origin_generation_v3/run_v3.py transfer ;;
 *) printf '%s\n' 'Usage: reproduce.sh check|smoke|data|train-v2|train-v3|eval-v2|eval-v3|transfer'; exit 2 ;;
esac
