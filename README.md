# 4-Origin: explicit runtime origin identity

[中文](README.zh-CN.md) · [Technical report](technical-report.md) · [Limitations](docs/limitations.md) · [Metric ledger](metric-ledger.json)

A small-model research artifact studying whether explicit origin identity can influence controlled decisions and short natural-language answers. **This is not a production anti-hallucination model.** Textual provenance labels remain a strong baseline; these experiments do not demonstrate an architectural advantage for Origin Embedding.

Initial research release. Code and adapters: Apache-2.0; project data and prose: CC BY 4.0. Commercial use is permitted. See [license scope](LICENSING.md). Public attribution is myczh; repository links are listed below.

## Method

| Runtime ID | Name | Meaning |
|---:|---|---|
| 0 | SYSTEM | Runtime/system task instruction |
| 1 | USER | User question or request |
| 2 | EXTERNAL | Reference/tool content supplied by the runtime |
| 3 | SELF | Model-generated content or a supplied internal candidate |

The runtime assigns IDs; the model does not infer or authenticate them. EXTERNAL is provenance, not a guarantee of truth. An accurate SELF restatement may be externally supported. Generated answer tokens receive SELF. Most controlled tests vary only EXTERNAL versus SELF; four-ID competence has not been comprehensively demonstrated.

Origin adds a learned four-row embedding to token embeddings before the frozen transformer. LoRA adapts q_proj/v_proj (rank 8, alpha 16). Text-SFT uses explicit [EXTERNAL]/[SELF] strings. No attention bias is added. Base: Qwen/Qwen3-0.6B, revision `c1899de289a04d12100db370d81485cdf75e47ca`. Inputs use the experiment's plain-text format, **not** the Qwen chat template.

## Counterfactual classification (V2)

Within each pair, fact text, question, tokens and attention masks are identical for Origin; only the fact span's EXTERNAL/SELF identity changes. Targets are YES/NO. Train pool: 2,000 pairs; only 100 pairs actually used before validation stopping. Test: 500 pairs; cross-template: 200 pairs; one training seed.

| Condition | Split | Sample accuracy | Pair accuracy | Source flip |
|---|---|---:|---:|---:|
| text/tagged | test | 100.00% | 100.00% | 100.00% |
| text/tagged | cross_template | 100.00% | 100.00% | 100.00% |
| origin/normal | test | 99.90% | 99.80% | 99.80% |
| origin/zero | test | 50.00% | 0.00% | 0.00% |
| origin/swap | test | 0.10% | 0.00% | 99.80% |
| origin/shuffle | test | 49.50% | 23.20% | 47.40% |
| origin/normal | cross_template | 94.25% | 88.50% | 88.50% |
| origin/zero | cross_template | 50.00% | 0.00% | 0.00% |
| origin/swap | cross_template | 5.75% | 0.00% | 88.50% |
| origin/shuffle | cross_template | 53.75% | 24.50% | 41.50% |

Source: [V2 report](evidence/origin_counterfactual_v2/experiment-report.md), [results](evidence/origin_counterfactual_v2/results.json); every table entry has a JSON path in the ledger. Zero skips the embedding contribution (it does not assign ID 0). Swap exchanges fact-span identities only. Shuffle is a fixed balanced sample-level identity permutation, not per-token shuffling. **V2 swap/shuffle accuracy uses original labels**; low swap accuracy is expected if the model follows the intervened identity. Flip rate alone is not correctness. Text/tagged retains the source information; Text/normal in the archive removes it and is not an information-matched competitor.

The cross-template paired score falls to 88.50%. A post-hoc factorial diagnosis on the same 200 unseen facts found F0/Q0=100%, F1/Q0=100%, F0/Q1=93%, F1/Q1=88.50%; configuration cases were 33/33, 33/33, 21/33, 14/33. Question wording was the main observed trigger, interacting with fact wording. This does not isolate passive voice, length or an individual token as the mechanism. [Original diagnosis](evidence/origin_counterfactual_v2/template_diagnosis/diagnosis-report.txt).

## Controlled single-turn generation (V3)

Fresh adapters were trained from the same base, not continued from V2. Each model used 200 steps / 400 records, with two supplied claims per answer. Test and cross-template each contain 96 records. SELF-promotion denominator is 96 supplied SELF candidates; confirmed-external retention denominator is 72; requested-candidate retention denominator is 48. This measures retention and qualification of supplied candidates, not quality of novel inference.

| Condition | Split | SELF candidate asserted | External retained | Requested candidate retained | Complete content + attribution |
|---|---|---:|---:|---:|---:|
| text/normal | test | 0.00% | 100.00% | 100.00% | 100.00% |
| text/normal | cross_template | 0.00% | 98.61% | 97.92% | 98.96% |
| origin/normal | test | 0.00% | 100.00% | 100.00% | 100.00% |
| origin/zero | test | 41.67% | 55.56% | 50.00% | 50.00% |
| origin/swap | test | 0.00% | 100.00% | 100.00% | 100.00% |
| origin/normal | cross_template | 0.00% | 97.22% | 97.92% | 93.75% |
| origin/zero | cross_template | 48.96% | 65.28% | 50.00% | 48.96% |
| origin/swap | cross_template | 0.00% | 97.22% | 97.92% | 93.75% |

Source: [V3 report](evidence/origin_generation_v3/experiment-report.txt), [results](evidence/origin_generation_v3/results.json), [raw outputs](evidence/origin_generation_v3/predictions.jsonl). **V3 swap is scored against effective swapped provenance**, unlike V2. Normal→swap changes every Origin answer and reproduces its paired counterpart; zero makes paired outputs identical. No V3 shuffle condition was run.

Both trained systems achieved complete correctness on 96/96 ordinary-test answers. Across templates, Text was 95/96 and Origin 90/96. Origin errors include path shortening, identifier changes and an invented path character despite qualified wording. Zero detected promotions of supplied SELF candidates is not zero hallucination. Untuned base outputs all reached the length cap; low parser coverage prevents a trustworthy semantic hallucination comparison against the base. The V2-to-generation transfer also had poor coverage and does not establish general transfer.

## Long-horizon status

V4 produced 16,600 outputs but its scorer has confirmed qualifier, conjunction and uncertainty-scope defects. V4.1 prepared 250 review records; **0 human-verified, 0 formally rescored** at this release snapshot. No V4 error-depth trend, overall hallucination rate or net benefit of source-preserved recovery is presented as a validated result. [Disclosure](docs/long-horizon.md).

## Reproduce

Use a new checkout. Python 3.10+ is the intended minimum; the tested environment and limits are recorded in release-checklist.md. CPU and Apple MPS are selected by the existing code; CUDA is not implemented. A full base-model download and sufficient memory are needed for optional model runs. Exact timings and identical CPU/MPS outputs are not promised.

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
bash scripts/reproduce.sh check
bash scripts/reproduce.sh smoke
```

`check` uses standard-library code only. `smoke` uses the pinned tokenizer, regenerates synthetic datasets in a temporary release-local directory, checks byte identity, and checks existing V3 output scores. It does not load model weights or generate text. Network may be needed for the tokenizer; set HF_HUB_OFFLINE=1 only after caching it.

The following commands are provided for researchers to run later; **they were not executed during release preparation**:

```sh
bash scripts/reproduce.sh data
bash scripts/reproduce.sh train-v2
bash scripts/reproduce.sh eval-v2
bash scripts/reproduce.sh transfer
bash scripts/reproduce.sh train-v3
bash scripts/reproduce.sh eval-v3
```

V2 Text automatically uses the Origin validation-selected step budget. V3 uses a fixed 200 steps per model. Training refuses existing checkpoint directories. Evaluation writes run-local metrics.json/predictions.jsonl; it does not overwrite evidence/. V3 eval's metrics.json contains the original core metrics, while the archive's results.json additionally contains offline summary/manual-review fields. Do not compare unlike fields. To evaluate staged adapters without training, first install both variants using `python scripts/install_adapter.py PATH_TO_ADAPTER --stage v3 --variant text` (repeat with origin), then run data and eval-v3. V2 transfer requires both V2 checkpoints. Ordinary PEFT loading alone is insufficient for Origin: its separate origin_embedding.pt and per-token runtime IDs are mandatory.

Synthetic data generators are the make-data entry points in experiments/. No scraped corpora or private repositories were used in these fixtures. Evidence snapshots retain original predicted text and scores, including failures. [Data provenance](docs/data-provenance.md), [release changes](docs/portability-changes.md), and [environment snapshots](evidence/origin_generation_v3/environment-observed.txt).

## Repositories

GitHub: `https://github.com/myczh-1/four-origin`  
HF Text: `https://huggingface.co/myczh/four-origin-qwen3-0.6b-v3-text`  
HF Origin: `https://huggingface.co/myczh/four-origin-qwen3-0.6b-v3-origin`
