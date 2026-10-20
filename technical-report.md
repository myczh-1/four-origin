# Explicit Origin Identity in Small Language Models: Controlled Experiments and Limitations

Research note, 2026-10-20. Not a peer-reviewed paper or a production model claim. Author: myczh (individual pseudonym). Repository identities await confirmation.

## Motivation

A generated assertion may later be reused as if it were evidence. We investigate a narrower prerequisite: can an explicit runtime origin channel affect decisions and short answers while retaining a distinction between reference statements and internal candidates? Origin identity is not truth, and source sensitivity is not sufficient for reliable reasoning.

## Method

Four IDs encode SYSTEM, USER, EXTERNAL and SELF. The runtime supplies them. A four-row learned embedding is added to token embeddings of frozen Qwen3-0.6B; LoRA adapts q_proj/v_proj with rank 8 and alpha 16. Text-SFT receives ordinary provenance strings. Base revision is c1899de289a04d12100db370d81485cdf75e47ca. No attention bias is used; all answer tokens are SELF, even when they faithfully restate reference facts. Primary manipulations distinguish EXTERNAL/SELF rather than comprehensively testing all four roles.

## Counterfactual Evaluation

V2 holds each paired prompt's tokens constant for Origin and changes only the fact-span identity. Ordinary-test pair accuracy is 99.80%; it becomes 0% with source contribution removed. Swap produces 0.10% sample accuracy against original labels, consistent with 99.90% accuracy against effective swapped identity. Balanced sample-level shuffling gives 23.20% pair accuracy. Text with tags achieves 100% pair accuracy on both ordinary and cross-template tests.

Origin cross-template pair accuracy is 88.50%. A post-hoc same-fact factorial comparison gives 100/100/93/88.50% for old/new fact crossed with old/new question wording (F0Q0,F1Q0,F0Q1,F1Q1). It implicates question wording and interaction at the input level, without identifying a specific neural or grammatical cause. Only 100 of 2000 training pairs were seen; validation selected the stop, not test results. Evidence: [V2 report](evidence/origin_counterfactual_v2/experiment-report.md), [factorial diagnosis](evidence/origin_counterfactual_v2/template_diagnosis/diagnosis-report.txt).

## Controlled Generation

Fresh V3 adapters use equal 200-step budgets, not the classification checkpoints. Ordinary test has 96 records; both trained systems retain and qualify both supplied claims correctly on all 96. Origin zeroing promotes 40/96 SELF candidates to facts while retaining only 40/72 confirmed external facts. Swapping identities changes all Origin outputs and yields the normal paired counterpart; V3 swap is evaluated against effective identities, unlike V2. No V3 shuffle result exists.

Cross-template complete correctness is 95/96 for Text and 90/96 for Origin. Origin makes content errors despite preserving qualification, including altered path values. Thus there is evidence of source-conditioned controlled generation but no architectural advantage over textual labels. Supplied-candidate retention is not novel reasoning quality; low base parsing coverage prevents a reliable overall semantic hallucination comparison. [V3 report](evidence/origin_generation_v3/experiment-report.txt), [limited manual failure review](evidence/origin_generation_v3/manual-review.json).

## Long-Horizon Limitations

V4's 16,600 outputs are retained privately in the original experiment, but qualifier, conjunction and uncertainty-scope scoring defects prevent treating its scores as validated long-horizon effects. V4.1 has 250 review candidates and zero human-verified answers or formal rescores. No error-depth trend or recovery net benefit is established here. Metadata preservation, its influence on output, and evidence-faithful generation are distinct claims. [Disclosure](docs/long-horizon.md).

## Related Work

[LoRA](https://arxiv.org/abs/2106.09685) provides the parameter-efficient adaptation method used here. [The Instruction Hierarchy](https://arxiv.org/abs/2404.13208) studies prioritization of instructions with different privilege levels; our controlled task concerns provenance and factual qualification, rather than establishing robustness to adversarial instructions. Neither is a direct experimental baseline in this project. This small study does not claim novelty over all provenance-aware methods or a comprehensive literature review.

## Reproducibility

Source, deterministic generators, raw V2/V3 predictions, historical configurations, dependency snapshots and metric paths are bundled. [Metric ledger](metric-ledger.json) maps headline values to original reports and JSON paths; [provenance](provenance.json) records original and sanitized-copy hashes. Clean-run code preserves training/scoring logic except documented offline/cache-cleanup portability edits. All evidence checks and tokenizer-only generation smoke tests avoid model inference; future training/evaluation commands are supplied but not run for this release. CPU/MPS behavior and single-seed limits prevent assuming exact numerical replication everywhere.

Code and adapters are licensed Apache-2.0; project data and report prose are CC BY 4.0. Commercial use is allowed. Scope and upstream attribution are documented in LICENSING.md and NOTICE. No base weights are redistributed. Repository links are provided in the README.
