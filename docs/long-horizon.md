# V4 and V4.1: measurement limits

V4 generated 16,600 existing outputs across real generated chains, probes, restoration and interventions. It reused V3 checkpoints without additional training. Its numerical scores are NOT validated release conclusions.

Confirmed scorer problems: E1 failed to recognize hypothesis as qualification; E2 did not split and that conjunctions; E3 propagated uncertainty across an entire sentence rather than the relevant atomic claim. A faithful SELF restatement must not automatically be counted as unsupported. Runtime provenance and factual support are separate.

Example E1: origin-s37-chain-d20-Full-normal says “The pool limit is 166, but this is a hypothesis.” The old scorer called this an unsupported factual assertion. E2: text-s2-probe-d20-Source-Preserved-normal expresses SQLite and pool limit 24 in an and that construction, which the old scorer leaves unresolved. E3: text-s7-conflict-d0-Full-normal has database certainty contaminated by proposed in the pool clause. These examples demonstrate defects, not the share of all failures explained by scoring.

V4.1 status: 250 sampled answers, no human-verified labels, no formal rescore. Its 17 program tests are not human validation. Frozen validation and independent random samples cannot be used for tuning. The release excludes the unvalidated V4 metric tables to prevent accidental headline reuse; the archived status is in evidence/v41-status/.

Do not infer that errors increase with depth, that source-preserved recovery helps on average, that metadata is lost at runtime, or that all failures are scorer artifacts. The previous audit reported reproducible input/source IDs, but that is not evidence of consistent factual use. Recovery can change answers in either direction. Human reference and clustered paired analysis are still needed. No new inference or rescoring was performed in release preparation.
