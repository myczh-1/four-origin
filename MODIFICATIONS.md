# Modification and artifact provenance

Base: Qwen/Qwen3-0.6B at c1899de289a04d12100db370d81485cdf75e47ca (Apache-2.0).

lora/adapter_model.safetensors: newly trained LoRA adaptation tensors, not an unchanged base checkpoint. lora/adapter_config.json: experiment-specific PEFT configuration. origin_embedding.pt, when present: newly trained four-row source embedding. These files are licensed Apache-2.0 by the project rights holder; base copyright and attribution are retained separately. Binary files are unaltered by this documentation update. A corresponding .license sidecar records the notice next to each weight file.

V2 uses source-counterfactual classification; V3 uses two supplied claims and constrained natural-language targets. The Origin loader adds runtime-provided source vectors to token embeddings and marks generated tokens SELF. Text variants use textual provenance labels. The source code and reports describe exact budgets and limitations.

Release code changes relative to the original experiments: caller-controlled offline mode and conditional MPS cache cleanup in V3; see the GitHub docs/v3-release.patch. No new training or model generation was performed for this release. No endorsement is implied.
