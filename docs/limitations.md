# Known limitations and failures

- Single training seed; short controlled English prompts; no independent replication or multi-seed confidence interval. No general language, reasoning, security or production claim.
- Runtime IDs are supplied and must be trusted; this is not source authentication. Primarily EXTERNAL/SELF was manipulated, not all four roles comprehensively tested.
- V2 can be solved by reading one fact span's identity without understanding its content. Its cross-template drop and post-hoc diagnosis reveal prompt sensitivity.
- V3 consists of two supplied claims and constrained two-sentence targets. Source qualification is distinct from truthful content. Text and Origin have different trainable parameter counts and input-token budgets; no parameter-matched superiority claim.
- V3 cross-template: Text 95/96 complete, Origin 90/96. Origin shortens full paths, alters identifiers and invents one extra path character; an incorrect value can still be introduced with cautious wording. See evidence/origin_generation_v3/manual-review.json and original report.
- Base generations are truncated and poorly parsed; detected assertion rates are lower-bound proxy scores, not complete semantic hallucination estimates. V2 classification transfer was not a reliable natural-language solution.
- V4 scorer defects invalidate a confident net-benefit conclusion; V4.1 has no human-validated reference yet. No metric is presented as a general hallucination rate.
- CPU/MPS numerical differences and dependency drift may affect future runs. Historical dependency snapshots are observational, not a guarantee all versions are available on every platform.
