# Release-only changes

The V2 runner is copied without algorithm changes. V3 no longer forces offline mode; callers may set HF_HUB_OFFLINE=1 after downloading the pinned base revision. Two unconditional MPS cache cleanup calls are guarded by device type. All other training, sampling, encoding and scoring logic is retained. The implementation supports the original MPS float16 path or CPU float32; no CUDA support or identical cross-device results are claimed.

The historical V1 structural diagnostic is included to satisfy V2's existing training precondition. It is archived evidence, not a fresh diagnostic on the new user's machine. V1 diagnostic execution is not required to train the independent V2 release. Original report-generation scripts depend on the old workspace/V1 artifacts and are not entry points here. Use scripts/check.py for archived evidence and scripts/reproduce.sh for new runs; evaluation emits metrics.json and predictions.jsonl. Historical results live separately in evidence/.

No model loading, model forward pass, training or generation is part of release smoke tests. No model weights were modified. Source and released file hashes, including sanitized path transformations, are in provenance.json. Synthetic /srv/... fact values are deliberately preserved: they are test content, not private machine paths.
