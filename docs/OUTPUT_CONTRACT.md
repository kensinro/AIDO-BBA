# Output contract

Each analysis writes to a timestamped run directory. The primary BRCA implementation uses ordered module folders (`00_manifest` through measurement-triage outputs). Each module writes machine-readable TSV/JSON files and, where relevant, per-patient reports.

Key principles:

1. Inputs and configuration are preserved in manifests.
2. Fold-level outputs are retained before aggregation.
3. Patient-level outputs use out-of-fold predictions or held-out attribution.
4. Identifier ambiguity and unresolved mappings are not silently discarded.
5. External stress tests write separate manifests and never overwrite the primary run.
