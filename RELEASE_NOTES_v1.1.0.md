# AIDO-BBA v1.1.0 — release notes

## Release purpose

AIDO-BBA v1.1.0 is the repaired scientific-contract release accompanying the manuscript:

**A modular computational audit framework for high-dimensional transcriptomic classifiers: patient-level disagreement, explanatory completeness, and representation gaps**

This release preserves historical v1.0.1 provenance while correcting manuscript-facing definitions, terminology, and release metadata. The software remains a computational audit framework; it is not a clinical decision-support system, treatment-recommendation engine, validated diagnostic assay, molecular-subtype classifier, or causal-mechanism discovery tool.

## Principal scientific-contract repairs

- Replaces raw cross-model probability separation as the primary disagreement quantity with calibration-insensitive percentile-rank separation, `D_rank`.
- Binds percentile ranks to the governing execution convention: model-wise `Series.rank(method="average", pct=True)` before patient-aligned absolute rank separation.
- Defines explanatory completeness as mapped selected-gene absolute attribution mass divided by total selected-gene absolute attribution mass.
- Restores the non-simplex fuzzy-membership execution contract: repeat/core median centering, magnitude percentile ranking, and patient-level averaging across repeats.
- Retires the historical mutually exclusive integrated patient-state taxonomy from repaired primary evidence.
- Retains four overlapping computational audit follow-up flags while retiring historical model arbitration from the repaired action layer.
- Binds repaired multiplicity families and finite-sample empirical-P conventions used by the manuscript analyses.
- Constrains replacement analyses to workflow portability/reconfigurability rather than frozen-model external validation.
- Harmonizes manuscript-facing terminology to audit follow-up / computational triage boundaries.

## Reproducibility and provenance

- Historical development scripts remain under `legacy/` for provenance.
- The repaired contract is documented in `docs/SCIENTIFIC_REPAIR_CONTRACT_2026-10-04.md` and encoded in `aido_bba/audit_contract.py`.
- Repository tests and the compact deterministic demonstration remain part of continuous integration.
- Large governed molecular datasets and manuscript-scale machine-output trees are not redistributed by the repository.

## Release boundary

The public release validates the software contract, code organization, metadata, tests, and demonstration workflow. It does not itself reproduce every manuscript-scale TCGA-BRCA, METABRIC, GSE96058, or TCGA-KIRC analysis without the governed source data and local execution artifacts.

## Archival note

Historical v1.0.1 remains immutable provenance. v1.1.0 should be cited as the repaired scientific-contract release only after the GitHub tag/release and corresponding archival record are created and verified.
