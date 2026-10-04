# AIDO-BBA scientific repair contract — 2026-10-04

This document records the manuscript-facing scientific definitions adopted for the v1.1.0 repair line after forensic review of the manuscript, source code, and preserved AIDO-Audit outputs. Historical v1.0.1 artifacts remain provenance evidence and are not rewritten retroactively.

## 1. Cross-model disagreement

Raw absolute probability separation is retained only as a descriptive, scale-dependent diagnostic. The primary cross-model disagreement quantity is

`D_rank_i = |rankpct_EN_i - rankpct_ET_i|`

where each percentile is computed within the corresponding model's patient-level out-of-fold score distribution. This quantity is invariant to monotone probability rescaling. Q75/Q90 cut-points may be used only as secondary descriptive strata; no hard patient state is defined from them.

The historical 108-patient `model-dependent` group and the corresponding `model arbitration` action are superseded as primary manuscript results.

## 2. Attribution-mass coverage C_i

The canonical completeness quantity is the gene-level absolute attribution-mass fraction

`C_i = sum(mapped_g * |phi_ig|) / sum(|phi_ig|)`.

It is not the absolute mass of a reconstructed process-level sum. The primary TCGA-BRCA mean remains 0.8323 because the source implementation already used this canonical definition.

## 3. Fuzzy explanatory membership

The three per-axis values are independent fuzzy magnitude memberships derived from the repeat-supported residual axes. They are not simplex proportions and are not constrained to sum to one.

Source-verified state precedence is:

1. at least two memberships >= 0.75 -> `shared_high`;
2. exactly one membership >= 0.75 and top-minus-second >= 0.15 -> corresponding dominant core;
3. all memberships <= 0.25 -> `shared_low`;
4. otherwise -> `blended_intermediate`.

The observed state counts 633 blended, 136/142/98 dominant, and 64 shared-high therefore remain numerically valid after correcting the manuscript definition.

## 4. Integrated patient taxonomy

The historical mutually exclusive six-state `integrated_bba_state` is retired from the repaired primary manuscript. The repaired presentation is modular and non-collapsing: cross-model rank disagreement, repeat instability, representation deficit, signed residual magnitude, fuzzy allocation uncertainty, and specificity remain separate audit dimensions.

Historical integrated-state fields may remain in preserved files for provenance but must not be presented as a repaired primary patient taxonomy.

## 5. Measurement-triage rules

The preserved AIDO-Audit patient-level report recovered the exact source lineage for the surviving action rules:

- representation expansion: `excess_coverage_mean < -0.005`;
- residual-direction review: `abs(excess_signed_residual_mean) > 0.005`;
- repeat or orthogonal measurement: `repeat_state_consistency < 0.60 OR repeat_instability_tier == high_repeat_instability`;
- label-model discrepancy review: preserved label/model discordance flag, renamed from `clinical-molecular reconciliation`.

Primary TCGA-BRCA counts are 519, 538, 302, and 164 respectively. Categories are overlapping computational audit flags and are not clinical recommendations.

The historical `model arbitration = 108` action is retired because its trigger depended on calibration-confounded raw probability separation.

## 6. Repaired statistical families

Invalid groupings based on `integrated_bba_state` and raw-probability `model_dependence_tier` were removed. Valid retained grouping variables are `true_group`, `repeat_instability_tier`, and the renamed stage-label/model-score rank geometry.

Repaired counts are:

- representation-gap omnibus: 18 tested / 14 FDR-significant;
- representation-gap pairwise: 84 tested / 44 FDR-significant; 16 medium/large significant effects;
- anchor/boundary omnibus: 36 tested / 23 FDR-significant;
- anchor/boundary pairwise: 168 tested / 95 FDR-significant; 47 medium/large significant effects.

Omnibus Benjamini-Hochberg correction is recomputed over the repaired valid hypothesis family. Pairwise FDR remains within each grouping-variable x metric family.

## 7. Interpretation ceiling

AIDO-BBA is an auditable computational/software framework for separating predictive disagreement, resampling instability, representation completeness, residual structure, fuzzy allocation, and output specificity.

The repaired manuscript does not claim clinical decision support, treatment recommendation, validated assay guidance, molecular subtype discovery, calibrated disease-state probabilities, or demonstrated clinical utility.

## 8. Release governance

Historical release v1.0.1 is preserved as superseded evidence. The repair line targets v1.1.0. No v1.1.0 tag, GitHub release, or archival DOI should be described as final until release conformance and archival deposition are explicitly verified.

Executable reference definitions are in `aido_bba/audit_contract.py`; regression tests are in `tests/test_audit_contract.py`.
