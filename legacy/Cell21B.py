# ============================================================
# CELL 21B
# Resume after display-sort error
# ============================================================

print("\n" + "=" * 80)
print("STRONGEST PAIRWISE EFFECTS")
print("=" * 80)

pairwise_display = (
    pairwise_tests
    .sort_values(
        [
            "fdr_within_metric",
            "cliffs_delta_absolute"
        ],
        ascending=[
            True,
            False
        ]
    )
)

display(
    pairwise_display[
        [
            "state_variable",
            "metric",
            "group_1",
            "group_2",
            "n_group_1",
            "n_group_2",
            "mean_group_1",
            "mean_group_2",
            "mean_difference_group1_minus_group2",
            "cliffs_delta",
            "cliffs_delta_absolute",
            "cliffs_delta_magnitude",
            "p_value",
            "fdr_within_metric",
            "fdr_global"
        ]
    ].head(40)
)


print("\n" + "=" * 80)
print("PRIORITY CONTRASTS")
print("=" * 80)

if len(priority_pairwise_tests) > 0:

    display(
        priority_pairwise_tests[
            [
                "state_variable",
                "metric",
                "group_1",
                "group_2",
                "mean_group_1",
                "mean_group_2",
                "mean_difference_group1_minus_group2",
                "cliffs_delta",
                "cliffs_delta_absolute",
                "cliffs_delta_magnitude",
                "p_value",
                "fdr_within_metric",
                "fdr_global"
            ]
        ].head(50)
    )

else:

    print("No priority contrasts were available.")


print("\n" + "=" * 80)
print("TOP CANDIDATE RESIDUAL-STATE PATIENTS")
print("=" * 80)

candidate_display_columns = [
    column
    for column in [
        "patient_id",
        "true_group",
        "integrated_bba_state",
        "clinical_molecular_rank_state",
        "model_dependence_tier",
        "repeat_instability_tier",
        "mean_probability_advanced",
        "mean_attribution_mass_coverage",
        "mean_unmapped_absolute_shap",
        "mean_unmapped_signed_residual",
        "residual_repeat_sd",
        "fraction_positive_residual",
        "fraction_negative_residual",
        "residual_direction_consistency",
        "residual_direction_state",
        "residual_direction_matches_clinical_stage",
        "candidate_residual_state_score",
        "candidate_residual_state_tier"
    ]
    if column in candidate_residual_patients.columns
]

display(
    candidate_residual_patients[
        candidate_display_columns
    ].head(40)
)


# ============================================================
# SAVE OUTPUTS
# ============================================================

patient_residual_states.to_csv(
    RESIDUAL_AUDIT_DIR
    / "patient_residual_direction_states.tsv",
    sep="\t",
    index=False
)

residual_direction_counts.to_csv(
    RESIDUAL_AUDIT_DIR
    / "residual_direction_state_counts.tsv",
    sep="\t",
    index=False
)

clinical_direction_counts.to_csv(
    RESIDUAL_AUDIT_DIR
    / "clinical_residual_direction_agreement_counts.tsv",
    sep="\t",
    index=False
)

residual_direction_by_state.to_csv(
    RESIDUAL_AUDIT_DIR
    / "residual_direction_by_bba_state.tsv",
    sep="\t",
    index=False
)

state_descriptive_summary.to_csv(
    RESIDUAL_AUDIT_DIR
    / "state_metric_descriptive_summary.tsv",
    sep="\t",
    index=False
)

omnibus_tests.to_csv(
    RESIDUAL_AUDIT_DIR
    / "residual_state_omnibus_kruskal_tests.tsv",
    sep="\t",
    index=False
)

pairwise_tests.to_csv(
    RESIDUAL_AUDIT_DIR
    / "residual_state_pairwise_mannwhitney_cliffs_delta.tsv",
    sep="\t",
    index=False
)

priority_pairwise_tests.to_csv(
    RESIDUAL_AUDIT_DIR
    / "priority_state_contrasts.tsv",
    sep="\t",
    index=False
)

candidate_residual_patients.to_csv(
    RESIDUAL_AUDIT_DIR
    / "candidate_residual_state_patients.tsv",
    sep="\t",
    index=False
)

candidate_residual_patients.head(
    100
).to_csv(
    RESIDUAL_AUDIT_DIR
    / "candidate_residual_state_patients_top100.tsv",
    sep="\t",
    index=False
)


# ============================================================
# RUN SUMMARY
# ============================================================

run_summary = pd.DataFrame([
    {
        "metric": "n_patients",
        "value": len(patient_residual_states)
    },
    {
        "metric": "near_zero_residual_patients",
        "value": int(
            (
                patient_residual_states[
                    "residual_direction_state"
                ]
                == "near_zero_residual"
            ).sum()
        )
    },
    {
        "metric": "stable_toward_advanced_patients",
        "value": int(
            (
                patient_residual_states[
                    "residual_direction_state"
                ]
                == "stable_toward_advanced"
            ).sum()
        )
    },
    {
        "metric": "stable_toward_early_patients",
        "value": int(
            (
                patient_residual_states[
                    "residual_direction_state"
                ]
                == "stable_toward_early"
            ).sum()
        )
    },
    {
        "metric": "directionally_mixed_patients",
        "value": int(
            (
                patient_residual_states[
                    "residual_direction_state"
                ]
                == "directionally_mixed"
            ).sum()
        )
    },
    {
        "metric": "direction_discordant_patients",
        "value": int(
            (
                patient_residual_states[
                    "residual_direction_matches_clinical_stage"
                ]
                == "direction_discordant"
            ).sum()
        )
    },
    {
        "metric": "high_priority_candidate_patients",
        "value": int(
            (
                patient_residual_states[
                    "candidate_residual_state_tier"
                ]
                == "high_priority_candidate"
            ).sum()
        )
    },
    {
        "metric": "significant_omnibus_tests_fdr_0_05",
        "value": int(
            (
                omnibus_tests["fdr_bh"] <= 0.05
            ).sum()
        )
        if len(omnibus_tests) > 0
        else 0
    },
    {
        "metric": "significant_pairwise_tests_fdr_0_05",
        "value": int(
            (
                pairwise_tests[
                    "fdr_within_metric"
                ] <= 0.05
            ).sum()
        )
        if len(pairwise_tests) > 0
        else 0
    },
    {
        "metric": "medium_or_large_pairwise_effects",
        "value": int(
            pairwise_tests[
                "cliffs_delta_magnitude"
            ].isin([
                "medium",
                "large"
            ]).sum()
        )
        if len(pairwise_tests) > 0
        else 0
    }
])

run_summary.to_csv(
    RESIDUAL_AUDIT_DIR
    / "residual_direction_audit_summary.tsv",
    sep="\t",
    index=False
)


print("\n" + "=" * 80)
print("CELL 21 COMPLETED")
print("=" * 80)

display(run_summary)

print("\nOutput directory:")
print(RESIDUAL_AUDIT_DIR)