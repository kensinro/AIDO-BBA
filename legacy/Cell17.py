# ============================================================
# CELL 17
# Construct the AIDO-BBA patient-state taxonomy
# ============================================================

# ------------------------------------------------------------
# Important interpretation rule
# ------------------------------------------------------------
#
# The thresholds selected from the same OOF cohort are used only
# for descriptive audit-state construction.
#
# They are NOT externally validated clinical cut-offs.
# ------------------------------------------------------------


# ------------------------------------------------------------
# 1. Retrieve descriptive operating thresholds
# ------------------------------------------------------------

balanced_threshold_table = (
    optimal_thresholds[
        optimal_thresholds["criterion"]
        == "maximum_balanced_accuracy"
    ]
    [
        [
            "model_name",
            "threshold",
            "balanced_accuracy",
            "sensitivity",
            "specificity"
        ]
    ]
    .copy()
)

balanced_threshold_lookup = dict(
    zip(
        balanced_threshold_table["model_name"],
        balanced_threshold_table["threshold"]
    )
)

ELASTICNET_AUDIT_THRESHOLD = float(
    balanced_threshold_lookup[
        "ElasticNet_Logistic"
    ]
)

EXTRATREES_AUDIT_THRESHOLD = float(
    balanced_threshold_lookup[
        "ExtraTrees_BlackBox"
    ]
)

print("=" * 72)
print("DESCRIPTIVE AUDIT THRESHOLDS")
print("=" * 72)

display(
    balanced_threshold_table.round(4)
)

print(
    "\nElasticNet audit threshold:",
    round(ELASTICNET_AUDIT_THRESHOLD, 4)
)

print(
    "ExtraTrees audit threshold:",
    round(EXTRATREES_AUDIT_THRESHOLD, 4)
)


# ------------------------------------------------------------
# 2. Start from the cross-model patient table
# ------------------------------------------------------------

bba_patient_states = (
    model_disagreement
    .copy()
)


# ------------------------------------------------------------
# 3. Cohort-relative probability ranks
# ------------------------------------------------------------

bba_patient_states[
    "elasticnet_probability_percentile"
] = (
    bba_patient_states[
        "elasticnet_probability_advanced"
    ]
    .rank(
        method="average",
        pct=True
    )
)

bba_patient_states[
    "extratrees_probability_percentile"
] = (
    bba_patient_states[
        "extratrees_probability_advanced"
    ]
    .rank(
        method="average",
        pct=True
    )
)

bba_patient_states[
    "mean_cross_model_percentile"
] = (
    bba_patient_states[
        [
            "elasticnet_probability_percentile",
            "extratrees_probability_percentile"
        ]
    ]
    .mean(axis=1)
)


# ------------------------------------------------------------
# 4. Rank-based molecular state
# ------------------------------------------------------------
#
# Lower 25%:
#     early-like molecular rank
#
# Middle 50%:
#     intermediate molecular rank
#
# Upper 25%:
#     advanced-like molecular rank
#
# These are cohort-relative states, not biological subtypes.
# ------------------------------------------------------------

bba_patient_states[
    "extratrees_rank_state"
] = pd.cut(
    bba_patient_states[
        "extratrees_probability_percentile"
    ],
    bins=[
        0,
        0.25,
        0.75,
        1.0
    ],
    labels=[
        "lower_rank_early_like",
        "intermediate_rank",
        "upper_rank_advanced_like"
    ],
    include_lowest=True
)

bba_patient_states[
    "cross_model_consensus_rank_state"
] = pd.cut(
    bba_patient_states[
        "mean_cross_model_percentile"
    ],
    bins=[
        0,
        0.25,
        0.75,
        1.0
    ],
    labels=[
        "lower_rank_early_like",
        "intermediate_rank",
        "upper_rank_advanced_like"
    ],
    include_lowest=True
)


# ------------------------------------------------------------
# 5. Threshold-based descriptive states
# ------------------------------------------------------------

bba_patient_states[
    "elasticnet_label_audit_threshold"
] = (
    bba_patient_states[
        "elasticnet_probability_advanced"
    ]
    >= ELASTICNET_AUDIT_THRESHOLD
).astype(int)

bba_patient_states[
    "extratrees_label_audit_threshold"
] = (
    bba_patient_states[
        "extratrees_probability_advanced"
    ]
    >= EXTRATREES_AUDIT_THRESHOLD
).astype(int)

bba_patient_states[
    "audit_threshold_label_agreement"
] = (
    bba_patient_states[
        "elasticnet_label_audit_threshold"
    ]
    ==
    bba_patient_states[
        "extratrees_label_audit_threshold"
    ]
)


# ------------------------------------------------------------
# 6. Cross-model probability direction
# ------------------------------------------------------------

bba_patient_states[
    "cross_model_probability_direction"
] = np.select(
    [
        bba_patient_states[
            "probability_difference_elasticnet_minus_extratrees"
        ] > 0.05,

        bba_patient_states[
            "probability_difference_elasticnet_minus_extratrees"
        ] < -0.05
    ],
    [
        "elasticnet_higher",
        "extratrees_higher"
    ],
    default="similar_probability"
)


# ------------------------------------------------------------
# 7. Data-driven disagreement tiers
# ------------------------------------------------------------

probability_difference_q75 = float(
    bba_patient_states[
        "absolute_probability_difference"
    ].quantile(0.75)
)

probability_difference_q90 = float(
    bba_patient_states[
        "absolute_probability_difference"
    ].quantile(0.90)
)

bba_patient_states[
    "model_dependence_tier"
] = np.select(
    [
        bba_patient_states[
            "absolute_probability_difference"
        ] >= probability_difference_q90,

        bba_patient_states[
            "absolute_probability_difference"
        ] >= probability_difference_q75
    ],
    [
        "high_model_dependence",
        "moderate_model_dependence"
    ],
    default="low_model_dependence"
)


# ------------------------------------------------------------
# 8. Repeat-instability tiers
# ------------------------------------------------------------

combined_instability = (
    bba_patient_states[
        [
            "elasticnet_prediction_instability",
            "extratrees_prediction_instability"
        ]
    ]
    .mean(axis=1)
)

bba_patient_states[
    "mean_repeat_instability"
] = combined_instability

instability_q75 = float(
    combined_instability.quantile(0.75)
)

instability_q90 = float(
    combined_instability.quantile(0.90)
)

bba_patient_states[
    "repeat_instability_tier"
] = np.select(
    [
        combined_instability >= instability_q90,
        combined_instability >= instability_q75
    ],
    [
        "high_repeat_instability",
        "moderate_repeat_instability"
    ],
    default="low_repeat_instability"
)


# ------------------------------------------------------------
# 9. Clinical–molecular rank discordance
# ------------------------------------------------------------

bba_patient_states[
    "clinical_molecular_rank_state"
] = np.select(
    [
        (
            bba_patient_states["true_label"] == 0
        )
        & (
            bba_patient_states[
                "cross_model_consensus_rank_state"
            ]
            == "lower_rank_early_like"
        ),

        (
            bba_patient_states["true_label"] == 1
        )
        & (
            bba_patient_states[
                "cross_model_consensus_rank_state"
            ]
            == "upper_rank_advanced_like"
        ),

        (
            bba_patient_states["true_label"] == 0
        )
        & (
            bba_patient_states[
                "cross_model_consensus_rank_state"
            ]
            == "upper_rank_advanced_like"
        ),

        (
            bba_patient_states["true_label"] == 1
        )
        & (
            bba_patient_states[
                "cross_model_consensus_rank_state"
            ]
            == "lower_rank_early_like"
        )
    ],
    [
        "early_rank_concordant",
        "advanced_rank_concordant",
        "clinical_early_molecular_advanced_like",
        "clinical_advanced_molecular_early_like"
    ],
    default="intermediate_or_ambiguous"
)


# ------------------------------------------------------------
# 10. Integrated AIDO-BBA audit state
# ------------------------------------------------------------

bba_patient_states[
    "integrated_bba_state"
] = np.select(
    [
        (
            bba_patient_states[
                "model_dependence_tier"
            ] == "high_model_dependence"
        ),

        (
            bba_patient_states[
                "repeat_instability_tier"
            ] == "high_repeat_instability"
        ),

        (
            bba_patient_states[
                "clinical_molecular_rank_state"
            ]
            == "clinical_early_molecular_advanced_like"
        ),

        (
            bba_patient_states[
                "clinical_molecular_rank_state"
            ]
            == "clinical_advanced_molecular_early_like"
        ),

        (
            bba_patient_states[
                "clinical_molecular_rank_state"
            ].isin([
                "early_rank_concordant",
                "advanced_rank_concordant"
            ])
        )
    ],
    [
        "model_dependent",
        "resampling_unstable",
        "clinical_early_molecular_advanced_like",
        "clinical_advanced_molecular_early_like",
        "clinical_molecular_concordant"
    ],
    default="intermediate_ambiguous"
)


# ------------------------------------------------------------
# 11. Evidence flags
# ------------------------------------------------------------

bba_patient_states[
    "flag_high_model_dependence"
] = (
    bba_patient_states[
        "model_dependence_tier"
    ] == "high_model_dependence"
)

bba_patient_states[
    "flag_high_repeat_instability"
] = (
    bba_patient_states[
        "repeat_instability_tier"
    ] == "high_repeat_instability"
)

bba_patient_states[
    "flag_clinical_molecular_discordance"
] = (
    bba_patient_states[
        "clinical_molecular_rank_state"
    ].isin([
        "clinical_early_molecular_advanced_like",
        "clinical_advanced_molecular_early_like"
    ])
)

bba_patient_states[
    "n_audit_flags"
] = (
    bba_patient_states[
        [
            "flag_high_model_dependence",
            "flag_high_repeat_instability",
            "flag_clinical_molecular_discordance"
        ]
    ]
    .sum(axis=1)
)


# ------------------------------------------------------------
# 12. State summaries
# ------------------------------------------------------------

integrated_state_counts = (
    bba_patient_states
    .groupby(
        [
            "true_group",
            "integrated_bba_state"
        ],
        as_index=False
    )
    .size()
    .rename(columns={"size": "n"})
)

clinical_molecular_state_counts = (
    bba_patient_states
    .groupby(
        [
            "true_group",
            "clinical_molecular_rank_state"
        ],
        as_index=False
    )
    .size()
    .rename(columns={"size": "n"})
)

model_dependence_counts = (
    bba_patient_states[
        "model_dependence_tier"
    ]
    .value_counts()
    .rename_axis("model_dependence_tier")
    .reset_index(name="n")
)

repeat_instability_counts = (
    bba_patient_states[
        "repeat_instability_tier"
    ]
    .value_counts()
    .rename_axis("repeat_instability_tier")
    .reset_index(name="n")
)


print("=" * 72)
print("AIDO-BBA PATIENT-STATE TAXONOMY")
print("=" * 72)

print("\nIntegrated BBA states:")
display(
    integrated_state_counts
)

print("\nClinical–molecular rank states:")
display(
    clinical_molecular_state_counts
)

print("\nModel-dependence tiers:")
display(
    model_dependence_counts
)

print("\nRepeat-instability tiers:")
display(
    repeat_instability_counts
)

print("\nAudit thresholds and quantiles:")

audit_state_thresholds = pd.DataFrame([
    {
        "parameter":
            "elasticnet_balanced_accuracy_threshold",
        "value":
            ELASTICNET_AUDIT_THRESHOLD
    },
    {
        "parameter":
            "extratrees_balanced_accuracy_threshold",
        "value":
            EXTRATREES_AUDIT_THRESHOLD
    },
    {
        "parameter":
            "absolute_probability_difference_q75",
        "value":
            probability_difference_q75
    },
    {
        "parameter":
            "absolute_probability_difference_q90",
        "value":
            probability_difference_q90
    },
    {
        "parameter":
            "mean_repeat_instability_q75",
        "value":
            instability_q75
    },
    {
        "parameter":
            "mean_repeat_instability_q90",
        "value":
            instability_q90
    }
])

display(
    audit_state_thresholds.round(4)
)


# ------------------------------------------------------------
# 13. Priority audit patients
# ------------------------------------------------------------

priority_audit_patients = (
    bba_patient_states
    .sort_values(
        [
            "n_audit_flags",
            "absolute_probability_difference",
            "mean_repeat_instability"
        ],
        ascending=[
            False,
            False,
            False
        ]
    )
)

print("\nPriority audit patients:")

display(
    priority_audit_patients[
        [
            "patient_id",
            "true_group",
            "extratrees_probability_advanced",
            "elasticnet_probability_advanced",
            "absolute_probability_difference",
            "mean_repeat_instability",
            "clinical_molecular_rank_state",
            "model_dependence_tier",
            "repeat_instability_tier",
            "integrated_bba_state",
            "n_audit_flags"
        ]
    ].head(30)
)


# ------------------------------------------------------------
# 14. Save outputs
# ------------------------------------------------------------

bba_patient_states.to_csv(
    DIRS["blackbox"]
    / "bba_patient_state_taxonomy.tsv",
    sep="\t",
    index=False
)

integrated_state_counts.to_csv(
    DIRS["blackbox"]
    / "integrated_bba_state_counts.tsv",
    sep="\t",
    index=False
)

clinical_molecular_state_counts.to_csv(
    DIRS["blackbox"]
    / "clinical_molecular_rank_state_counts.tsv",
    sep="\t",
    index=False
)

model_dependence_counts.to_csv(
    DIRS["blackbox"]
    / "model_dependence_tier_counts.tsv",
    sep="\t",
    index=False
)

repeat_instability_counts.to_csv(
    DIRS["blackbox"]
    / "repeat_instability_tier_counts.tsv",
    sep="\t",
    index=False
)

audit_state_thresholds.to_csv(
    DIRS["blackbox"]
    / "bba_state_thresholds.tsv",
    sep="\t",
    index=False
)

priority_audit_patients.head(100).to_csv(
    DIRS["blackbox"]
    / "priority_audit_patients_top100.tsv",
    sep="\t",
    index=False
)

logger.info(
    "AIDO-BBA patient-state taxonomy completed."
)

print("\n" + "=" * 72)
print("CELL 17 COMPLETED")
print("=" * 72)