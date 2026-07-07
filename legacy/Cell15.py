# ============================================================
# CELL 15
# Aggregate fold-level and patient-level OOF performance
# ============================================================

# ------------------------------------------------------------
# 1. Fold-level summary
# ------------------------------------------------------------

PERFORMANCE_METRICS = [
    "roc_auc",
    "average_precision",
    "balanced_accuracy",
    "accuracy",
    "precision",
    "recall_sensitivity",
    "specificity",
    "f1_score",
    "positive_predictive_value",
    "negative_predictive_value",
    "brier_score",
    "log_loss"
]

fold_summary_records = []

for model_name, model_df in fold_performance.groupby(
    "model_name"
):

    for metric in PERFORMANCE_METRICS:

        values = (
            model_df[metric]
            .dropna()
            .astype(float)
        )

        fold_summary_records.append({
            "model_name": model_name,
            "metric": metric,
            "mean": values.mean(),
            "sd": values.std(ddof=1),
            "median": values.median(),
            "minimum": values.min(),
            "maximum": values.max(),
            "q025": values.quantile(0.025),
            "q975": values.quantile(0.975),
            "n_folds": len(values)
        })

fold_performance_summary = pd.DataFrame(
    fold_summary_records
)

print("=" * 72)
print("FOLD-LEVEL PERFORMANCE SUMMARY")
print("=" * 72)

display(
    fold_performance_summary.pivot(
        index="metric",
        columns="model_name",
        values="mean"
    ).round(4)
)


# ------------------------------------------------------------
# 2. Aggregate held-out predictions by patient
# ------------------------------------------------------------

patient_oof_summary = (
    oof_predictions_all
    .groupby(
        [
            "model_name",
            "patient_id",
            "true_label",
            "true_group"
        ],
        as_index=False
    )
    .agg(
        mean_probability_advanced=(
            "predicted_probability_advanced",
            "mean"
        ),
        median_probability_advanced=(
            "predicted_probability_advanced",
            "median"
        ),
        sd_probability_advanced=(
            "predicted_probability_advanced",
            "std"
        ),
        minimum_probability_advanced=(
            "predicted_probability_advanced",
            "min"
        ),
        maximum_probability_advanced=(
            "predicted_probability_advanced",
            "max"
        ),
        fraction_predicted_advanced=(
            "predicted_label",
            "mean"
        ),
        fraction_correct=(
            "prediction_correct",
            "mean"
        ),
        n_oof_predictions=(
            "predicted_probability_advanced",
            "size"
        )
    )
)

patient_oof_summary[
    "predicted_label_at_0_5"
] = (
    patient_oof_summary[
        "mean_probability_advanced"
    ] >= 0.5
).astype(int)

patient_oof_summary[
    "predicted_group_at_0_5"
] = (
    patient_oof_summary[
        "predicted_label_at_0_5"
    ]
    .map({
        0: "Early",
        1: "Advanced"
    })
)

patient_oof_summary[
    "probability_margin_from_0_5"
] = (
    patient_oof_summary[
        "mean_probability_advanced"
    ] - 0.5
).abs()

# Higher value = closer to decision boundary
patient_oof_summary[
    "boundary_uncertainty"
] = (
    1
    - 2
    * patient_oof_summary[
        "probability_margin_from_0_5"
    ]
).clip(0, 1)

# Repeat-to-repeat variability
patient_oof_summary[
    "prediction_instability"
] = (
    patient_oof_summary[
        "sd_probability_advanced"
    ]
)


# ------------------------------------------------------------
# 3. Patient-level metrics
# ------------------------------------------------------------

patient_performance_records = []

for model_name, model_df in patient_oof_summary.groupby(
    "model_name"
):

    y_true_model = (
        model_df["true_label"]
        .to_numpy(dtype=int)
    )

    y_probability_model = (
        model_df[
            "mean_probability_advanced"
        ]
        .to_numpy(dtype=float)
    )

    metrics = calculate_binary_metrics(
        y_true=y_true_model,
        y_probability=y_probability_model,
        threshold=0.5
    )

    patient_performance_records.append({
        "model_name": model_name,
        "aggregation": (
            "mean_probability_across_repeats"
        ),
        "n_patients": len(model_df),
        **metrics
    })

patient_level_performance = pd.DataFrame(
    patient_performance_records
)

print("\n" + "=" * 72)
print("PATIENT-LEVEL REPEATED-OOF PERFORMANCE")
print("=" * 72)

display(
    patient_level_performance[
        [
            "model_name",
            "n_patients",
            "roc_auc",
            "average_precision",
            "balanced_accuracy",
            "accuracy",
            "recall_sensitivity",
            "specificity",
            "brier_score",
            "log_loss",
            "true_negative",
            "false_positive",
            "false_negative",
            "true_positive"
        ]
    ].round(4)
)


# ------------------------------------------------------------
# 4. Probability-distribution audit
# ------------------------------------------------------------

probability_distribution_summary = (
    patient_oof_summary
    .groupby(
        [
            "model_name",
            "true_group"
        ],
        as_index=False
    )
    .agg(
        n=(
            "patient_id",
            "size"
        ),
        mean_probability=(
            "mean_probability_advanced",
            "mean"
        ),
        sd_probability=(
            "mean_probability_advanced",
            "std"
        ),
        median_probability=(
            "mean_probability_advanced",
            "median"
        ),
        q025_probability=(
            "mean_probability_advanced",
            lambda values: values.quantile(0.025)
        ),
        q25_probability=(
            "mean_probability_advanced",
            lambda values: values.quantile(0.25)
        ),
        q75_probability=(
            "mean_probability_advanced",
            lambda values: values.quantile(0.75)
        ),
        q975_probability=(
            "mean_probability_advanced",
            lambda values: values.quantile(0.975)
        ),
        mean_repeat_sd=(
            "sd_probability_advanced",
            "mean"
        )
    )
)

print("\nProbability distribution by true class:")

display(
    probability_distribution_summary.round(4)
)


# ------------------------------------------------------------
# 5. Prediction-state audit
# ------------------------------------------------------------

patient_oof_summary[
    "prediction_state_0_5"
] = np.select(
    [
        (
            patient_oof_summary["true_label"] == 0
        )
        & (
            patient_oof_summary[
                "predicted_label_at_0_5"
            ] == 0
        ),

        (
            patient_oof_summary["true_label"] == 0
        )
        & (
            patient_oof_summary[
                "predicted_label_at_0_5"
            ] == 1
        ),

        (
            patient_oof_summary["true_label"] == 1
        )
        & (
            patient_oof_summary[
                "predicted_label_at_0_5"
            ] == 0
        ),

        (
            patient_oof_summary["true_label"] == 1
        )
        & (
            patient_oof_summary[
                "predicted_label_at_0_5"
            ] == 1
        )
    ],
    [
        "early_concordant",
        "early_predicted_advanced",
        "advanced_predicted_early",
        "advanced_concordant"
    ],
    default="unresolved"
)

prediction_state_counts = (
    patient_oof_summary
    .groupby(
        [
            "model_name",
            "prediction_state_0_5"
        ],
        as_index=False
    )
    .size()
    .rename(columns={"size": "n"})
)

print("\nPrediction-state counts:")

display(
    prediction_state_counts
)


# ------------------------------------------------------------
# 6. Save results
# ------------------------------------------------------------

fold_performance_summary.to_csv(
    DIRS["blackbox"]
    / "fold_performance_summary.tsv",
    sep="\t",
    index=False
)

patient_oof_summary.to_csv(
    DIRS["blackbox"]
    / "patient_oof_prediction_summary.tsv",
    sep="\t",
    index=False
)

patient_level_performance.to_csv(
    DIRS["blackbox"]
    / "patient_level_performance.tsv",
    sep="\t",
    index=False
)

probability_distribution_summary.to_csv(
    DIRS["blackbox"]
    / "probability_distribution_by_true_class.tsv",
    sep="\t",
    index=False
)

prediction_state_counts.to_csv(
    DIRS["blackbox"]
    / "prediction_state_counts_at_0_5.tsv",
    sep="\t",
    index=False
)

logger.info(
    "Fold-level and patient-level OOF aggregation completed."
)

print("\n" + "=" * 72)
print("CELL 15 COMPLETED")
print("=" * 72)