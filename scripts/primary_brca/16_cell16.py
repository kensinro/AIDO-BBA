from aido_bba.config import (
    data_root, brca_output_root, brca_run_dir, go_bp_gmt,
    hgnc_file, ncbi_gene_info, metabric_dir, gse96058_dir,
    kirc_dir, kirc_output_root,
)

# ============================================================
# CELL 16
# Threshold, calibration, and model-disagreement audit
# ============================================================

from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve
)

# ------------------------------------------------------------
# 1. Threshold grid
# ------------------------------------------------------------

THRESHOLD_GRID = np.round(
    np.arange(
        0.05,
        0.951,
        0.01
    ),
    2
)

threshold_audit_records = []


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

    for threshold in THRESHOLD_GRID:

        metrics = calculate_binary_metrics(
            y_true=y_true_model,
            y_probability=y_probability_model,
            threshold=float(threshold)
        )

        youden_j = (
            metrics["recall_sensitivity"]
            + metrics["specificity"]
            - 1
        )

        threshold_audit_records.append({
            "model_name": model_name,
            "threshold": float(threshold),

            "roc_auc": metrics["roc_auc"],
            "average_precision": (
                metrics["average_precision"]
            ),

            "balanced_accuracy": (
                metrics["balanced_accuracy"]
            ),

            "accuracy": metrics["accuracy"],
            "sensitivity": (
                metrics["recall_sensitivity"]
            ),

            "specificity": metrics["specificity"],
            "precision": metrics["precision"],
            "f1_score": metrics["f1_score"],

            "positive_predictive_value": (
                metrics[
                    "positive_predictive_value"
                ]
            ),

            "negative_predictive_value": (
                metrics[
                    "negative_predictive_value"
                ]
            ),

            "youden_j": youden_j,

            "true_negative": (
                metrics["true_negative"]
            ),

            "false_positive": (
                metrics["false_positive"]
            ),

            "false_negative": (
                metrics["false_negative"]
            ),

            "true_positive": (
                metrics["true_positive"]
            )
        })


threshold_audit = pd.DataFrame(
    threshold_audit_records
)


# ------------------------------------------------------------
# 2. Identify candidate operating thresholds
# ------------------------------------------------------------

optimal_threshold_records = []

for model_name, model_df in threshold_audit.groupby(
    "model_name"
):

    model_df = model_df.copy()

    # Highest balanced accuracy
    best_balanced = model_df.loc[
        model_df["balanced_accuracy"].idxmax()
    ]

    # Highest Youden J
    best_youden = model_df.loc[
        model_df["youden_j"].idxmax()
    ]

    # Highest F1
    best_f1 = model_df.loc[
        model_df["f1_score"].idxmax()
    ]

    for criterion, row in [
        (
            "maximum_balanced_accuracy",
            best_balanced
        ),
        (
            "maximum_youden_j",
            best_youden
        ),
        (
            "maximum_f1",
            best_f1
        )
    ]:

        optimal_threshold_records.append({
            "model_name": model_name,
            "criterion": criterion,
            "threshold": row["threshold"],
            "balanced_accuracy": (
                row["balanced_accuracy"]
            ),
            "sensitivity": row["sensitivity"],
            "specificity": row["specificity"],
            "precision": row["precision"],
            "f1_score": row["f1_score"],
            "youden_j": row["youden_j"],
            "true_negative": row["true_negative"],
            "false_positive": row["false_positive"],
            "false_negative": row["false_negative"],
            "true_positive": row["true_positive"]
        })


optimal_thresholds = pd.DataFrame(
    optimal_threshold_records
)

print("=" * 72)
print("THRESHOLD AUDIT")
print("=" * 72)

print("\nCandidate operating thresholds:")

display(
    optimal_thresholds.round(4)
)


# ------------------------------------------------------------
# 3. Threshold-free ROC operating-point audit
# ------------------------------------------------------------

roc_operating_records = []

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

    false_positive_rate, true_positive_rate, thresholds = (
        roc_curve(
            y_true_model,
            y_probability_model
        )
    )

    specificity = 1 - false_positive_rate
    youden_j = (
        true_positive_rate
        + specificity
        - 1
    )

    for index in range(len(thresholds)):

        roc_operating_records.append({
            "model_name": model_name,
            "threshold": float(
                thresholds[index]
            ),
            "false_positive_rate": float(
                false_positive_rate[index]
            ),
            "specificity": float(
                specificity[index]
            ),
            "sensitivity": float(
                true_positive_rate[index]
            ),
            "youden_j": float(
                youden_j[index]
            )
        })


roc_operating_points = pd.DataFrame(
    roc_operating_records
)


# ------------------------------------------------------------
# 4. Calibration curves
# ------------------------------------------------------------

N_CALIBRATION_BINS = 10

calibration_records = []

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

    fraction_positive, mean_predicted_probability = (
        calibration_curve(
            y_true_model,
            y_probability_model,
            n_bins=N_CALIBRATION_BINS,
            strategy="quantile"
        )
    )

    for bin_index, (
        observed_fraction,
        predicted_mean
    ) in enumerate(
        zip(
            fraction_positive,
            mean_predicted_probability
        ),
        start=1
    ):

        calibration_records.append({
            "model_name": model_name,
            "bin_id": bin_index,
            "mean_predicted_probability": float(
                predicted_mean
            ),
            "observed_advanced_fraction": float(
                observed_fraction
            ),
            "calibration_residual": float(
                observed_fraction
                - predicted_mean
            ),
            "absolute_calibration_error": float(
                abs(
                    observed_fraction
                    - predicted_mean
                )
            )
        })


calibration_audit = pd.DataFrame(
    calibration_records
)


# ------------------------------------------------------------
# 5. Calibration summary
# ------------------------------------------------------------

calibration_summary = (
    calibration_audit
    .groupby(
        "model_name",
        as_index=False
    )
    .agg(
        mean_absolute_calibration_error=(
            "absolute_calibration_error",
            "mean"
        ),
        maximum_absolute_calibration_error=(
            "absolute_calibration_error",
            "max"
        ),
        mean_signed_calibration_residual=(
            "calibration_residual",
            "mean"
        )
    )
)

print("\nCalibration summary:")

display(
    calibration_summary.round(4)
)

print("\nCalibration bins:")

display(
    calibration_audit.round(4)
)


# ------------------------------------------------------------
# 6. Construct model-comparison patient table
# ------------------------------------------------------------

elasticnet_patient = (
    patient_oof_summary[
        patient_oof_summary[
            "model_name"
        ] == "ElasticNet_Logistic"
    ]
    [
        [
            "patient_id",
            "true_label",
            "true_group",
            "mean_probability_advanced",
            "sd_probability_advanced",
            "fraction_predicted_advanced",
            "fraction_correct",
            "predicted_label_at_0_5",
            "prediction_state_0_5",
            "boundary_uncertainty",
            "prediction_instability"
        ]
    ]
    .rename(
        columns={
            "mean_probability_advanced":
                "elasticnet_probability_advanced",

            "sd_probability_advanced":
                "elasticnet_probability_sd",

            "fraction_predicted_advanced":
                "elasticnet_fraction_predicted_advanced",

            "fraction_correct":
                "elasticnet_fraction_correct",

            "predicted_label_at_0_5":
                "elasticnet_label_0_5",

            "prediction_state_0_5":
                "elasticnet_state_0_5",

            "boundary_uncertainty":
                "elasticnet_boundary_uncertainty",

            "prediction_instability":
                "elasticnet_prediction_instability"
        }
    )
)


extratrees_patient = (
    patient_oof_summary[
        patient_oof_summary[
            "model_name"
        ] == "ExtraTrees_BlackBox"
    ]
    [
        [
            "patient_id",
            "true_label",
            "true_group",
            "mean_probability_advanced",
            "sd_probability_advanced",
            "fraction_predicted_advanced",
            "fraction_correct",
            "predicted_label_at_0_5",
            "prediction_state_0_5",
            "boundary_uncertainty",
            "prediction_instability"
        ]
    ]
    .rename(
        columns={
            "mean_probability_advanced":
                "extratrees_probability_advanced",

            "sd_probability_advanced":
                "extratrees_probability_sd",

            "fraction_predicted_advanced":
                "extratrees_fraction_predicted_advanced",

            "fraction_correct":
                "extratrees_fraction_correct",

            "predicted_label_at_0_5":
                "extratrees_label_0_5",

            "prediction_state_0_5":
                "extratrees_state_0_5",

            "boundary_uncertainty":
                "extratrees_boundary_uncertainty",

            "prediction_instability":
                "extratrees_prediction_instability"
        }
    )
)


model_disagreement = elasticnet_patient.merge(
    extratrees_patient,
    on=[
        "patient_id",
        "true_label",
        "true_group"
    ],
    how="inner",
    validate="one_to_one"
)


# ------------------------------------------------------------
# 7. Model-disagreement metrics
# ------------------------------------------------------------

model_disagreement[
    "probability_difference_elasticnet_minus_extratrees"
] = (
    model_disagreement[
        "elasticnet_probability_advanced"
    ]
    -
    model_disagreement[
        "extratrees_probability_advanced"
    ]
)

model_disagreement[
    "absolute_probability_difference"
] = (
    model_disagreement[
        "probability_difference_elasticnet_minus_extratrees"
    ].abs()
)

model_disagreement[
    "label_disagreement_at_0_5"
] = (
    model_disagreement[
        "elasticnet_label_0_5"
    ]
    !=
    model_disagreement[
        "extratrees_label_0_5"
    ]
)

model_disagreement[
    "both_correct_at_0_5"
] = (
    (
        model_disagreement[
            "elasticnet_label_0_5"
        ]
        ==
        model_disagreement["true_label"]
    )
    &
    (
        model_disagreement[
            "extratrees_label_0_5"
        ]
        ==
        model_disagreement["true_label"]
    )
)

model_disagreement[
    "both_wrong_at_0_5"
] = (
    (
        model_disagreement[
            "elasticnet_label_0_5"
        ]
        !=
        model_disagreement["true_label"]
    )
    &
    (
        model_disagreement[
            "extratrees_label_0_5"
        ]
        !=
        model_disagreement["true_label"]
    )
)

model_disagreement[
    "elasticnet_only_correct_at_0_5"
] = (
    (
        model_disagreement[
            "elasticnet_label_0_5"
        ]
        ==
        model_disagreement["true_label"]
    )
    &
    (
        model_disagreement[
            "extratrees_label_0_5"
        ]
        !=
        model_disagreement["true_label"]
    )
)

model_disagreement[
    "extratrees_only_correct_at_0_5"
] = (
    (
        model_disagreement[
            "elasticnet_label_0_5"
        ]
        !=
        model_disagreement["true_label"]
    )
    &
    (
        model_disagreement[
            "extratrees_label_0_5"
        ]
        ==
        model_disagreement["true_label"]
    )
)


# ------------------------------------------------------------
# 8. Patient-level cross-model audit state
# ------------------------------------------------------------

model_disagreement[
    "cross_model_state"
] = np.select(
    [
        model_disagreement[
            "both_correct_at_0_5"
        ],

        model_disagreement[
            "both_wrong_at_0_5"
        ],

        model_disagreement[
            "elasticnet_only_correct_at_0_5"
        ],

        model_disagreement[
            "extratrees_only_correct_at_0_5"
        ]
    ],
    [
        "both_correct",
        "both_wrong",
        "elasticnet_only_correct",
        "extratrees_only_correct"
    ],
    default="unresolved"
)


cross_model_state_counts = (
    model_disagreement
    .groupby(
        [
            "true_group",
            "cross_model_state"
        ],
        as_index=False
    )
    .size()
    .rename(columns={"size": "n"})
)

print("\nCross-model state counts:")

display(
    cross_model_state_counts
)


# ------------------------------------------------------------
# 9. Disagreement summary
# ------------------------------------------------------------

model_disagreement_summary = pd.DataFrame([
    {
        "metric": "n_patients",
        "value": len(model_disagreement)
    },
    {
        "metric": "label_disagreement_at_0_5",
        "value": int(
            model_disagreement[
                "label_disagreement_at_0_5"
            ].sum()
        )
    },
    {
        "metric": "both_correct",
        "value": int(
            model_disagreement[
                "both_correct_at_0_5"
            ].sum()
        )
    },
    {
        "metric": "both_wrong",
        "value": int(
            model_disagreement[
                "both_wrong_at_0_5"
            ].sum()
        )
    },
    {
        "metric": "elasticnet_only_correct",
        "value": int(
            model_disagreement[
                "elasticnet_only_correct_at_0_5"
            ].sum()
        )
    },
    {
        "metric": "extratrees_only_correct",
        "value": int(
            model_disagreement[
                "extratrees_only_correct_at_0_5"
            ].sum()
        )
    },
    {
        "metric": "mean_absolute_probability_difference",
        "value": float(
            model_disagreement[
                "absolute_probability_difference"
            ].mean()
        )
    },
    {
        "metric": "median_absolute_probability_difference",
        "value": float(
            model_disagreement[
                "absolute_probability_difference"
            ].median()
        )
    }
])

print("\nModel-disagreement summary:")

display(
    model_disagreement_summary
)


# ------------------------------------------------------------
# 10. Most discordant patients
# ------------------------------------------------------------

most_discordant_patients = (
    model_disagreement
    .sort_values(
        "absolute_probability_difference",
        ascending=False
    )
    .head(50)
)

print("\nMost discordant patients:")

display(
    most_discordant_patients[
        [
            "patient_id",
            "true_group",
            "elasticnet_probability_advanced",
            "extratrees_probability_advanced",
            "absolute_probability_difference",
            "elasticnet_state_0_5",
            "extratrees_state_0_5",
            "cross_model_state"
        ]
    ].head(20)
)


# ------------------------------------------------------------
# 11. Save outputs
# ------------------------------------------------------------

threshold_audit.to_csv(
    DIRS["blackbox"]
    / "threshold_sweep.tsv",
    sep="\t",
    index=False
)

optimal_thresholds.to_csv(
    DIRS["blackbox"]
    / "candidate_operating_thresholds.tsv",
    sep="\t",
    index=False
)

roc_operating_points.to_csv(
    DIRS["blackbox"]
    / "roc_operating_points.tsv",
    sep="\t",
    index=False
)

calibration_audit.to_csv(
    DIRS["blackbox"]
    / "calibration_bins.tsv",
    sep="\t",
    index=False
)

calibration_summary.to_csv(
    DIRS["blackbox"]
    / "calibration_summary.tsv",
    sep="\t",
    index=False
)

model_disagreement.to_csv(
    DIRS["blackbox"]
    / "patient_cross_model_disagreement.tsv",
    sep="\t",
    index=False
)

cross_model_state_counts.to_csv(
    DIRS["blackbox"]
    / "cross_model_state_counts.tsv",
    sep="\t",
    index=False
)

model_disagreement_summary.to_csv(
    DIRS["blackbox"]
    / "model_disagreement_summary.tsv",
    sep="\t",
    index=False
)

most_discordant_patients.to_csv(
    DIRS["blackbox"]
    / "most_discordant_patients_top50.tsv",
    sep="\t",
    index=False
)

logger.info(
    "Threshold, calibration, and model-disagreement audit completed."
)

print("\n" + "=" * 72)
print("CELL 16 COMPLETED")
print("=" * 72)