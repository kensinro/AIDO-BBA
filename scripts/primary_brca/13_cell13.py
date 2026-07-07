from aido_bba.config import (
    data_root, brca_output_root, brca_run_dir, go_bp_gmt,
    hgnc_file, ncbi_gene_info, metabric_dir, gse96058_dir,
    kirc_dir, kirc_output_root,
)

# ============================================================
# CELL 13
# Run repeated cross-validation for both models
# ============================================================

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
    log_loss,
    confusion_matrix
)

import time


def safe_divide(numerator, denominator):
    """
    Safe division for diagnostic metrics.
    """
    if denominator == 0:
        return np.nan

    return numerator / denominator


def calculate_binary_metrics(
    y_true,
    y_probability,
    threshold=0.5
):
    """
    Calculate binary-classification metrics.

    Positive class:
        Advanced stage = 1
    """

    y_true = np.asarray(y_true, dtype=int)

    y_probability = np.asarray(
        y_probability,
        dtype=float
    )

    y_predicted = (
        y_probability >= threshold
    ).astype(int)

    probability_clipped = np.clip(
        y_probability,
        1e-7,
        1 - 1e-7
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_predicted,
        labels=[0, 1]
    ).ravel()

    sensitivity = safe_divide(
        tp,
        tp + fn
    )

    specificity = safe_divide(
        tn,
        tn + fp
    )

    positive_predictive_value = safe_divide(
        tp,
        tp + fp
    )

    negative_predictive_value = safe_divide(
        tn,
        tn + fn
    )

    return {
        "roc_auc": (
            roc_auc_score(
                y_true,
                y_probability
            )
            if len(np.unique(y_true)) == 2
            else np.nan
        ),

        "average_precision": (
            average_precision_score(
                y_true,
                y_probability
            )
            if len(np.unique(y_true)) == 2
            else np.nan
        ),

        "balanced_accuracy": (
            balanced_accuracy_score(
                y_true,
                y_predicted
            )
        ),

        "accuracy": accuracy_score(
            y_true,
            y_predicted
        ),

        "precision": precision_score(
            y_true,
            y_predicted,
            zero_division=0
        ),

        "recall_sensitivity": recall_score(
            y_true,
            y_predicted,
            zero_division=0
        ),

        "specificity": specificity,

        "f1_score": f1_score(
            y_true,
            y_predicted,
            zero_division=0
        ),

        "positive_predictive_value": (
            positive_predictive_value
        ),

        "negative_predictive_value": (
            negative_predictive_value
        ),

        "brier_score": brier_score_loss(
            y_true,
            y_probability
        ),

        "log_loss": log_loss(
            y_true,
            np.column_stack([
                1 - probability_clipped,
                probability_clipped
            ]),
            labels=[0, 1]
        ),

        "threshold": threshold,

        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp)
    }


# ------------------------------------------------------------
# Output collectors
# ------------------------------------------------------------

fold_performance_records = []
prediction_records = []
selected_gene_records = []
model_feature_records = []
fit_timing_records = []


# ------------------------------------------------------------
# Run models
# ------------------------------------------------------------

total_jobs = (
    len(MODEL_REGISTRY)
    * len(cv_splits)
)

completed_jobs = 0

overall_start_time = time.time()

print("=" * 72)
print("REPEATED CROSS-VALIDATION")
print("=" * 72)

print("Models:", len(MODEL_REGISTRY))
print("Folds per model:", len(cv_splits))
print("Total fits:", total_jobs)
print()


for model_name, base_pipeline in MODEL_REGISTRY.items():

    print("\n" + "=" * 72)
    print("MODEL:", model_name)
    print("=" * 72)

    model_start_time = time.time()

    for split_number, (
        train_index,
        test_index
    ) in enumerate(
        cv_splits,
        start=1
    ):

        repeat_id = (
            (split_number - 1) // N_SPLITS
        ) + 1

        fold_id = (
            (split_number - 1) % N_SPLITS
        ) + 1

        completed_jobs += 1

        fold_start_time = time.time()

        print(
            f"[{completed_jobs:>2}/{total_jobs}] "
            f"{model_name} | "
            f"Repeat {repeat_id}/{N_REPEATS} | "
            f"Fold {fold_id}/{N_SPLITS}"
        )

        X_train = X_values[train_index]
        X_test = X_values[test_index]

        y_train = y_values[train_index]
        y_test = y_values[test_index]

        patient_train = patient_ids[train_index]
        patient_test = patient_ids[test_index]

        model = clone(base_pipeline)

        # ----------------------------------------------------
        # Fit using training data only
        # ----------------------------------------------------

        model.fit(
            X_train,
            y_train
        )

        # ----------------------------------------------------
        # Held-out probabilities
        # ----------------------------------------------------

        y_probability = model.predict_proba(
            X_test
        )[:, 1]

        y_predicted = (
            y_probability >= 0.5
        ).astype(int)

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        fold_metrics = calculate_binary_metrics(
            y_true=y_test,
            y_probability=y_probability,
            threshold=0.5
        )

        fold_duration_seconds = (
            time.time() - fold_start_time
        )

        fold_performance_records.append({
            "model_name": model_name,
            "repeat_id": repeat_id,
            "fold_id": fold_id,

            "n_train": int(len(train_index)),
            "n_test": int(len(test_index)),

            "n_train_early": int(
                (y_train == 0).sum()
            ),

            "n_train_advanced": int(
                (y_train == 1).sum()
            ),

            "n_test_early": int(
                (y_test == 0).sum()
            ),

            "n_test_advanced": int(
                (y_test == 1).sum()
            ),

            "fit_duration_seconds": (
                fold_duration_seconds
            ),

            **fold_metrics
        })

        # ----------------------------------------------------
        # Patient-level prediction records
        # ----------------------------------------------------

        for local_index, original_index in enumerate(
            test_index
        ):

            probability = float(
                y_probability[local_index]
            )

            predicted_label = int(
                y_predicted[local_index]
            )

            true_label = int(
                y_values[original_index]
            )

            prediction_records.append({
                "model_name": model_name,
                "patient_id": patient_ids[
                    original_index
                ],

                "repeat_id": repeat_id,
                "fold_id": fold_id,

                "true_label": true_label,

                "true_group": (
                    "Advanced"
                    if true_label == 1
                    else "Early"
                ),

                "predicted_probability_advanced": (
                    probability
                ),

                "predicted_probability_early": (
                    1 - probability
                ),

                "predicted_label": (
                    predicted_label
                ),

                "predicted_group": (
                    "Advanced"
                    if predicted_label == 1
                    else "Early"
                ),

                "prediction_correct": int(
                    predicted_label
                    == true_label
                ),

                "probability_margin_from_0_5": (
                    abs(probability - 0.5)
                )
            })

        # ----------------------------------------------------
        # Selected genes from training-fold selector
        # ----------------------------------------------------

        selector = model.named_steps[
            "feature_selection"
        ]

        selected_mask = (
            selector.get_support()
        )

        selected_gene_names = gene_names[
            selected_mask
        ]

        selector_scores = selector.scores_[
            selected_mask
        ]

        selector_pvalues = selector.pvalues_[
            selected_mask
        ]

        for gene_name, f_score, p_value in zip(
            selected_gene_names,
            selector_scores,
            selector_pvalues
        ):

            selected_gene_records.append({
                "model_name": model_name,
                "repeat_id": repeat_id,
                "fold_id": fold_id,
                "gene_id": gene_name,

                "training_f_score": (
                    float(f_score)
                    if np.isfinite(f_score)
                    else np.nan
                ),

                "training_p_value": (
                    float(p_value)
                    if np.isfinite(p_value)
                    else np.nan
                )
            })

        # ----------------------------------------------------
        # Model-specific feature contribution
        # ----------------------------------------------------

        classifier = model.named_steps[
            "classifier"
        ]

        if model_name == "ElasticNet_Logistic":

            coefficients = (
                classifier.coef_[0]
            )

            for gene_name, coefficient in zip(
                selected_gene_names,
                coefficients
            ):

                model_feature_records.append({
                    "model_name": model_name,
                    "repeat_id": repeat_id,
                    "fold_id": fold_id,
                    "gene_id": gene_name,

                    "feature_measure": (
                        "logistic_coefficient"
                    ),

                    "feature_value": float(
                        coefficient
                    ),

                    "absolute_feature_value": float(
                        abs(coefficient)
                    )
                })

        elif model_name == "ExtraTrees_BlackBox":

            importances = (
                classifier.feature_importances_
            )

            for gene_name, importance in zip(
                selected_gene_names,
                importances
            ):

                model_feature_records.append({
                    "model_name": model_name,
                    "repeat_id": repeat_id,
                    "fold_id": fold_id,
                    "gene_id": gene_name,

                    "feature_measure": (
                        "impurity_importance"
                    ),

                    "feature_value": float(
                        importance
                    ),

                    "absolute_feature_value": float(
                        abs(importance)
                    )
                })

        fit_timing_records.append({
            "model_name": model_name,
            "repeat_id": repeat_id,
            "fold_id": fold_id,
            "duration_seconds": (
                fold_duration_seconds
            )
        })

    model_duration_minutes = (
        time.time() - model_start_time
    ) / 60

    print(
        f"\n{model_name} completed in "
        f"{model_duration_minutes:.2f} minutes."
    )


overall_duration_minutes = (
    time.time() - overall_start_time
) / 60

print("\n" + "=" * 72)
print("ALL MODEL FITS COMPLETED")
print("=" * 72)
print(
    "Total duration:",
    round(overall_duration_minutes, 2),
    "minutes"
)


# ------------------------------------------------------------
# Convert collectors into DataFrames
# ------------------------------------------------------------

fold_performance = pd.DataFrame(
    fold_performance_records
)

oof_predictions_all = pd.DataFrame(
    prediction_records
)

selected_genes_by_fold = pd.DataFrame(
    selected_gene_records
)

model_features_by_fold = pd.DataFrame(
    model_feature_records
)

fit_timing = pd.DataFrame(
    fit_timing_records
)


# ------------------------------------------------------------
# Basic integrity checks
# ------------------------------------------------------------

expected_prediction_rows = (
    len(MODEL_REGISTRY)
    * X.shape[0]
    * N_REPEATS
)

assert len(oof_predictions_all) == (
    expected_prediction_rows
), (
    "Unexpected number of held-out prediction rows."
)

prediction_count_check = (
    oof_predictions_all
    .groupby(
        [
            "model_name",
            "patient_id"
        ]
    )
    .size()
)

assert (
    prediction_count_check == N_REPEATS
).all(), (
    "At least one patient does not have exactly "
    "one held-out prediction per repeat."
)

assert len(fold_performance) == (
    len(MODEL_REGISTRY)
    * len(cv_splits)
)

print("\nFold performance preview:")
display(
    fold_performance.head(10)
)

print("\nHeld-out predictions preview:")
display(
    oof_predictions_all.head(10)
)

logger.info(
    "Repeated CV completed: %s total model fits; "
    "%.2f minutes.",
    total_jobs,
    overall_duration_minutes
)