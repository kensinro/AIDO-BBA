# ============================================================
# CELL 18
# Held-out SHAP attribution for ExtraTrees black-box model
# ============================================================

from pathlib import Path
from aido_bba.config import (
    data_root, brca_output_root, brca_run_dir, go_bp_gmt,
    hgnc_file, ncbi_gene_info, metabric_dir, gse96058_dir,
    kirc_dir, kirc_output_root,
)

import importlib.util
import time
import json

# ------------------------------------------------------------
# 1. Check SHAP installation
# ------------------------------------------------------------

if importlib.util.find_spec("shap") is None:
    raise ImportError(
        "The 'shap' package is not installed.\n"
        "Run the following in a new Jupyter cell:\n\n"
        "%pip install shap\n\n"
        "Then restart the kernel, rerun the previous cells, "
        "and run CELL 18 again."
    )

import shap

print("SHAP version:", shap.__version__)


# ------------------------------------------------------------
# 2. Output directories
# ------------------------------------------------------------

ATTRIBUTION_DIR = RUN_DIR / "05_attribution"
SHAP_MATRIX_DIR = ATTRIBUTION_DIR / "shap_matrices"
SHAP_SUMMARY_DIR = ATTRIBUTION_DIR / "summaries"

ATTRIBUTION_DIR.mkdir(
    parents=True,
    exist_ok=True
)

SHAP_MATRIX_DIR.mkdir(
    parents=True,
    exist_ok=True
)

SHAP_SUMMARY_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ------------------------------------------------------------
# 3. Settings
# ------------------------------------------------------------

SHAP_MODEL_NAME = "ExtraTrees_BlackBox"

# Number of top absolute-attribution genes retained in the
# compact patient-level table.
TOP_GENES_PER_PATIENT = 50

# Save full fold-specific SHAP matrices as compressed NPZ files.
SAVE_FULL_SHAP_MATRICES = True


# ------------------------------------------------------------
# 4. Helper for SHAP output formats
# ------------------------------------------------------------

def extract_positive_class_shap(
    shap_output,
    expected_value,
    n_samples,
    n_features
):
    """
    Normalize different SHAP-version output formats.

    Returns:
        shap_positive:
            n_samples x n_features array for Advanced class.

        expected_positive:
            scalar expected value for Advanced class.
    """

    # Older SHAP versions:
    # list[class_0_array, class_1_array]
    if isinstance(shap_output, list):

        if len(shap_output) == 2:
            shap_positive = np.asarray(
                shap_output[1],
                dtype=float
            )

        elif len(shap_output) == 1:
            shap_positive = np.asarray(
                shap_output[0],
                dtype=float
            )

        else:
            raise ValueError(
                "Unexpected SHAP list length: "
                f"{len(shap_output)}"
            )

    else:
        shap_array = np.asarray(
            shap_output,
            dtype=float
        )

        # n_samples x n_features
        if shap_array.ndim == 2:

            shap_positive = shap_array

        # n_samples x n_features x n_classes
        elif (
            shap_array.ndim == 3
            and shap_array.shape[0] == n_samples
            and shap_array.shape[1] == n_features
        ):

            if shap_array.shape[2] >= 2:
                shap_positive = shap_array[:, :, 1]
            else:
                shap_positive = shap_array[:, :, 0]

        # n_classes x n_samples x n_features
        elif (
            shap_array.ndim == 3
            and shap_array.shape[1] == n_samples
            and shap_array.shape[2] == n_features
        ):

            if shap_array.shape[0] >= 2:
                shap_positive = shap_array[1, :, :]
            else:
                shap_positive = shap_array[0, :, :]

        else:
            raise ValueError(
                "Unexpected SHAP array shape: "
                f"{shap_array.shape}"
            )

    expected_array = np.asarray(
        expected_value
    )

    if expected_array.ndim == 0:
        expected_positive = float(
            expected_array
        )

    elif expected_array.size >= 2:
        expected_positive = float(
            expected_array.reshape(-1)[1]
        )

    else:
        expected_positive = float(
            expected_array.reshape(-1)[0]
        )

    if shap_positive.shape != (
        n_samples,
        n_features
    ):
        raise ValueError(
            "Normalized SHAP shape mismatch: "
            f"{shap_positive.shape}; expected "
            f"{(n_samples, n_features)}"
        )

    return (
        shap_positive,
        expected_positive
    )


# ------------------------------------------------------------
# 5. Output collectors
# ------------------------------------------------------------

shap_fold_audit_records = []
shap_patient_gene_records = []
shap_global_gene_records = []

overall_shap_start = time.time()

print("=" * 72)
print("HELD-OUT EXTRATREES SHAP ATTRIBUTION")
print("=" * 72)

print("Folds:", len(cv_splits))
print("Selected genes per fold:", effective_k)
print("Top genes retained per patient:", TOP_GENES_PER_PATIENT)
print()


# ------------------------------------------------------------
# 6. Fit and explain each held-out fold
# ------------------------------------------------------------

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

    fold_start = time.time()

    print(
        f"[{split_number:>2}/{len(cv_splits)}] "
        f"Repeat {repeat_id}/{N_REPEATS} | "
        f"Fold {fold_id}/{N_SPLITS}"
    )

    X_train = X_values[train_index]
    X_test = X_values[test_index]

    y_train = y_values[train_index]
    y_test = y_values[test_index]

    test_patient_ids = patient_ids[
        test_index
    ]

    # --------------------------------------------------------
    # Fit ExtraTrees pipeline on training fold only
    # --------------------------------------------------------

    fold_model = clone(
        extratrees_pipeline
    )

    fold_model.fit(
        X_train,
        y_train
    )

    imputer = fold_model.named_steps[
        "imputer"
    ]

    selector = fold_model.named_steps[
        "feature_selection"
    ]

    classifier = fold_model.named_steps[
        "classifier"
    ]

    # --------------------------------------------------------
    # Apply fold-specific preprocessing
    # --------------------------------------------------------

    X_test_imputed = imputer.transform(
        X_test
    )

    X_test_selected = selector.transform(
        X_test_imputed
    )

    selected_mask = selector.get_support()

    selected_gene_names = gene_names[
        selected_mask
    ]

    predicted_probability = (
        classifier.predict_proba(
            X_test_selected
        )[:, 1]
    )

    # --------------------------------------------------------
    # Tree SHAP
    # --------------------------------------------------------

    explainer = shap.TreeExplainer(
        classifier,
        feature_perturbation="tree_path_dependent",
        model_output="raw"
    )

    shap_output = explainer.shap_values(
        X_test_selected,
        check_additivity=False
    )

    (
        shap_positive,
        expected_positive
    ) = extract_positive_class_shap(
        shap_output=shap_output,
        expected_value=explainer.expected_value,
        n_samples=X_test_selected.shape[0],
        n_features=X_test_selected.shape[1]
    )

    reconstructed_output = (
        expected_positive
        + shap_positive.sum(axis=1)
    )

    # For sklearn ExtraTrees, this will normally reconstruct
    # positive-class probability. Audit rather than assume.
    additivity_error = (
        reconstructed_output
        - predicted_probability
    )

    # --------------------------------------------------------
    # Save complete fold-specific matrix
    # --------------------------------------------------------

    matrix_file = (
        SHAP_MATRIX_DIR
        / (
            f"extratrees_shap_"
            f"repeat_{repeat_id:02d}_"
            f"fold_{fold_id:02d}.npz"
        )
    )

    if SAVE_FULL_SHAP_MATRICES:

        np.savez_compressed(
            matrix_file,

            patient_ids=np.asarray(
                test_patient_ids,
                dtype=str
            ),

            true_labels=np.asarray(
                y_test,
                dtype=np.int8
            ),

            predicted_probability_advanced=np.asarray(
                predicted_probability,
                dtype=np.float32
            ),

            expected_value_advanced=np.asarray(
                [expected_positive],
                dtype=np.float32
            ),

            selected_genes=np.asarray(
                selected_gene_names,
                dtype=str
            ),

            selected_expression=np.asarray(
                X_test_selected,
                dtype=np.float32
            ),

            shap_values_advanced=np.asarray(
                shap_positive,
                dtype=np.float32
            )
        )

    # --------------------------------------------------------
    # Fold-level additivity audit
    # --------------------------------------------------------

    fold_duration = (
        time.time() - fold_start
    )

    shap_fold_audit_records.append({
        "repeat_id": repeat_id,
        "fold_id": fold_id,
        "n_train": len(train_index),
        "n_test": len(test_index),
        "n_selected_genes": len(
            selected_gene_names
        ),
        "expected_value_advanced": (
            expected_positive
        ),
        "mean_predicted_probability": float(
            predicted_probability.mean()
        ),
        "mean_reconstructed_output": float(
            reconstructed_output.mean()
        ),
        "mean_absolute_additivity_error": float(
            np.mean(
                np.abs(additivity_error)
            )
        ),
        "maximum_absolute_additivity_error": float(
            np.max(
                np.abs(additivity_error)
            )
        ),
        "correlation_reconstructed_vs_probability": float(
            np.corrcoef(
                reconstructed_output,
                predicted_probability
            )[0, 1]
        ),
        "duration_seconds": fold_duration,
        "matrix_file": str(matrix_file)
    })

    # --------------------------------------------------------
    # Fold-level global gene attribution
    # --------------------------------------------------------

    mean_absolute_shap = np.mean(
        np.abs(shap_positive),
        axis=0
    )

    mean_signed_shap = np.mean(
        shap_positive,
        axis=0
    )

    for gene_name, mean_abs, mean_signed in zip(
        selected_gene_names,
        mean_absolute_shap,
        mean_signed_shap
    ):

        shap_global_gene_records.append({
            "repeat_id": repeat_id,
            "fold_id": fold_id,
            "gene_id": gene_name,
            "mean_absolute_shap": float(
                mean_abs
            ),
            "mean_signed_shap": float(
                mean_signed
            ),
            "n_test_patients": len(
                test_index
            )
        })

    # --------------------------------------------------------
    # Compact top-attribution genes per held-out patient
    # --------------------------------------------------------

    top_n = min(
        TOP_GENES_PER_PATIENT,
        shap_positive.shape[1]
    )

    for patient_position, patient_id in enumerate(
        test_patient_ids
    ):

        patient_shap = shap_positive[
            patient_position,
            :
        ]

        top_indices = np.argsort(
            np.abs(patient_shap)
        )[::-1][:top_n]

        for rank_position, feature_index in enumerate(
            top_indices,
            start=1
        ):

            shap_patient_gene_records.append({
                "patient_id": patient_id,
                "repeat_id": repeat_id,
                "fold_id": fold_id,
                "true_label": int(
                    y_test[patient_position]
                ),
                "true_group": (
                    "Advanced"
                    if y_test[patient_position] == 1
                    else "Early"
                ),
                "predicted_probability_advanced": float(
                    predicted_probability[
                        patient_position
                    ]
                ),
                "expected_value_advanced": float(
                    expected_positive
                ),
                "gene_rank": rank_position,
                "gene_id": selected_gene_names[
                    feature_index
                ],
                "expression_value": float(
                    X_test_selected[
                        patient_position,
                        feature_index
                    ]
                ),
                "shap_value_advanced": float(
                    patient_shap[
                        feature_index
                    ]
                ),
                "absolute_shap_value": float(
                    abs(
                        patient_shap[
                            feature_index
                        ]
                    )
                ),
                "attribution_direction": (
                    "toward_advanced"
                    if patient_shap[
                        feature_index
                    ] > 0
                    else "toward_early"
                )
            })


# ------------------------------------------------------------
# 7. Convert outputs to DataFrames
# ------------------------------------------------------------

shap_fold_audit = pd.DataFrame(
    shap_fold_audit_records
)

shap_patient_top_genes = pd.DataFrame(
    shap_patient_gene_records
)

shap_global_by_fold = pd.DataFrame(
    shap_global_gene_records
)


# ------------------------------------------------------------
# 8. Aggregate global SHAP stability
# ------------------------------------------------------------

shap_global_gene_stability = (
    shap_global_by_fold
    .groupby(
        "gene_id",
        as_index=False
    )
    .agg(
        n_folds_selected=(
            "fold_id",
            "size"
        ),
        mean_absolute_shap=(
            "mean_absolute_shap",
            "mean"
        ),
        median_absolute_shap=(
            "mean_absolute_shap",
            "median"
        ),
        sd_absolute_shap=(
            "mean_absolute_shap",
            "std"
        ),
        mean_signed_shap=(
            "mean_signed_shap",
            "mean"
        )
    )
)

shap_global_gene_stability[
    "selection_frequency"
] = (
    shap_global_gene_stability[
        "n_folds_selected"
    ]
    / len(cv_splits)
)

shap_global_gene_stability[
    "absolute_shap_cv"
] = (
    shap_global_gene_stability[
        "sd_absolute_shap"
    ]
    /
    shap_global_gene_stability[
        "mean_absolute_shap"
    ].replace(0, np.nan)
)

shap_global_gene_stability = (
    shap_global_gene_stability
    .sort_values(
        [
            "mean_absolute_shap",
            "selection_frequency"
        ],
        ascending=[
            False,
            False
        ]
    )
    .reset_index(drop=True)
)


# ------------------------------------------------------------
# 9. Save tabular outputs
# ------------------------------------------------------------

shap_fold_audit.to_csv(
    SHAP_SUMMARY_DIR
    / "shap_fold_additivity_audit.tsv",
    sep="\t",
    index=False
)

shap_patient_top_genes.to_csv(
    SHAP_SUMMARY_DIR
    / "heldout_patient_top50_gene_attributions.tsv",
    sep="\t",
    index=False
)

shap_global_by_fold.to_csv(
    SHAP_SUMMARY_DIR
    / "global_gene_shap_by_fold.tsv",
    sep="\t",
    index=False
)

shap_global_gene_stability.to_csv(
    SHAP_SUMMARY_DIR
    / "global_gene_shap_stability.tsv",
    sep="\t",
    index=False
)


# ------------------------------------------------------------
# 10. Display summary
# ------------------------------------------------------------

total_shap_minutes = (
    time.time() - overall_shap_start
) / 60

print("\n" + "=" * 72)
print("HELD-OUT SHAP ATTRIBUTION COMPLETED")
print("=" * 72)

print(
    "Total duration:",
    round(total_shap_minutes, 2),
    "minutes"
)

print(
    "Fold audit rows:",
    len(shap_fold_audit)
)

print(
    "Patient top-gene rows:",
    len(shap_patient_top_genes)
)

print(
    "Global fold-gene rows:",
    len(shap_global_by_fold)
)

print("\nAdditivity audit:")

display(
    shap_fold_audit[
        [
            "repeat_id",
            "fold_id",
            "n_test",
            "expected_value_advanced",
            "mean_predicted_probability",
            "mean_reconstructed_output",
            "mean_absolute_additivity_error",
            "maximum_absolute_additivity_error",
            "correlation_reconstructed_vs_probability"
        ]
    ].round(8)
)

print("\nTop global held-out SHAP genes:")

display(
    shap_global_gene_stability.head(30)
)

logger.info(
    "Held-out ExtraTrees SHAP attribution completed: "
    "%s folds; %.2f minutes.",
    len(cv_splits),
    total_shap_minutes
)