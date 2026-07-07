# ============================================================
# AIDO-BBA BRCA 1.0
# ONE-CELL RESTORE + HELD-OUT EXTRATREES SHAP
#
# 功能：
# 1. 自動尋找原本已完成的 BBA run folder
# 2. 重新讀取 TCGA-BRCA GE 與 stage
# 3. 重建 1073 patients × 20247 genes
# 4. 重建相同的 5-fold × 5-repeat CV
# 5. 重建 ExtraTrees pipeline
# 6. 對每個 held-out fold 計算 SHAP
# 7. 每 fold 即時存檔，可中斷後續跑
# 8. 輸出 gene-level SHAP stability 與 additivity audit
# ============================================================

from pathlib import Path
from datetime import datetime
import json
import logging
import platform
import sys
import time
import warnings

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import RepeatedStratifiedKFold

import shap

warnings.filterwarnings("ignore")

print("=" * 78)
print("AIDO-BBA BRCA | RESTORE + HELD-OUT SHAP")
print("=" * 78)

print("Python :", sys.version.split()[0])
print("NumPy  :", np.__version__)
print("pandas :", pd.__version__)
print("SHAP   :", shap.__version__)


# ============================================================
# 0. SETTINGS
# ============================================================

RANDOM_SEED = 20260701

N_SPLITS = 5
N_REPEATS = 5
N_SELECTED_GENES = 1500

N_TREES = 400
MAX_DEPTH = None
MIN_SAMPLES_LEAF = 3
MAX_FEATURES = "sqrt"
N_JOBS = -1

TOP_GENES_PER_PATIENT = 50
SAVE_FULL_SHAP_MATRICES = True
RESUME_EXISTING_FOLDS = True

DATA_ROOT = Path(r"D:\AIDO-Data")

TCGA_BRCA_DIR = (
    DATA_ROOT
    / "UCSC_XENA"
    / "Breast Cancer (BRCA)"
)

GE_FILE = TCGA_BRCA_DIR / "GE.tsv"

STAGE_FILE = (
    TCGA_BRCA_DIR
    / "BRCA_stage_groups_from_survival.tsv"
)

OUTPUT_ROOT = Path(
    r"D:\AIDO-Temp\AIDO_BBA_BRCA_1_0"
)


# ============================================================
# 1. FIND THE COMPLETED ORIGINAL RUN
# ============================================================

required_blackbox_files = [
    "fold_performance_all_models.tsv",
    "oof_predictions_all_models_all_repeats.tsv",
    "patient_oof_prediction_summary.tsv",
    "bba_patient_state_taxonomy.tsv"
]

candidate_runs = []

if OUTPUT_ROOT.exists():

    for run_dir in OUTPUT_ROOT.iterdir():

        if not run_dir.is_dir():
            continue

        blackbox_dir = run_dir / "04_blackbox"

        n_required_found = sum(
            (blackbox_dir / filename).exists()
            for filename in required_blackbox_files
        )

        candidate_runs.append({
            "run_dir": run_dir,
            "n_required_found": n_required_found,
            "modified_time": run_dir.stat().st_mtime
        })

if len(candidate_runs) == 0:
    raise FileNotFoundError(
        f"No AIDO-BBA run folders found under:\n{OUTPUT_ROOT}"
    )

candidate_runs = sorted(
    candidate_runs,
    key=lambda record: (
        record["n_required_found"],
        record["modified_time"]
    ),
    reverse=True
)

RUN_DIR = candidate_runs[0]["run_dir"]

print("\nSelected original run:")
print(RUN_DIR)

print(
    "Completed black-box files found:",
    candidate_runs[0]["n_required_found"],
    "/",
    len(required_blackbox_files)
)

DIRS = {
    "manifest": RUN_DIR / "00_manifest",
    "input_audit": RUN_DIR / "01_input_audit",
    "harmonization": RUN_DIR / "02_harmonization",
    "endpoint": RUN_DIR / "03_endpoint",
    "blackbox": RUN_DIR / "04_blackbox",
    "logs": RUN_DIR / "logs"
}

ATTRIBUTION_DIR = RUN_DIR / "05_attribution"
SHAP_MATRIX_DIR = ATTRIBUTION_DIR / "shap_matrices"
SHAP_FOLD_DIR = ATTRIBUTION_DIR / "fold_tables"
SHAP_SUMMARY_DIR = ATTRIBUTION_DIR / "summaries"

for directory in [
    ATTRIBUTION_DIR,
    SHAP_MATRIX_DIR,
    SHAP_FOLD_DIR,
    SHAP_SUMMARY_DIR
]:
    directory.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# 2. LOGGING
# ============================================================

LOG_FILE = (
    ATTRIBUTION_DIR
    / "AIDO_BBA_BRCA_SHAP.log"
)

logger = logging.getLogger(
    "AIDO_BBA_BRCA_SHAP"
)

logger.setLevel(logging.INFO)
logger.handlers.clear()

file_handler = logging.FileHandler(
    LOG_FILE,
    mode="a",
    encoding="utf-8"
)

console_handler = logging.StreamHandler()

formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(message)s"
)

file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)


# ============================================================
# 3. UTILITY FUNCTIONS
# ============================================================

def normalize_tcga_barcode(value):

    if pd.isna(value):
        return np.nan

    value = str(value).strip()
    value = value.replace(".", "-")
    value = value.upper()

    return value


def tcga_patient_id(value):

    value = normalize_tcga_barcode(value)

    if pd.isna(value):
        return np.nan

    if value.startswith("TCGA-") and len(value) >= 12:
        return value[:12]

    return value


def tcga_sample_type_code(value):

    value = normalize_tcga_barcode(value)

    if pd.isna(value):
        return np.nan

    parts = value.split("-")

    if len(parts) >= 4 and len(parts[3]) >= 2:
        return parts[3][:2]

    return np.nan


def normalize_stage_group(value):

    if pd.isna(value):
        return np.nan

    text = str(value).strip().lower()

    if text == "early":
        return "Early"

    if text in {"advanced", "late"}:
        return "Advanced"

    return np.nan


def extract_positive_class_shap(
    shap_output,
    expected_value,
    n_samples,
    n_features
):
    """
    支援不同 SHAP 版本的輸出格式。
    """

    if isinstance(shap_output, list):

        if len(shap_output) >= 2:
            shap_positive = np.asarray(
                shap_output[1],
                dtype=float
            )
        else:
            shap_positive = np.asarray(
                shap_output[0],
                dtype=float
            )

    else:

        shap_array = np.asarray(
            shap_output,
            dtype=float
        )

        if shap_array.ndim == 2:

            shap_positive = shap_array

        elif (
            shap_array.ndim == 3
            and shap_array.shape[0] == n_samples
            and shap_array.shape[1] == n_features
        ):

            class_index = (
                1
                if shap_array.shape[2] >= 2
                else 0
            )

            shap_positive = (
                shap_array[:, :, class_index]
            )

        elif (
            shap_array.ndim == 3
            and shap_array.shape[1] == n_samples
            and shap_array.shape[2] == n_features
        ):

            class_index = (
                1
                if shap_array.shape[0] >= 2
                else 0
            )

            shap_positive = (
                shap_array[class_index, :, :]
            )

        else:
            raise ValueError(
                "Unexpected SHAP output shape: "
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
            "SHAP matrix shape mismatch: "
            f"{shap_positive.shape}; expected "
            f"{(n_samples, n_features)}"
        )

    return shap_positive, expected_positive


# ============================================================
# 4. INPUT CHECK
# ============================================================

for required_file in [
    GE_FILE,
    STAGE_FILE
]:

    if not required_file.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{required_file}"
        )

print("\nInput files found:")
print("GE   :", GE_FILE)
print("Stage:", STAGE_FILE)


# ============================================================
# 5. REBUILD PATIENT-LEVEL STAGE ENDPOINT
# ============================================================

logger.info("Loading stage data.")

stage_raw = pd.read_csv(
    STAGE_FILE,
    sep="\t",
    low_memory=False
)

required_stage_columns = [
    "sampleID",
    "_PATIENT",
    "stage_raw",
    "stage_group"
]

missing_stage_columns = [
    column
    for column in required_stage_columns
    if column not in stage_raw.columns
]

if missing_stage_columns:
    raise ValueError(
        "Missing stage columns:\n"
        + "\n".join(missing_stage_columns)
    )

stage_patient = stage_raw[
    required_stage_columns
].copy()

stage_patient.columns = [
    "sample_id",
    "patient_id",
    "stage_raw",
    "stage_group"
]

stage_patient["sample_id"] = (
    stage_patient["sample_id"]
    .map(normalize_tcga_barcode)
)

stage_patient["patient_id"] = (
    stage_patient["patient_id"]
    .map(normalize_tcga_barcode)
    .map(tcga_patient_id)
)

stage_patient["stage_group"] = (
    stage_patient["stage_group"]
    .map(normalize_stage_group)
)

stage_patient["stage_label"] = (
    stage_patient["stage_group"]
    .map({
        "Early": 0,
        "Advanced": 1
    })
)

stage_patient["sample_type_code"] = (
    stage_patient["sample_id"]
    .map(tcga_sample_type_code)
)

stage_patient["is_primary_tumour"] = (
    stage_patient["sample_type_code"] == "01"
)

stage_patient = stage_patient[
    stage_patient["stage_label"].notna()
    & stage_patient["patient_id"].notna()
].copy()

stage_patient["stage_label"] = (
    stage_patient["stage_label"]
    .astype(int)
)

# Exclude patients with conflicting labels
patient_label_count = (
    stage_patient
    .groupby("patient_id")["stage_label"]
    .nunique()
)

conflicting_patient_ids = (
    patient_label_count[
        patient_label_count > 1
    ]
    .index
    .tolist()
)

stage_patient = stage_patient[
    ~stage_patient["patient_id"].isin(
        conflicting_patient_ids
    )
].copy()

# Prefer primary-tumour record
stage_patient = (
    stage_patient
    .sort_values(
        [
            "patient_id",
            "is_primary_tumour",
            "sample_id"
        ],
        ascending=[
            True,
            False,
            True
        ]
    )
    .drop_duplicates(
        subset=["patient_id"],
        keep="first"
    )
    .reset_index(drop=True)
)

print("\nStage endpoint:")
print(
    stage_patient["stage_group"]
    .value_counts()
)


# ============================================================
# 6. LOAD AND TRANSPOSE GENE EXPRESSION
# ============================================================

logger.info("Loading GE matrix.")

ge_raw = pd.read_csv(
    GE_FILE,
    sep="\t",
    low_memory=False
)

gene_column = ge_raw.columns[0]

ge_raw = ge_raw.rename(
    columns={
        gene_column: "gene_id"
    }
)

ge_raw["gene_id"] = (
    ge_raw["gene_id"]
    .astype(str)
    .str.strip()
)

sample_columns = [
    column
    for column in ge_raw.columns
    if column != "gene_id"
]

expression_numeric = (
    ge_raw[sample_columns]
    .apply(
        pd.to_numeric,
        errors="coerce"
    )
)

expression_numeric.index = (
    ge_raw["gene_id"].values
)

# Safe duplicate aggregation
expression_gene_by_sample = (
    expression_numeric
    .groupby(
        level=0,
        sort=False
    )
    .mean()
)

expression = (
    expression_gene_by_sample.T
)

expression.index = [
    normalize_tcga_barcode(sample_id)
    for sample_id in expression.index
]

expression.index.name = "sample_id"

print("\nRaw expression matrix:")
print(expression.shape)


# ============================================================
# 7. PRIMARY-TUMOUR FILTER
# ============================================================

expression_sample_manifest = pd.DataFrame({
    "sample_id": expression.index
})

expression_sample_manifest[
    "patient_id"
] = (
    expression_sample_manifest["sample_id"]
    .map(tcga_patient_id)
)

expression_sample_manifest[
    "sample_type_code"
] = (
    expression_sample_manifest["sample_id"]
    .map(tcga_sample_type_code)
)

expression_sample_manifest[
    "is_primary_tumour"
] = (
    expression_sample_manifest[
        "sample_type_code"
    ] == "01"
)

primary_sample_manifest = (
    expression_sample_manifest[
        expression_sample_manifest[
            "is_primary_tumour"
        ]
    ]
    .sort_values(
        [
            "patient_id",
            "sample_id"
        ]
    )
    .drop_duplicates(
        subset=["patient_id"],
        keep="first"
    )
    .reset_index(drop=True)
)

expression_primary = expression.loc[
    primary_sample_manifest[
        "sample_id"
    ].tolist()
].copy()

print("\nPrimary-tumour GE:")
print(expression_primary.shape)


# ============================================================
# 8. MATCH EXPRESSION WITH STAGE
# ============================================================

analysis_cohort = (
    primary_sample_manifest[
        [
            "sample_id",
            "patient_id"
        ]
    ]
    .rename(
        columns={
            "sample_id":
                "expression_sample_id"
        }
    )
    .merge(
        stage_patient[
            [
                "patient_id",
                "stage_raw",
                "stage_group",
                "stage_label"
            ]
        ],
        on="patient_id",
        how="inner",
        validate="one_to_one"
    )
    .sort_values("patient_id")
    .reset_index(drop=True)
)

print("\nMatched cohort:")
print(analysis_cohort.shape)

print(
    analysis_cohort["stage_group"]
    .value_counts()
)

if len(analysis_cohort) != 1073:
    raise ValueError(
        "Expected 1073 patients, found "
        f"{len(analysis_cohort)}."
    )

if int(
    (analysis_cohort["stage_label"] == 0).sum()
) != 803:
    raise ValueError(
        "Expected 803 Early patients."
    )

if int(
    (analysis_cohort["stage_label"] == 1).sum()
) != 270:
    raise ValueError(
        "Expected 270 Advanced patients."
    )


# ============================================================
# 9. REBUILD MODEL MATRIX
# ============================================================

matched_sample_ids = (
    analysis_cohort[
        "expression_sample_id"
    ]
    .tolist()
)

matched_patient_ids = (
    analysis_cohort[
        "patient_id"
    ]
    .tolist()
)

X_raw = expression_primary.loc[
    matched_sample_ids
].copy()

X_raw.index = matched_patient_ids
X_raw.index.name = "patient_id"

y = pd.Series(
    analysis_cohort[
        "stage_label"
    ].to_numpy(dtype=int),
    index=matched_patient_ids,
    name="stage_label"
)

X_numeric = X_raw.apply(
    pd.to_numeric,
    errors="coerce"
)

# Remove all-nonfinite genes
finite_gene_mask = np.isfinite(
    X_numeric.to_numpy(dtype=float)
).any(axis=0)

X_numeric = X_numeric.loc[
    :,
    finite_gene_mask
]

# Remove zero-variance genes
gene_variance = X_numeric.var(
    axis=0,
    ddof=1
)

eligible_genes = gene_variance[
    gene_variance > 0
].index

X = X_numeric.loc[
    :,
    eligible_genes
].copy()

print("\nFinal model matrix:")
print(X.shape)

if X.shape != (1073, 20247):
    raise ValueError(
        "Expected model matrix (1073, 20247), "
        f"found {X.shape}."
    )


# ============================================================
# 10. REBUILD IDENTICAL CV SPLITS
# ============================================================

X_values = X.to_numpy(
    dtype=np.float32
)

y_values = y.loc[
    X.index
].to_numpy(
    dtype=int
)

patient_ids = X.index.to_numpy()
gene_names = X.columns.to_numpy()

effective_k = min(
    N_SELECTED_GENES,
    X.shape[1]
)

cv_generator = RepeatedStratifiedKFold(
    n_splits=N_SPLITS,
    n_repeats=N_REPEATS,
    random_state=RANDOM_SEED
)

cv_splits = list(
    cv_generator.split(
        X_values,
        y_values
    )
)

print("\nCV splits:", len(cv_splits))


# ============================================================
# 11. REBUILD EXTRATREES PIPELINE
# ============================================================

extratrees_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),
        (
            "feature_selection",
            SelectKBest(
                score_func=f_classif,
                k=effective_k
            )
        ),
        (
            "classifier",
            ExtraTreesClassifier(
                n_estimators=N_TREES,
                max_depth=MAX_DEPTH,
                min_samples_leaf=MIN_SAMPLES_LEAF,
                max_features=MAX_FEATURES,
                class_weight="balanced",
                random_state=RANDOM_SEED,
                n_jobs=N_JOBS
            )
        )
    ]
)


# ============================================================
# 12. SHAP FOLD LOOP
# ============================================================

print("\n" + "=" * 78)
print("STARTING HELD-OUT EXTRATREES SHAP")
print("=" * 78)

print("Folds                 :", len(cv_splits))
print("Selected genes/fold   :", effective_k)
print("Top genes/patient     :", TOP_GENES_PER_PATIENT)
print("Resume existing folds :", RESUME_EXISTING_FOLDS)

overall_start = time.time()

completed_fold_count = 0
skipped_fold_count = 0

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

    fold_prefix = (
        f"repeat_{repeat_id:02d}_"
        f"fold_{fold_id:02d}"
    )

    matrix_file = (
        SHAP_MATRIX_DIR
        / f"extratrees_shap_{fold_prefix}.npz"
    )

    patient_table_file = (
        SHAP_FOLD_DIR
        / f"patient_top_genes_{fold_prefix}.tsv"
    )

    global_table_file = (
        SHAP_FOLD_DIR
        / f"global_gene_shap_{fold_prefix}.tsv"
    )

    audit_file = (
        SHAP_FOLD_DIR
        / f"shap_audit_{fold_prefix}.json"
    )

    fold_outputs_exist = all([
        patient_table_file.exists(),
        global_table_file.exists(),
        audit_file.exists(),
        (
            matrix_file.exists()
            if SAVE_FULL_SHAP_MATRICES
            else True
        )
    ])

    if (
        RESUME_EXISTING_FOLDS
        and fold_outputs_exist
    ):

        skipped_fold_count += 1

        print(
            f"[{split_number:>2}/{len(cv_splits)}] "
            f"{fold_prefix} | SKIPPED"
        )

        continue

    fold_start = time.time()

    print(
        f"[{split_number:>2}/{len(cv_splits)}] "
        f"{fold_prefix} | RUNNING"
    )

    X_train = X_values[
        train_index
    ]

    X_test = X_values[
        test_index
    ]

    y_train = y_values[
        train_index
    ]

    y_test = y_values[
        test_index
    ]

    test_patient_ids = patient_ids[
        test_index
    ]

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

    explainer = shap.TreeExplainer(
        classifier,
        feature_perturbation=(
            "tree_path_dependent"
        ),
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
        expected_value=(
            explainer.expected_value
        ),
        n_samples=X_test_selected.shape[0],
        n_features=X_test_selected.shape[1]
    )

    reconstructed_output = (
        expected_positive
        + shap_positive.sum(axis=1)
    )

    additivity_error = (
        reconstructed_output
        - predicted_probability
    )

    correlation = np.corrcoef(
        reconstructed_output,
        predicted_probability
    )[0, 1]

    # --------------------------------------------------------
    # Save full compressed matrix
    # --------------------------------------------------------

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
    # Global fold-level SHAP
    # --------------------------------------------------------

    global_fold_table = pd.DataFrame({
        "repeat_id": repeat_id,
        "fold_id": fold_id,
        "gene_id": selected_gene_names,

        "mean_absolute_shap": np.mean(
            np.abs(shap_positive),
            axis=0
        ),

        "median_absolute_shap": np.median(
            np.abs(shap_positive),
            axis=0
        ),

        "mean_signed_shap": np.mean(
            shap_positive,
            axis=0
        ),

        "n_test_patients": len(
            test_index
        )
    })

    global_fold_table.to_csv(
        global_table_file,
        sep="\t",
        index=False
    )

    # --------------------------------------------------------
    # Top SHAP genes per patient
    # --------------------------------------------------------

    patient_records = []

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

            shap_value = float(
                patient_shap[
                    feature_index
                ]
            )

            patient_records.append({
                "patient_id": patient_id,
                "repeat_id": repeat_id,
                "fold_id": fold_id,

                "true_label": int(
                    y_test[
                        patient_position
                    ]
                ),

                "true_group": (
                    "Advanced"
                    if y_test[
                        patient_position
                    ] == 1
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

                "shap_value_advanced": (
                    shap_value
                ),

                "absolute_shap_value": abs(
                    shap_value
                ),

                "attribution_direction": (
                    "toward_advanced"
                    if shap_value > 0
                    else (
                        "toward_early"
                        if shap_value < 0
                        else "neutral"
                    )
                )
            })

    patient_fold_table = pd.DataFrame(
        patient_records
    )

    patient_fold_table.to_csv(
        patient_table_file,
        sep="\t",
        index=False
    )

    # --------------------------------------------------------
    # Fold audit JSON
    # --------------------------------------------------------

    fold_duration = (
        time.time()
        - fold_start
    )

    audit_record = {
        "repeat_id": repeat_id,
        "fold_id": fold_id,
        "n_train": int(
            len(train_index)
        ),
        "n_test": int(
            len(test_index)
        ),
        "n_selected_genes": int(
            len(selected_gene_names)
        ),
        "expected_value_advanced": float(
            expected_positive
        ),
        "mean_predicted_probability": float(
            np.mean(
                predicted_probability
            )
        ),
        "mean_reconstructed_output": float(
            np.mean(
                reconstructed_output
            )
        ),
        "mean_absolute_additivity_error": float(
            np.mean(
                np.abs(
                    additivity_error
                )
            )
        ),
        "maximum_absolute_additivity_error": float(
            np.max(
                np.abs(
                    additivity_error
                )
            )
        ),
        "correlation_reconstructed_vs_probability": float(
            correlation
        ),
        "duration_seconds": float(
            fold_duration
        ),
        "matrix_file": str(
            matrix_file
        ),
        "patient_table_file": str(
            patient_table_file
        ),
        "global_table_file": str(
            global_table_file
        )
    }

    with open(
        audit_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            audit_record,
            file,
            indent=2
        )

    completed_fold_count += 1

    logger.info(
        "SHAP completed: repeat %s fold %s | "
        "mean abs additivity error %.10f | "
        "correlation %.8f | %.2f sec",
        repeat_id,
        fold_id,
        audit_record[
            "mean_absolute_additivity_error"
        ],
        correlation,
        fold_duration
    )


# ============================================================
# 13. COMBINE ALL FOLD OUTPUTS
# ============================================================

print("\n" + "=" * 78)
print("COMBINING SHAP FOLD OUTPUTS")
print("=" * 78)

audit_records = []

for audit_file in sorted(
    SHAP_FOLD_DIR.glob(
        "shap_audit_repeat_*_fold_*.json"
    )
):

    with open(
        audit_file,
        "r",
        encoding="utf-8"
    ) as file:

        audit_records.append(
            json.load(file)
        )

shap_fold_audit = pd.DataFrame(
    audit_records
)

if len(shap_fold_audit) != len(cv_splits):
    print(
        "WARNING: expected",
        len(cv_splits),
        "fold audits, found",
        len(shap_fold_audit)
    )

patient_fold_files = sorted(
    SHAP_FOLD_DIR.glob(
        "patient_top_genes_repeat_*_fold_*.tsv"
    )
)

global_fold_files = sorted(
    SHAP_FOLD_DIR.glob(
        "global_gene_shap_repeat_*_fold_*.tsv"
    )
)

shap_patient_top_genes = pd.concat(
    [
        pd.read_csv(
            file,
            sep="\t"
        )
        for file in patient_fold_files
    ],
    ignore_index=True
)

shap_global_by_fold = pd.concat(
    [
        pd.read_csv(
            file,
            sep="\t"
        )
        for file in global_fold_files
    ],
    ignore_index=True
)


# ============================================================
# 14. GLOBAL GENE SHAP STABILITY
# ============================================================

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
        ),

        median_signed_shap=(
            "mean_signed_shap",
            "median"
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
    ].replace(
        0,
        np.nan
    )
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


# ============================================================
# 15. PATIENT–GENE ATTRIBUTION STABILITY
# ============================================================

patient_gene_stability = (
    shap_patient_top_genes
    .groupby(
        [
            "patient_id",
            "gene_id"
        ],
        as_index=False
    )
    .agg(
        n_repeats_top50=(
            "repeat_id",
            "nunique"
        ),

        mean_absolute_shap=(
            "absolute_shap_value",
            "mean"
        ),

        median_absolute_shap=(
            "absolute_shap_value",
            "median"
        ),

        mean_signed_shap=(
            "shap_value_advanced",
            "mean"
        ),

        fraction_toward_advanced=(
            "shap_value_advanced",
            lambda values: float(
                np.mean(
                    values > 0
                )
            )
        ),

        best_median_rank=(
            "gene_rank",
            "median"
        )
    )
)

patient_gene_stability[
    "top50_repeat_frequency"
] = (
    patient_gene_stability[
        "n_repeats_top50"
    ]
    / N_REPEATS
)

patient_gene_stability = (
    patient_gene_stability
    .sort_values(
        [
            "patient_id",
            "top50_repeat_frequency",
            "mean_absolute_shap"
        ],
        ascending=[
            True,
            False,
            False
        ]
    )
)


# ============================================================
# 16. SAVE FINAL TABLES
# ============================================================

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

patient_gene_stability.to_csv(
    SHAP_SUMMARY_DIR
    / "patient_gene_attribution_stability.tsv",
    sep="\t",
    index=False
)


# ============================================================
# 17. SAVE SHAP MANIFEST
# ============================================================

shap_manifest = {
    "project": "AIDO-BBA BRCA 1.0",
    "analysis": (
        "heldout_extratrees_gene_level_shap"
    ),
    "run_directory": str(
        RUN_DIR
    ),
    "attribution_directory": str(
        ATTRIBUTION_DIR
    ),
    "datetime": datetime.now().isoformat(),
    "python_version": sys.version,
    "numpy_version": np.__version__,
    "pandas_version": pd.__version__,
    "shap_version": shap.__version__,
    "random_seed": RANDOM_SEED,
    "n_patients": int(
        X.shape[0]
    ),
    "n_gene_universe": int(
        X.shape[1]
    ),
    "n_selected_genes_per_fold": int(
        effective_k
    ),
    "n_splits": N_SPLITS,
    "n_repeats": N_REPEATS,
    "n_total_folds": int(
        len(cv_splits)
    ),
    "top_genes_per_patient": (
        TOP_GENES_PER_PATIENT
    ),
    "save_full_shap_matrices": (
        SAVE_FULL_SHAP_MATRICES
    )
}

with open(
    ATTRIBUTION_DIR
    / "shap_run_manifest.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        shap_manifest,
        file,
        indent=2
    )


# ============================================================
# 18. FINAL REPORT
# ============================================================

total_minutes = (
    time.time()
    - overall_start
) / 60

print("\n" + "=" * 78)
print("AIDO-BBA HELD-OUT SHAP COMPLETED")
print("=" * 78)

print(
    "New folds completed :",
    completed_fold_count
)

print(
    "Existing folds skipped:",
    skipped_fold_count
)

print(
    "Total duration      :",
    round(
        total_minutes,
        2
    ),
    "minutes"
)

print(
    "Fold audit rows     :",
    len(
        shap_fold_audit
    )
)

print(
    "Patient top-gene rows:",
    len(
        shap_patient_top_genes
    )
)

print(
    "Global fold-gene rows:",
    len(
        shap_global_by_fold
    )
)

print(
    "Patient-gene stability rows:",
    len(
        patient_gene_stability
    )
)

print("\nAdditivity audit summary:")

additivity_summary = pd.DataFrame([
    {
        "metric":
            "mean_fold_mean_absolute_error",

        "value":
            shap_fold_audit[
                "mean_absolute_additivity_error"
            ].mean()
    },
    {
        "metric":
            "maximum_fold_maximum_absolute_error",

        "value":
            shap_fold_audit[
                "maximum_absolute_additivity_error"
            ].max()
    },
    {
        "metric":
            "mean_reconstruction_correlation",

        "value":
            shap_fold_audit[
                "correlation_reconstructed_vs_probability"
            ].mean()
    },
    {
        "metric":
            "minimum_reconstruction_correlation",

        "value":
            shap_fold_audit[
                "correlation_reconstructed_vs_probability"
            ].min()
    }
])

display(
    additivity_summary
)

print("\nTop 30 global SHAP genes:")

display(
    shap_global_gene_stability.head(30)
)

print("\nOutput directory:")
print(ATTRIBUTION_DIR)

logger.info(
    "Complete restore and SHAP run finished."
)