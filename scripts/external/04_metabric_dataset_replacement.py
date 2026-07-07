
# %% [CELL 1] ============================================================
# AIDO-BBA BRCA 1.0
# PART C ONLY — METABRIC DATASET-REPLACEMENT STRESS TEST
#
# Purpose
# -------
# 1. Load METABRIC expression and clinical stage data.
# 2. Match samples robustly.
# 3. Rebuild an ExtraTrees stage classifier under repeated CV.
# 4. Audit fold-level and patient-level performance.
# 5. Test overlap with BRCA representation-gap and stable-core genes.
# 6. Project the three BRCA stable cores into METABRIC.
# 7. Save all outputs and a final manifest.
#
# Interpretation boundary
# -----------------------
# This is a dataset-replacement stress test, not definitive external
# validation. It evaluates transferability, data sensitivity, and
# representation dependence under a compatible external dataset.
# ============================================================

from pathlib import Path
from aido_bba.config import (
    data_root, brca_output_root, brca_run_dir, go_bp_gmt,
    hgnc_file, ncbi_gene_info, metabric_dir, gse96058_dir,
    kirc_dir, kirc_output_root,
)

from collections import Counter
import json
import math
import re
import time
import warnings

import numpy as np
import pandas as pd

from scipy.stats import mannwhitneyu
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
)
from sklearn.model_selection import RepeatedStratifiedKFold

warnings.filterwarnings("ignore")

print("=" * 88)
print("AIDO-BBA BRCA 1.0 — PART C METABRIC STRESS TEST")
print("=" * 88)


# %% [CELL 2] ============================================================
# PATHS AND SETTINGS
# ============================================================

OUTPUT_ROOT = brca_output_root()

METABRIC_GE_PATH = metabric_dir() / "data_mrna_illumina_microarray.txt"

METABRIC_CLINICAL_PATH = metabric_dir() / "brca_metabric_clinical_data.tsv"

N_SPLITS = 5
N_REPEATS = 5
TOP_K_GENES = 1500
N_TREES = 400
RANDOM_SEED = 20260701

# Minimum external sample requirement.
MIN_MATCHED_SAMPLES = 100

# AUC is orientation-corrected only for reporting if needed.
ORIENTATION_CORRECT_AUC = True

print("\nExpression path:")
print(METABRIC_GE_PATH)

print("\nClinical path:")
print(METABRIC_CLINICAL_PATH)


# %% [CELL 3] ============================================================
# DISCOVER COMPLETED AIDO-BBA RUN
# ============================================================

required_relative_paths = [
    Path(
        "11_representation_gap_genes_corrected"
        "/summaries/global_representation_gap_gene_stability.tsv"
    ),
    Path(
        "13_gap_module_cores"
        "/summaries/stable_core_gene_manifest.tsv"
    ),
]

candidate_runs = []

for run_dir in OUTPUT_ROOT.iterdir():

    if not run_dir.is_dir():
        continue

    if all(
        (run_dir / relative_path).exists()
        for relative_path
        in required_relative_paths
    ):
        candidate_runs.append(
            run_dir
        )

if len(candidate_runs) == 0:
    raise FileNotFoundError(
        "No completed AIDO-BBA run containing representation-gap "
        "and stable-core outputs was found."
    )

RUN_DIR = sorted(
    candidate_runs,
    key=lambda path: path.stat().st_mtime,
    reverse=True,
)[0]

STRESS_DIR = (
    RUN_DIR
    / "20_metabric_dataset_replacement_stress"
)

SUMMARY_DIR = (
    STRESS_DIR
    / "summaries"
)

REPORT_DIR = (
    STRESS_DIR
    / "reports"
)

for directory in [
    STRESS_DIR,
    SUMMARY_DIR,
    REPORT_DIR,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

print("\nSelected AIDO-BBA run:")
print(RUN_DIR)

print("\nOutput directory:")
print(STRESS_DIR)


# %% [CELL 4] ============================================================
# BASIC FILE CHECKS
# ============================================================

missing_files = []

for path in [
    METABRIC_GE_PATH,
    METABRIC_CLINICAL_PATH,
]:
    if not path.exists():
        missing_files.append(
            str(path)
        )

if missing_files:
    raise FileNotFoundError(
        "Required METABRIC files were not found:\n"
        + "\n".join(
            missing_files
        )
    )

print("\nInput files found successfully.")


# %% [CELL 5] ============================================================
# UTILITY FUNCTIONS
# ============================================================

def first_existing_column(
    dataframe,
    candidates,
    required=True,
):
    normalized_lookup = {
        str(column).strip().lower():
            column
        for column in dataframe.columns
    }

    for candidate in candidates:
        candidate_lower = str(candidate).strip().lower()

        if candidate_lower in normalized_lookup:
            return normalized_lookup[
                candidate_lower
            ]

    if required:
        raise KeyError(
            "None of the expected columns were found:\n"
            + "\n".join(
                str(candidate)
                for candidate in candidates
            )
        )

    return None


def read_table_flexible(
    path,
):
    path = Path(path)

    attempts = [
        {
            "sep": "\t",
            "comment": "#",
        },
        {
            "sep": "\t",
            "comment": None,
        },
        {
            "sep": ",",
            "comment": "#",
        },
        {
            "sep": ",",
            "comment": None,
        },
    ]

    errors = []

    for parameters in attempts:

        try:
            dataframe = pd.read_csv(
                path,
                sep=parameters["sep"],
                comment=parameters["comment"],
                low_memory=False,
            )

            if dataframe.shape[1] > 1:
                return dataframe

        except Exception as error:
            errors.append(
                str(error)
            )

    raise ValueError(
        f"Unable to parse table: {path}\n"
        + "\n".join(
            errors
        )
    )


def normalize_sample_id(
    value,
):
    if pd.isna(value):
        return None

    text = str(value).strip()

    if text == "":
        return None

    return text


def normalize_gene_symbol(
    value,
):
    if pd.isna(value):
        return None

    text = str(value).strip()

    if text == "":
        return None

    return text


def stage_to_binary(
    value,
):
    if pd.isna(value):
        return np.nan

    text = str(value).strip().upper()

    if text in {
        "",
        "NA",
        "NAN",
        "NONE",
        "NULL",
        "NOT AVAILABLE",
        "NOT REPORTED",
        "UNKNOWN",
    }:
        return np.nan

    # Normalize punctuation and common prefixes.
    text = (
        text
        .replace("PATHOLOGIC", "")
        .replace("PATHOLOGICAL", "")
        .replace("CLINICAL", "")
        .replace("TUMOR", "")
        .replace("STAGE", "")
        .replace("_", " ")
        .replace("-", " ")
        .strip()
    )

    # Roman-numeral parsing.
    if re.search(
        r"\bIV[A-C]?\b",
        text,
    ):
        return 1

    if re.search(
        r"\bIII[A-C]?\b",
        text,
    ):
        return 1

    if re.search(
        r"\bII[A-C]?\b",
        text,
    ):
        return 0

    if re.search(
        r"\bI[A-C]?\b",
        text,
    ):
        return 0

    # Numeric parsing.
    numeric_tokens = re.findall(
        r"\d+",
        text,
    )

    if numeric_tokens:
        stage_number = int(
            numeric_tokens[0]
        )

        if stage_number in {
            1,
            2,
        }:
            return 0

        if stage_number in {
            3,
            4,
        }:
            return 1

    return np.nan


def orientation_corrected_auc(
    y_true,
    probability,
):
    raw_auc = roc_auc_score(
        y_true,
        probability,
    )

    if ORIENTATION_CORRECT_AUC:
        corrected_auc = max(
            raw_auc,
            1.0 - raw_auc,
        )
    else:
        corrected_auc = raw_auc

    return raw_auc, corrected_auc


def cliffs_delta_from_mannwhitney(
    group_a,
    group_b,
):
    group_a = np.asarray(
        group_a,
        dtype=float,
    )

    group_b = np.asarray(
        group_b,
        dtype=float,
    )

    if (
        len(group_a) == 0
        or
        len(group_b) == 0
    ):
        return np.nan

    u_statistic, _ = mannwhitneyu(
        group_a,
        group_b,
        alternative="two-sided",
    )

    return (
        2.0
        *
        u_statistic
        /
        (
            len(group_a)
            *
            len(group_b)
        )
        -
        1.0
    )


# %% [CELL 6] ============================================================
# LOAD METABRIC EXPRESSION
# ============================================================

expression_raw = read_table_flexible(
    METABRIC_GE_PATH
)

print("\nRaw expression shape:")
print(
    expression_raw.shape
)

print("\nFirst expression columns:")
print(
    expression_raw.columns.tolist()[:12]
)

gene_column = first_existing_column(
    expression_raw,
    [
        "Hugo_Symbol",
        "Gene Symbol",
        "gene_symbol",
        "symbol",
        "gene",
        "Gene",
    ],
    required=False,
)

entrez_column = first_existing_column(
    expression_raw,
    [
        "Entrez_Gene_Id",
        "Entrez Gene ID",
        "entrez_gene_id",
    ],
    required=False,
)

if gene_column is None:
    gene_column = expression_raw.columns[0]

metadata_columns = {
    gene_column,
}

if entrez_column is not None:
    metadata_columns.add(
        entrez_column
    )

sample_columns = [
    column
    for column in expression_raw.columns
    if column not in metadata_columns
]

expression_numeric = (
    expression_raw[
        sample_columns
    ]
    .apply(
        pd.to_numeric,
        errors="coerce",
    )
)

valid_sample_columns = [
    column
    for column in expression_numeric.columns
    if expression_numeric[
        column
    ].notna().sum() > 0
]

expression_numeric = expression_numeric[
    valid_sample_columns
]

gene_symbols = (
    expression_raw[
        gene_column
    ]
    .map(
        normalize_gene_symbol
    )
)

expression_gene_by_sample = (
    expression_numeric.copy()
)

expression_gene_by_sample.index = (
    gene_symbols
)

expression_gene_by_sample = (
    expression_gene_by_sample[
        expression_gene_by_sample.index.notna()
    ]
)

# Aggregate duplicate gene symbols by mean.
expression_gene_by_sample = (
    expression_gene_by_sample
    .groupby(
        expression_gene_by_sample.index
    )
    .mean()
)

expression_gene_by_sample.columns = [
    normalize_sample_id(
        column
    )
    for column in expression_gene_by_sample.columns
]

expression_gene_by_sample = (
    expression_gene_by_sample.loc[
        :,
        [
            column
            for column in expression_gene_by_sample.columns
            if column is not None
        ]
    ]
)

print("\nProcessed expression shape (genes × samples):")
print(
    expression_gene_by_sample.shape
)

print("\nGene count:")
print(
    expression_gene_by_sample.shape[0]
)

print("Sample count:")
print(
    expression_gene_by_sample.shape[1]
)


# %% [CELL 7] ============================================================
# LOAD METABRIC CLINICAL DATA
# ============================================================

clinical_raw = read_table_flexible(
    METABRIC_CLINICAL_PATH
)

print("\nRaw clinical shape:")
print(
    clinical_raw.shape
)

print("\nClinical columns:")
print(
    clinical_raw.columns.tolist()
)

sample_column = first_existing_column(
    clinical_raw,
    [
        "SAMPLE_ID",
        "PATIENT_ID",
        "sample_id",
        "patient_id",
        "Sample ID",
        "Patient ID",
    ],
)

stage_column = first_existing_column(
    clinical_raw,
    [
        "TUMOR_STAGE",
        "STAGE",
        "PATHOLOGIC_STAGE",
        "CLINICAL_STAGE",
        "tumor_stage",
        "stage",
        "pathologic_stage",
        "clinical_stage",
    ],
    required=False,
)

if stage_column is None:

    stage_like_columns = [
        column
        for column in clinical_raw.columns
        if "stage" in str(column).lower()
    ]

    if len(stage_like_columns) == 0:
        raise KeyError(
            "No stage-like clinical column was found."
        )

    stage_column = stage_like_columns[0]

clinical_stage = (
    clinical_raw[
        [
            sample_column,
            stage_column,
        ]
    ]
    .copy()
)

clinical_stage.columns = [
    "sample_id",
    "stage_raw",
]

clinical_stage[
    "sample_id"
] = clinical_stage[
    "sample_id"
].map(
    normalize_sample_id
)

clinical_stage[
    "stage_binary"
] = clinical_stage[
    "stage_raw"
].map(
    stage_to_binary
)

clinical_stage = (
    clinical_stage
    .dropna(
        subset=[
            "sample_id",
            "stage_binary",
        ]
    )
    .drop_duplicates(
        "sample_id"
    )
)

clinical_stage[
    "stage_binary"
] = clinical_stage[
    "stage_binary"
].astype(int)

print("\nParsed stage rows:")
print(
    clinical_stage.shape
)

print("\nStage counts:")
display(
    clinical_stage[
        "stage_binary"
    ]
    .value_counts()
    .rename_axis(
        "stage_binary"
    )
    .reset_index(
        name="n_samples"
    )
)


# %% [CELL 8] ============================================================
# SAMPLE MATCHING AUDIT
# ============================================================

expression_sample_ids = set(
    expression_gene_by_sample.columns.astype(str)
)

clinical_sample_ids = set(
    clinical_stage[
        "sample_id"
    ].astype(str)
)

matched_samples = sorted(
    expression_sample_ids
    &
    clinical_sample_ids
)

if len(matched_samples) < MIN_MATCHED_SAMPLES:

    # Try common METABRIC sample-ID normalization:
    # remove trailing aliquot suffixes and spaces.
    expression_map = {
        str(sample).strip().replace(" ", ""):
            str(sample)
        for sample in expression_gene_by_sample.columns
    }

    clinical_stage[
        "sample_id_normalized"
    ] = (
        clinical_stage[
            "sample_id"
        ]
        .astype(str)
        .str.strip()
        .str.replace(
            " ",
            "",
            regex=False,
        )
    )

    normalized_matches = sorted(
        set(
            expression_map.keys()
        )
        &
        set(
            clinical_stage[
                "sample_id_normalized"
            ]
        )
    )

    if len(normalized_matches) > len(
        matched_samples
    ):
        matched_expression_samples = [
            expression_map[
                sample
            ]
            for sample in normalized_matches
        ]

        matched_clinical = (
            clinical_stage
            .set_index(
                "sample_id_normalized"
            )
            .loc[
                normalized_matches
            ]
            .copy()
        )

        matched_clinical[
            "expression_sample_id"
        ] = matched_expression_samples

    else:
        matched_clinical = (
            clinical_stage
            .set_index(
                "sample_id"
            )
            .loc[
                matched_samples
            ]
            .copy()
        )

        matched_clinical[
            "expression_sample_id"
        ] = matched_samples

else:
    matched_clinical = (
        clinical_stage
        .set_index(
            "sample_id"
        )
        .loc[
            matched_samples
        ]
        .copy()
    )

    matched_clinical[
        "expression_sample_id"
    ] = matched_samples

if len(matched_clinical) < MIN_MATCHED_SAMPLES:
    raise ValueError(
        "Fewer than "
        f"{MIN_MATCHED_SAMPLES} METABRIC samples matched "
        "between expression and clinical stage data."
    )

matched_clinical = (
    matched_clinical.reset_index()
)

matched_expression_samples = (
    matched_clinical[
        "expression_sample_id"
    ]
    .astype(str)
    .tolist()
)

matched_sample_ids = (
    matched_clinical[
        "sample_id"
    ]
    .astype(str)
    .tolist()
)

y = (
    matched_clinical[
        "stage_binary"
    ]
    .to_numpy(
        dtype=int
    )
)

print("\nMatched samples:")
print(
    len(
        matched_expression_samples
    )
)

print("\nMatched class counts:")
display(
    pd.Series(
        y
    )
    .value_counts()
    .rename_axis(
        "stage_binary"
    )
    .reset_index(
        name="n_samples"
    )
)

sample_match_summary = pd.DataFrame([
    {
        "metric":
            "expression_samples",
        "value":
            len(
                expression_sample_ids
            ),
    },
    {
        "metric":
            "clinical_stage_samples",
        "value":
            len(
                clinical_sample_ids
            ),
    },
    {
        "metric":
            "matched_samples",
        "value":
            len(
                matched_expression_samples
            ),
    },
    {
        "metric":
            "early_samples",
        "value":
            int(
                np.sum(
                    y == 0
                )
            ),
    },
    {
        "metric":
            "advanced_samples",
        "value":
            int(
                np.sum(
                    y == 1
                )
            ),
    },
])


# %% [CELL 9] ============================================================
# PREPARE MODEL MATRIX
# ============================================================

X_gene_by_sample = (
    expression_gene_by_sample.loc[
        :,
        matched_expression_samples,
    ]
    .copy()
)

# Remove genes with excessive missingness.
minimum_nonmissing = max(
    10,
    int(
        0.80
        *
        X_gene_by_sample.shape[1]
    ),
)

X_gene_by_sample = (
    X_gene_by_sample[
        X_gene_by_sample.notna().sum(
            axis=1
        )
        >=
        minimum_nonmissing
    ]
)

# Median imputation by gene.
gene_medians = (
    X_gene_by_sample.median(
        axis=1
    )
)

X_gene_by_sample = (
    X_gene_by_sample.T
    .fillna(
        gene_medians
    )
    .T
)

# Remove zero-variance genes.
gene_variance = (
    X_gene_by_sample.var(
        axis=1
    )
)

X_gene_by_sample = (
    X_gene_by_sample.loc[
        gene_variance > 0,
        :
    ]
)

gene_ids = (
    X_gene_by_sample.index
    .astype(str)
    .to_numpy()
)

X = (
    X_gene_by_sample.T
    .to_numpy(
        dtype=float
    )
)

print("\nFinal model matrix:")
print(
    X.shape
)

print("Final genes:")
print(
    len(
        gene_ids
    )
)


# %% [CELL 10] ============================================================
# REPEATED CROSS-VALIDATION
# ============================================================

cv = RepeatedStratifiedKFold(
    n_splits=N_SPLITS,
    n_repeats=N_REPEATS,
    random_state=RANDOM_SEED,
)

n_total_folds = (
    N_SPLITS
    *
    N_REPEATS
)

fold_records = []

patient_probability_sum = np.zeros(
    len(
        matched_expression_samples
    ),
    dtype=float,
)

patient_probability_count = np.zeros(
    len(
        matched_expression_samples
    ),
    dtype=int,
)

selected_gene_counter = Counter()

start_time = time.time()

for fold_number, (
    train_index,
    test_index,
) in enumerate(
    cv.split(
        X,
        y,
    ),
    start=1,
):
    selector = SelectKBest(
        score_func=f_classif,
        k=min(
            TOP_K_GENES,
            X.shape[1],
        ),
    )

    X_train_selected = (
        selector.fit_transform(
            X[
                train_index
            ],
            y[
                train_index
            ],
        )
    )

    X_test_selected = (
        selector.transform(
            X[
                test_index
            ]
        )
    )

    selected_mask = (
        selector.get_support()
    )

    selected_genes = (
        gene_ids[
            selected_mask
        ]
    )

    selected_gene_counter.update(
        selected_genes.tolist()
    )

    model = ExtraTreesClassifier(
        n_estimators=N_TREES,
        random_state=(
            RANDOM_SEED
            +
            fold_number
        ),
        class_weight="balanced",
        n_jobs=-1,
    )

    model.fit(
        X_train_selected,
        y[
            train_index
        ],
    )

    probability = (
        model.predict_proba(
            X_test_selected
        )[:, 1]
    )

    raw_auc, corrected_auc = (
        orientation_corrected_auc(
            y[
                test_index
            ],
            probability,
        )
    )

    average_precision = (
        average_precision_score(
            y[
                test_index
            ],
            probability,
        )
    )

    predicted_label = (
        probability >= 0.50
    ).astype(int)

    balanced_accuracy = (
        balanced_accuracy_score(
            y[
                test_index
            ],
            predicted_label,
        )
    )

    patient_probability_sum[
        test_index
    ] += probability

    patient_probability_count[
        test_index
    ] += 1

    fold_records.append({
        "fold_number":
            fold_number,
        "n_train":
            len(
                train_index
            ),
        "n_test":
            len(
                test_index
            ),
        "n_selected_genes":
            len(
                selected_genes
            ),
        "raw_auc":
            raw_auc,
        "orientation_corrected_auc":
            corrected_auc,
        "average_precision":
            average_precision,
        "balanced_accuracy_at_0_50":
            balanced_accuracy,
    })

    if (
        fold_number == 1
        or
        fold_number % 5 == 0
        or
        fold_number
        ==
        n_total_folds
    ):
        elapsed_minutes = (
            time.time()
            -
            start_time
        ) / 60.0

        print(
            f"Fold {fold_number:>2}/"
            f"{n_total_folds} | "
            f"AUC={corrected_auc:.4f} | "
            f"{elapsed_minutes:.2f} min"
        )

fold_results = pd.DataFrame(
    fold_records
)


# %% [CELL 11] ============================================================
# PATIENT-LEVEL OOF PERFORMANCE
# ============================================================

mean_oof_probability = np.divide(
    patient_probability_sum,
    patient_probability_count,
    out=np.full_like(
        patient_probability_sum,
        np.nan,
    ),
    where=(
        patient_probability_count
        >
        0
    ),
)

raw_oof_auc, corrected_oof_auc = (
    orientation_corrected_auc(
        y,
        mean_oof_probability,
    )
)

oof_average_precision = (
    average_precision_score(
        y,
        mean_oof_probability,
    )
)

oof_predicted_label = (
    mean_oof_probability
    >=
    0.50
).astype(int)

oof_balanced_accuracy = (
    balanced_accuracy_score(
        y,
        oof_predicted_label,
    )
)

oof_confusion_matrix = confusion_matrix(
    y,
    oof_predicted_label,
)

patient_predictions = pd.DataFrame({
    "sample_id":
        matched_sample_ids,
    "expression_sample_id":
        matched_expression_samples,
    "true_stage_binary":
        y,
    "true_stage_group":
        np.where(
            y == 1,
            "Advanced",
            "Early",
        ),
    "mean_oof_probability_advanced":
        mean_oof_probability,
    "n_oof_predictions":
        patient_probability_count,
    "predicted_stage_binary_at_0_50":
        oof_predicted_label,
})

performance_summary = pd.DataFrame([
    {
        "metric":
            "n_matched_samples",
        "value":
            len(
                y
            ),
    },
    {
        "metric":
            "n_early",
        "value":
            int(
                np.sum(
                    y == 0
                )
            ),
    },
    {
        "metric":
            "n_advanced",
        "value":
            int(
                np.sum(
                    y == 1
                )
            ),
    },
    {
        "metric":
            "n_model_genes",
        "value":
            X.shape[1],
    },
    {
        "metric":
            "mean_fold_raw_auc",
        "value":
            fold_results[
                "raw_auc"
            ].mean(),
    },
    {
        "metric":
            "sd_fold_raw_auc",
        "value":
            fold_results[
                "raw_auc"
            ].std(),
    },
    {
        "metric":
            "mean_fold_orientation_corrected_auc",
        "value":
            fold_results[
                "orientation_corrected_auc"
            ].mean(),
    },
    {
        "metric":
            "sd_fold_orientation_corrected_auc",
        "value":
            fold_results[
                "orientation_corrected_auc"
            ].std(),
    },
    {
        "metric":
            "patient_mean_oof_raw_auc",
        "value":
            raw_oof_auc,
    },
    {
        "metric":
            "patient_mean_oof_orientation_corrected_auc",
        "value":
            corrected_oof_auc,
    },
    {
        "metric":
            "patient_mean_oof_average_precision",
        "value":
            oof_average_precision,
    },
    {
        "metric":
            "patient_mean_oof_balanced_accuracy_at_0_50",
        "value":
            oof_balanced_accuracy,
    },
])

print("\nPerformance summary:")
display(
    performance_summary
)

print("\nOOF confusion matrix:")
display(
    pd.DataFrame(
        oof_confusion_matrix,
        index=[
            "True_Early",
            "True_Advanced",
        ],
        columns=[
            "Pred_Early",
            "Pred_Advanced",
        ],
    )
)


# %% [CELL 12] ============================================================
# SELECTED-GENE RECURRENCE
# ============================================================

selected_gene_frequency = pd.DataFrame([
    {
        "gene_id":
            gene,
        "n_folds_selected":
            count,
        "selection_frequency":
            count
            /
            n_total_folds,
    }
    for gene, count
    in selected_gene_counter.items()
])

selected_gene_frequency = (
    selected_gene_frequency
    .sort_values(
        [
            "selection_frequency",
            "gene_id",
        ],
        ascending=[
            False,
            True,
        ],
    )
    .reset_index(
        drop=True
    )
)

print("\nMost recurrent selected genes:")
display(
    selected_gene_frequency.head(
        40
    )
)


# %% [CELL 13] ============================================================
# LOAD BRCA GAP-GENE AND STABLE-CORE REFERENCES
# ============================================================

gap_gene_path = (
    RUN_DIR
    / "11_representation_gap_genes_corrected"
    / "summaries"
    / "global_representation_gap_gene_stability.tsv"
)

core_manifest_path = (
    RUN_DIR
    / "13_gap_module_cores"
    / "summaries"
    / "stable_core_gene_manifest.tsv"
)

brca_gap_genes = pd.read_csv(
    gap_gene_path,
    sep="\t",
)

core_manifest = pd.read_csv(
    core_manifest_path,
    sep="\t",
)

brca_gap_genes[
    "raw_gene_id"
] = brca_gap_genes[
    "raw_gene_id"
].astype(str)

core_manifest[
    "gene_id"
] = core_manifest[
    "gene_id"
].astype(str)

metabric_selected_gene_set = set(
    selected_gene_frequency[
        "gene_id"
    ].astype(str)
)

brca_gap_gene_set = set(
    brca_gap_genes[
        "raw_gene_id"
    ].astype(str)
)

stable_core_gene_set = set(
    core_manifest[
        "gene_id"
    ].astype(str)
)

gap_overlap = sorted(
    metabric_selected_gene_set
    &
    brca_gap_gene_set
)

core_overlap = sorted(
    metabric_selected_gene_set
    &
    stable_core_gene_set
)

gene_overlap_summary = pd.DataFrame([
    {
        "metric":
            "n_metabric_selected_genes",
        "value":
            len(
                metabric_selected_gene_set
            ),
    },
    {
        "metric":
            "n_brca_gap_genes",
        "value":
            len(
                brca_gap_gene_set
            ),
    },
    {
        "metric":
            "n_stable_core_genes",
        "value":
            len(
                stable_core_gene_set
            ),
    },
    {
        "metric":
            "n_metabric_brca_gap_overlap",
        "value":
            len(
                gap_overlap
            ),
    },
    {
        "metric":
            "n_metabric_stable_core_overlap",
        "value":
            len(
                core_overlap
            ),
    },
    {
        "metric":
            "brca_gap_overlap_fraction_of_brca_gap_genes",
        "value":
            (
                len(
                    gap_overlap
                )
                /
                max(
                    1,
                    len(
                        brca_gap_gene_set
                    )
                )
            ),
    },
    {
        "metric":
            "stable_core_overlap_fraction_of_core_genes",
        "value":
            (
                len(
                    core_overlap
                )
                /
                max(
                    1,
                    len(
                        stable_core_gene_set
                    )
                )
            ),
    },
])

gap_overlap_table = pd.DataFrame({
    "gene_id":
        gap_overlap
})

core_overlap_table = pd.DataFrame({
    "gene_id":
        core_overlap
})

print("\nGene-overlap summary:")
display(
    gene_overlap_summary
)


# %% [CELL 14] ============================================================
# PROJECT STABLE CORES INTO METABRIC
# ============================================================

metabric_gene_set = set(
    X_gene_by_sample.index.astype(str)
)

core_projection_records = []

for core_name, core_df in (
    core_manifest.groupby(
        "core_module_name"
    )
):
    core_genes = set(
        core_df[
            "gene_id"
        ].astype(str)
    )

    matched_core_genes = sorted(
        core_genes
        &
        metabric_gene_set
    )

    if len(
        matched_core_genes
    ) == 0:
        core_projection_records.append({
            "core_module_name":
                core_name,
            "n_core_genes_total":
                len(
                    core_genes
                ),
            "n_core_genes_matched":
                0,
            "matched_fraction":
                0.0,
            "mean_early":
                np.nan,
            "mean_advanced":
                np.nan,
            "mean_difference_advanced_minus_early":
                np.nan,
            "cliffs_delta_early_minus_advanced":
                np.nan,
            "mannwhitney_p_value":
                np.nan,
        })

        continue

    core_expression = (
        X_gene_by_sample.loc[
            matched_core_genes,
            matched_expression_samples,
        ]
        .copy()
    )

    gene_mean = core_expression.mean(
        axis=1
    )

    gene_sd = (
        core_expression.std(
            axis=1
        )
        .replace(
            0,
            np.nan,
        )
    )

    core_z = (
        core_expression
        .sub(
            gene_mean,
            axis=0,
        )
        .div(
            gene_sd,
            axis=0,
        )
    )

    core_score = (
        core_z.mean(
            axis=0
        )
    )

    early_values = (
        core_score.to_numpy()[
            y == 0
        ]
    )

    advanced_values = (
        core_score.to_numpy()[
            y == 1
        ]
    )

    u_statistic, p_value = (
        mannwhitneyu(
            early_values,
            advanced_values,
            alternative="two-sided",
        )
    )

    cliffs_delta = (
        cliffs_delta_from_mannwhitney(
            early_values,
            advanced_values,
        )
    )

    core_projection_records.append({
        "core_module_name":
            core_name,
        "n_core_genes_total":
            len(
                core_genes
            ),
        "n_core_genes_matched":
            len(
                matched_core_genes
            ),
        "matched_fraction":
            len(
                matched_core_genes
            )
            /
            len(
                core_genes
            ),
        "mean_early":
            float(
                np.mean(
                    early_values
                )
            ),
        "mean_advanced":
            float(
                np.mean(
                    advanced_values
                )
            ),
        "mean_difference_advanced_minus_early":
            float(
                np.mean(
                    advanced_values
                )
                -
                np.mean(
                    early_values
                )
            ),
        "cliffs_delta_early_minus_advanced":
            float(
                cliffs_delta
            ),
        "mannwhitney_p_value":
            float(
                p_value
            ),
    })

core_projection = pd.DataFrame(
    core_projection_records
)

print("\nStable-core projection:")
display(
    core_projection
)


# %% [CELL 15] ============================================================
# SAVE ALL OUTPUTS
# ============================================================

sample_match_summary.to_csv(
    SUMMARY_DIR
    / "metabric_sample_match_summary.tsv",
    sep="\t",
    index=False,
)

clinical_stage.to_csv(
    SUMMARY_DIR
    / "metabric_parsed_stage_labels.tsv",
    sep="\t",
    index=False,
)

fold_results.to_csv(
    SUMMARY_DIR
    / "metabric_fold_results.tsv",
    sep="\t",
    index=False,
)

patient_predictions.to_csv(
    SUMMARY_DIR
    / "metabric_patient_oof_predictions.tsv",
    sep="\t",
    index=False,
)

performance_summary.to_csv(
    SUMMARY_DIR
    / "metabric_performance_summary.tsv",
    sep="\t",
    index=False,
)

selected_gene_frequency.to_csv(
    SUMMARY_DIR
    / "metabric_selected_gene_frequency.tsv",
    sep="\t",
    index=False,
)

gene_overlap_summary.to_csv(
    SUMMARY_DIR
    / "metabric_brca_gene_overlap_summary.tsv",
    sep="\t",
    index=False,
)

gap_overlap_table.to_csv(
    SUMMARY_DIR
    / "metabric_brca_gap_gene_overlap.tsv",
    sep="\t",
    index=False,
)

core_overlap_table.to_csv(
    SUMMARY_DIR
    / "metabric_stable_core_gene_overlap.tsv",
    sep="\t",
    index=False,
)

core_projection.to_csv(
    SUMMARY_DIR
    / "metabric_stable_core_projection.tsv",
    sep="\t",
    index=False,
)

final_summary = pd.concat(
    [
        sample_match_summary,
        performance_summary,
        gene_overlap_summary,
    ],
    ignore_index=True,
)

final_summary.to_csv(
    SUMMARY_DIR
    / "metabric_dataset_replacement_stress_summary.tsv",
    sep="\t",
    index=False,
)

manifest = {
    "analysis":
        "AIDO-BBA BRCA METABRIC dataset-replacement stress test",
    "run_directory":
        str(
            RUN_DIR
        ),
    "expression_path":
        str(
            METABRIC_GE_PATH
        ),
    "clinical_path":
        str(
            METABRIC_CLINICAL_PATH
        ),
    "n_splits":
        N_SPLITS,
    "n_repeats":
        N_REPEATS,
    "top_k_genes":
        TOP_K_GENES,
    "n_trees":
        N_TREES,
    "random_seed":
        RANDOM_SEED,
    "orientation_correct_auc":
        ORIENTATION_CORRECT_AUC,
    "interpretation_boundary":
        (
            "This analysis is a dataset-replacement stress test, "
            "not definitive external validation. Outputs assess "
            "transferability, data sensitivity, and stable-core "
            "projection under METABRIC."
        ),
}

with open(
    STRESS_DIR
    / "metabric_dataset_replacement_stress_manifest.json",
    "w",
    encoding="utf-8",
) as handle:
    json.dump(
        manifest,
        handle,
        indent=2,
    )


# %% [CELL 16] ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 88)
print("AIDO-BBA PART C METABRIC STRESS TEST COMPLETED")
print("=" * 88)

print("\nPerformance summary:")
display(
    performance_summary
)

print("\nGene overlap summary:")
display(
    gene_overlap_summary
)

print("\nStable-core projection:")
display(
    core_projection
)

print("\nOutput directory:")
print(
    STRESS_DIR
)
