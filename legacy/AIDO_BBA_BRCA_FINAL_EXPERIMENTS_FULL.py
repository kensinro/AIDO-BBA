
# %% [CELL 1] ============================================================
# AIDO-BBA BRCA 1.0 — FINAL EXPERIMENTS
#
# PART A  Full-pool patient-specific target reconstruction
# PART B  Recommendation null / negative-control audit
# PART C  Optional METABRIC dataset-replacement stress test
#
# Run cells in order.
#
# IMPORTANT
# - Parts A and B use existing AIDO-BBA outputs automatically.
# - Part C runs only when compatible METABRIC files are found or
#   explicitly provided in CELL 2.
# - Outputs are written under the selected AIDO-BBA run directory.
# ============================================================

from pathlib import Path
from collections import defaultdict
from itertools import combinations
import json
import math
import re
import time
import warnings

import numpy as np
import pandas as pd

from scipy.stats import spearmanr, mannwhitneyu
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

print("=" * 88)
print("AIDO-BBA BRCA 1.0 — FINAL EXPERIMENTS")
print("=" * 88)


# %% [CELL 2] ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_ROOT = Path(r"D:\AIDO-Temp\AIDO_BBA_BRCA_1_0")

# ---------- PART A ----------
MIN_FOLD_SELECTION_FREQUENCY = 0.20
MIN_REPEAT_SELECTION_FREQUENCY = 0.40
MIN_SIGN_CONSISTENCY = 0.60

GENERIC_FREQUENCY_THRESHOLD = 0.25
RECURRENT_FREQUENCY_THRESHOLD = 0.10

TOP_GENERIC_PER_PATIENT = 5
TOP_RECURRENT_PER_PATIENT = 5
TOP_PATIENT_SPECIFIC_PER_PATIENT = 10
TOP_TOTAL_PER_PATIENT = 20

# Score weights sum to 1.
W_PATIENT_ATTRIBUTION = 0.35
W_PATIENT_PERCENTILE = 0.20
W_GLOBAL_STABILITY = 0.20
W_CORE_STABILITY = 0.10
W_SPECIFICITY = 0.15

# ---------- PART B ----------
N_NULL_PERMUTATIONS = 100
NULL_RANDOM_SEED = 20260701
NULL_TOP_K = 20

# ---------- PART C ----------
RUN_METABRIC_STRESS_IF_AVAILABLE = True

# Set these explicitly if auto-detection fails.
METABRIC_GE_PATH = None
METABRIC_CLINICAL_PATH = None

# Common optional candidates.
METABRIC_GE_CANDIDATES = [
    Path(r"D:\AIDO-Data\METABRIC\GE.tsv"),
    Path(r"D:\AIDO-Data\METABRIC\expression.tsv"),
    Path(r"D:\AIDO-Data\METABRIC\data_expression_median.txt"),
    Path(r"D:\AIDO-Data\cBioPortal\brca_metabric\data_mrna_illumina_microarray.txt"),
]

METABRIC_CLINICAL_CANDIDATES = [
    Path(r"D:\AIDO-Data\METABRIC\clinical.tsv"),
    Path(r"D:\AIDO-Data\METABRIC\data_clinical_patient.txt"),
    Path(r"D:\AIDO-Data\cBioPortal\brca_metabric\data_clinical_patient.txt"),
]

METABRIC_N_SPLITS = 5
METABRIC_N_REPEATS = 5
METABRIC_TOP_K_GENES = 1500
METABRIC_N_TREES = 400
METABRIC_RANDOM_SEED = 20260701


# %% [CELL 3] ============================================================
# DISCOVER COMPLETED AIDO-BBA RUN
# ============================================================

required_relative_paths = [
    Path("11_representation_gap_genes_corrected/summaries/patient_gap_gene_attribution_stability.tsv"),
    Path("11_representation_gap_genes_corrected/summaries/global_representation_gap_gene_stability.tsv"),
    Path("13_gap_module_cores/summaries/stable_core_gene_manifest.tsv"),
    Path("15_fuzzy_anchor_boundary_audit/summaries/patient_anchor_and_boundary_scores.tsv"),
    Path("16_missing_measurement_recommendation/summaries/all_patient_measurement_recommendation_report.tsv"),
]

candidate_runs = []

for run_dir in OUTPUT_ROOT.iterdir():
    if not run_dir.is_dir():
        continue

    if all((run_dir / rel).exists() for rel in required_relative_paths):
        candidate_runs.append(run_dir)

if not candidate_runs:
    raise FileNotFoundError(
        "No completed AIDO-BBA run containing CELL 24–29 outputs was found."
    )

RUN_DIR = sorted(
    candidate_runs,
    key=lambda path: path.stat().st_mtime,
    reverse=True,
)[0]

FULL_POOL_DIR = RUN_DIR / "18_full_pool_target_reconstruction"
NULL_DIR = RUN_DIR / "19_recommendation_null_audit"
STRESS_DIR = RUN_DIR / "20_metabric_dataset_replacement_stress"

for directory in [FULL_POOL_DIR, NULL_DIR, STRESS_DIR]:
    (directory / "summaries").mkdir(parents=True, exist_ok=True)
    (directory / "reports").mkdir(parents=True, exist_ok=True)

print("\nSelected run:")
print(RUN_DIR)


# %% [CELL 4] ============================================================
# UTILITIES
# ============================================================

def first_existing_column(dataframe, candidates, required=True):
    for column in candidates:
        if column in dataframe.columns:
            return column

    if required:
        raise KeyError(
            "None of the expected columns were found:\n"
            + "\n".join(candidates)
        )
    return None


def minmax_scale(values):
    values = pd.Series(values, dtype=float)
    finite = values[np.isfinite(values)]

    if len(finite) == 0:
        return pd.Series(np.nan, index=values.index)

    low = finite.min()
    high = finite.max()

    if high <= low:
        return pd.Series(0.5, index=values.index)

    return (values - low) / (high - low)


def rank_pct_within_group(dataframe, value_column, group_column="patient_id"):
    return (
        dataframe
        .groupby(group_column)[value_column]
        .rank(method="average", pct=True)
    )


def jaccard(set_a, set_b):
    set_a = set(set_a)
    set_b = set(set_b)

    union = set_a | set_b

    if len(union) == 0:
        return np.nan

    return len(set_a & set_b) / len(union)


def empirical_p_greater_equal(observed, null_values):
    null_values = np.asarray(null_values, dtype=float)

    return float(
        (
            1
            + np.sum(null_values >= observed)
        )
        /
        (
            len(null_values)
            + 1
        )
    )


def empirical_p_less_equal(observed, null_values):
    null_values = np.asarray(null_values, dtype=float)

    return float(
        (
            1
            + np.sum(null_values <= observed)
        )
        /
        (
            len(null_values)
            + 1
        )
    )


def clean_symbol(value):
    if pd.isna(value):
        return None

    text = str(value).strip().upper()

    if text in {"", "-", "NA", "NAN", "NONE", "NULL"}:
        return None

    return text


def stage_to_binary(value):
    if pd.isna(value):
        return np.nan

    text = str(value).strip().upper()
    text = (
        text.replace("STAGE", "")
        .replace("PATHOLOGIC", "")
        .replace("CLINICAL", "")
        .strip()
    )

    # Advanced: III/IV.
    if re.search(r"\bIII\b|\bIIIA\b|\bIIIB\b|\bIIIC\b|\bIV\b|\bIVA\b|\bIVB\b", text):
        return 1

    # Early: I/II.
    if re.search(r"\bI\b|\bIA\b|\bIB\b|\bII\b|\bIIA\b|\bIIB\b", text):
        return 0

    # Numeric fallback.
    numeric = re.findall(r"\d+", text)
    if numeric:
        value_num = int(numeric[0])
        if value_num in {1, 2}:
            return 0
        if value_num in {3, 4}:
            return 1

    return np.nan


# %% [CELL 5] ============================================================
# LOAD FULL-POOL INPUTS
# ============================================================

GAP_SUMMARY_DIR = (
    RUN_DIR
    / "11_representation_gap_genes_corrected"
    / "summaries"
)

CORE_SUMMARY_DIR = (
    RUN_DIR
    / "13_gap_module_cores"
    / "summaries"
)

ANCHOR_SUMMARY_DIR = (
    RUN_DIR
    / "15_fuzzy_anchor_boundary_audit"
    / "summaries"
)

RECOMMENDATION_SUMMARY_DIR = (
    RUN_DIR
    / "16_missing_measurement_recommendation"
    / "summaries"
)

patient_gene = pd.read_csv(
    GAP_SUMMARY_DIR
    / "patient_gap_gene_attribution_stability.tsv",
    sep="\t",
)

global_gene = pd.read_csv(
    GAP_SUMMARY_DIR
    / "global_representation_gap_gene_stability.tsv",
    sep="\t",
)

core_manifest = pd.read_csv(
    CORE_SUMMARY_DIR
    / "stable_core_gene_manifest.tsv",
    sep="\t",
)

patient_states = pd.read_csv(
    ANCHOR_SUMMARY_DIR
    / "patient_anchor_and_boundary_scores.tsv",
    sep="\t",
)

all_patient_reports = pd.read_csv(
    RECOMMENDATION_SUMMARY_DIR
    / "all_patient_measurement_recommendation_report.tsv",
    sep="\t",
)

for dataframe in [patient_gene, patient_states, all_patient_reports]:
    dataframe["patient_id"] = dataframe["patient_id"].astype(str)

patient_gene["raw_gene_id"] = patient_gene["raw_gene_id"].astype(str)
global_gene["raw_gene_id"] = global_gene["raw_gene_id"].astype(str)
core_manifest["gene_id"] = core_manifest["gene_id"].astype(str)

print("\nPatient-gene rows:")
print(patient_gene.shape)

print("Patients:")
print(patient_gene["patient_id"].nunique())

print("Unique patient-gap genes:")
print(patient_gene["raw_gene_id"].nunique())


# %% [CELL 6] ============================================================
# BUILD FULL CANDIDATE POOL
# ============================================================

core_gene_lookup = (
    core_manifest[
        [
            "gene_id",
            "harmonized_gene_id",
            "core_module_name",
            "mean_within_consensus",
            "mean_between_consensus",
            "consensus_margin",
            "fold_selection_frequency",
            "pipeline_mean_absolute_shap",
            "pipeline_mean_signed_shap",
            "representation_gap_support_tier",
        ]
    ]
    .drop_duplicates("gene_id")
    .copy()
)

full_pool = (
    patient_gene.merge(
        core_gene_lookup,
        left_on="raw_gene_id",
        right_on="gene_id",
        how="inner",
        validate="many_to_one",
        suffixes=("_patient", "_core"),
    )
)

global_columns = [
    "raw_gene_id",
    "n_folds_selected_as_gap",
    "n_repeats_selected_as_gap",
    "fold_selection_frequency",
    "repeat_selection_frequency",
    "pipeline_mean_absolute_shap",
    "pipeline_mean_signed_shap",
    "fold_direction_consistency",
    "representation_gap_support_tier",
]

available_global_columns = [
    column
    for column in global_columns
    if column in global_gene.columns
]

global_gene_small = (
    global_gene[available_global_columns]
    .drop_duplicates("raw_gene_id")
    .copy()
)

full_pool = (
    full_pool.merge(
        global_gene_small,
        on="raw_gene_id",
        how="left",
        validate="many_to_one",
        suffixes=("", "_global"),
    )
)

# Resolve patient-level columns.
patient_abs_column = first_existing_column(
    full_pool,
    [
        "pipeline_mean_absolute_shap_patient",
        "pipeline_mean_absolute_shap_x",
        "pipeline_mean_absolute_shap",
        "conditional_mean_absolute_shap",
    ],
)

patient_signed_column = first_existing_column(
    full_pool,
    [
        "pipeline_mean_signed_shap_patient",
        "pipeline_mean_signed_shap_x",
        "pipeline_mean_signed_shap",
        "conditional_mean_signed_shap",
    ],
)

repeat_frequency_column = first_existing_column(
    full_pool,
    [
        "repeat_selection_frequency",
        "repeat_selection_frequency_patient",
    ],
)

sign_consistency_column = first_existing_column(
    full_pool,
    [
        "attribution_sign_consistency",
    ],
)

fold_frequency_column = first_existing_column(
    full_pool,
    [
        "fold_selection_frequency",
        "fold_selection_frequency_core",
        "fold_selection_frequency_global",
    ],
)

full_pool["patient_abs_attribution"] = pd.to_numeric(
    full_pool[patient_abs_column],
    errors="coerce",
)

full_pool["patient_signed_attribution"] = pd.to_numeric(
    full_pool[patient_signed_column],
    errors="coerce",
)

full_pool["repeat_frequency"] = pd.to_numeric(
    full_pool[repeat_frequency_column],
    errors="coerce",
)

full_pool["sign_consistency"] = pd.to_numeric(
    full_pool[sign_consistency_column],
    errors="coerce",
)

full_pool["fold_frequency"] = pd.to_numeric(
    full_pool[fold_frequency_column],
    errors="coerce",
)

full_pool["passes_minimum_support"] = (
    (
        full_pool["repeat_frequency"]
        >= MIN_REPEAT_SELECTION_FREQUENCY
    )
    &
    (
        full_pool["fold_frequency"]
        >= MIN_FOLD_SELECTION_FREQUENCY
    )
    &
    (
        full_pool["sign_consistency"]
        >= MIN_SIGN_CONSISTENCY
    )
)

full_pool_supported = (
    full_pool[
        full_pool["passes_minimum_support"]
    ]
    .copy()
)

print("\nFull stable-core candidate rows:")
print(full_pool.shape)

print("Rows passing support:")
print(full_pool_supported.shape)


# %% [CELL 7] ============================================================
# COHORT FREQUENCY AND SPECIFICITY
# ============================================================

n_patients_full = patient_states["patient_id"].nunique()

cohort_gene_frequency = (
    full_pool_supported
    .groupby(
        [
            "raw_gene_id",
            "harmonized_gene_id",
            "core_module_name",
        ],
        as_index=False,
    )
    .agg(
        n_patients_with_supported_attribution=(
            "patient_id",
            "nunique",
        ),
        n_patient_gene_rows=(
            "patient_id",
            "size",
        ),
        mean_patient_abs_attribution=(
            "patient_abs_attribution",
            "mean",
        ),
        median_patient_abs_attribution=(
            "patient_abs_attribution",
            "median",
        ),
        mean_repeat_frequency=(
            "repeat_frequency",
            "mean",
        ),
        mean_fold_frequency=(
            "fold_frequency",
            "mean",
        ),
        mean_sign_consistency=(
            "sign_consistency",
            "mean",
        ),
        mean_consensus_margin=(
            "consensus_margin",
            "mean",
        ),
    )
)

cohort_gene_frequency[
    "cohort_frequency_fraction"
] = (
    cohort_gene_frequency[
        "n_patients_with_supported_attribution"
    ]
    /
    n_patients_full
)

cohort_gene_frequency["idf_raw"] = (
    np.log(
        (
            1.0 + n_patients_full
        )
        /
        (
            1.0
            +
            cohort_gene_frequency[
                "n_patients_with_supported_attribution"
            ]
        )
    )
    + 1.0
)

cohort_gene_frequency[
    "idf_normalized"
] = minmax_scale(
    cohort_gene_frequency["idf_raw"]
)

cohort_gene_frequency[
    "cohort_target_class"
] = np.select(
    [
        (
            cohort_gene_frequency[
                "cohort_frequency_fraction"
            ]
            >= GENERIC_FREQUENCY_THRESHOLD
        ),
        (
            cohort_gene_frequency[
                "cohort_frequency_fraction"
            ]
            >= RECURRENT_FREQUENCY_THRESHOLD
        ),
    ],
    [
        "cohort_level_sentinel",
        "recurrent_gap_gene",
    ],
    default="patient_specific_candidate",
)

cohort_gene_frequency = (
    cohort_gene_frequency
    .sort_values(
        [
            "cohort_frequency_fraction",
            "mean_patient_abs_attribution",
        ],
        ascending=[
            False,
            False,
        ],
    )
    .reset_index(drop=True)
)

display(
    cohort_gene_frequency[
        [
            "raw_gene_id",
            "harmonized_gene_id",
            "core_module_name",
            "n_patients_with_supported_attribution",
            "cohort_frequency_fraction",
            "idf_normalized",
            "cohort_target_class",
        ]
    ].head(40)
)


# %% [CELL 8] ============================================================
# PATIENT-SPECIFIC FULL-POOL SCORING
# ============================================================

full_pool_scored = (
    full_pool_supported.merge(
        cohort_gene_frequency[
            [
                "raw_gene_id",
                "core_module_name",
                "cohort_frequency_fraction",
                "idf_raw",
                "idf_normalized",
                "cohort_target_class",
            ]
        ],
        on=[
            "raw_gene_id",
            "core_module_name",
        ],
        how="left",
        validate="many_to_one",
    )
)

full_pool_scored[
    "patient_abs_percentile"
] = rank_pct_within_group(
    full_pool_scored,
    "patient_abs_attribution",
)

full_pool_scored[
    "patient_repeat_support"
] = (
    0.50
    * full_pool_scored["repeat_frequency"]
    +
    0.25
    * full_pool_scored["sign_consistency"]
    +
    0.25
    * full_pool_scored["fold_frequency"]
)

full_pool_scored[
    "patient_abs_scaled"
] = (
    full_pool_scored
    .groupby("patient_id")[
        "patient_abs_attribution"
    ]
    .transform(
        lambda values:
            minmax_scale(values).to_numpy()
    )
)

full_pool_scored[
    "core_stability_scaled"
] = minmax_scale(
    full_pool_scored["consensus_margin"]
)

full_pool_scored[
    "full_pool_specificity_score"
] = (
    W_PATIENT_ATTRIBUTION
    * full_pool_scored[
        "patient_abs_scaled"
    ]
    +
    W_PATIENT_PERCENTILE
    * full_pool_scored[
        "patient_abs_percentile"
    ]
    +
    W_GLOBAL_STABILITY
    * full_pool_scored[
        "patient_repeat_support"
    ]
    +
    W_CORE_STABILITY
    * full_pool_scored[
        "core_stability_scaled"
    ]
    +
    W_SPECIFICITY
    * full_pool_scored[
        "idf_normalized"
    ]
)

full_pool_scored[
    "rank_within_patient"
] = (
    full_pool_scored
    .groupby("patient_id")[
        "full_pool_specificity_score"
    ]
    .rank(
        method="first",
        ascending=False,
    )
    .astype(int)
)

full_pool_scored[
    "rank_within_patient_and_class"
] = (
    full_pool_scored
    .groupby(
        [
            "patient_id",
            "cohort_target_class",
        ]
    )[
        "full_pool_specificity_score"
    ]
    .rank(
        method="first",
        ascending=False,
    )
    .astype(int)
)


# %% [CELL 9] ============================================================
# SELECT THREE-LAYER TARGET PANELS
# ============================================================

generic_targets = (
    full_pool_scored[
        (
            full_pool_scored[
                "cohort_target_class"
            ]
            ==
            "cohort_level_sentinel"
        )
        &
        (
            full_pool_scored[
                "rank_within_patient_and_class"
            ]
            <=
            TOP_GENERIC_PER_PATIENT
        )
    ]
    .copy()
)

recurrent_targets = (
    full_pool_scored[
        (
            full_pool_scored[
                "cohort_target_class"
            ]
            ==
            "recurrent_gap_gene"
        )
        &
        (
            full_pool_scored[
                "rank_within_patient_and_class"
            ]
            <=
            TOP_RECURRENT_PER_PATIENT
        )
    ]
    .copy()
)

patient_specific_targets = (
    full_pool_scored[
        (
            full_pool_scored[
                "cohort_target_class"
            ]
            ==
            "patient_specific_candidate"
        )
        &
        (
            full_pool_scored[
                "rank_within_patient_and_class"
            ]
            <=
            TOP_PATIENT_SPECIFIC_PER_PATIENT
        )
    ]
    .copy()
)

three_layer_targets = pd.concat(
    [
        generic_targets,
        recurrent_targets,
        patient_specific_targets,
    ],
    ignore_index=True,
)

three_layer_targets[
    "target_layer"
] = three_layer_targets[
    "cohort_target_class"
].map(
    {
        "cohort_level_sentinel":
            "Layer_A_cohort_representation_expansion",
        "recurrent_gap_gene":
            "Layer_B_recurrent_patient_relevant",
        "patient_specific_candidate":
            "Layer_C_patient_specific",
    }
)

three_layer_targets[
    "rank_within_patient_final"
] = (
    three_layer_targets
    .groupby("patient_id")[
        "full_pool_specificity_score"
    ]
    .rank(
        method="first",
        ascending=False,
    )
    .astype(int)
)

# Keep no more than the requested total.
three_layer_targets = (
    three_layer_targets[
        three_layer_targets[
            "rank_within_patient_final"
        ]
        <=
        TOP_TOTAL_PER_PATIENT
    ]
    .copy()
)

print("\nThree-layer target rows:")
print(three_layer_targets.shape)

print("\nLayer counts:")
display(
    three_layer_targets[
        "target_layer"
    ]
    .value_counts()
    .rename_axis("target_layer")
    .reset_index(name="n_rows")
)


# %% [CELL 10] ============================================================
# PATIENT-LEVEL FULL-POOL SUMMARY
# ============================================================

patient_target_summary = (
    three_layer_targets
    .groupby(
        "patient_id",
        as_index=False,
    )
    .agg(
        n_total_targets=(
            "raw_gene_id",
            "size",
        ),
        n_generic_targets=(
            "target_layer",
            lambda values:
                int(
                    np.sum(
                        values
                        ==
                        "Layer_A_cohort_representation_expansion"
                    )
                ),
        ),
        n_recurrent_targets=(
            "target_layer",
            lambda values:
                int(
                    np.sum(
                        values
                        ==
                        "Layer_B_recurrent_patient_relevant"
                    )
                ),
        ),
        n_patient_specific_targets=(
            "target_layer",
            lambda values:
                int(
                    np.sum(
                        values
                        ==
                        "Layer_C_patient_specific"
                    )
                ),
        ),
        mean_specificity_score=(
            "full_pool_specificity_score",
            "mean",
        ),
        maximum_specificity_score=(
            "full_pool_specificity_score",
            "max",
        ),
        mean_idf_normalized=(
            "idf_normalized",
            "mean",
        ),
        dominant_core=(
            "core_module_name",
            lambda values:
                values.value_counts().index[0],
        ),
    )
)

patient_target_summary[
    "patient_specific_fraction"
] = (
    patient_target_summary[
        "n_patient_specific_targets"
    ]
    /
    patient_target_summary[
        "n_total_targets"
    ].replace(
        0,
        np.nan,
    )
)

patient_target_summary[
    "generic_fraction"
] = (
    patient_target_summary[
        "n_generic_targets"
    ]
    /
    patient_target_summary[
        "n_total_targets"
    ].replace(
        0,
        np.nan,
    )
)

patient_target_summary[
    "full_pool_recommendation_specificity"
] = (
    0.50
    * patient_target_summary[
        "patient_specific_fraction"
    ]
    +
    0.30
    * patient_target_summary[
        "mean_idf_normalized"
    ]
    +
    0.20
    * (
        1.0
        -
        patient_target_summary[
            "generic_fraction"
        ]
    )
)

patient_target_summary[
    "full_pool_specificity_tier"
] = pd.cut(
    patient_target_summary[
        "full_pool_recommendation_specificity"
    ],
    bins=[
        -np.inf,
        0.35,
        0.60,
        np.inf,
    ],
    labels=[
        "low_specificity",
        "moderate_specificity",
        "higher_specificity",
    ],
)

patient_target_summary = (
    patient_states.merge(
        patient_target_summary,
        on="patient_id",
        how="left",
        validate="one_to_one",
    )
)

for column in [
    "n_total_targets",
    "n_generic_targets",
    "n_recurrent_targets",
    "n_patient_specific_targets",
]:
    patient_target_summary[column] = (
        patient_target_summary[column]
        .fillna(0)
        .astype(int)
    )


# %% [CELL 11] ============================================================
# TARGET-SET DIVERSITY AND JACCARD AUDIT
# ============================================================

target_sets = {
    patient_id:
        set(
            patient_df["raw_gene_id"]
        )
    for patient_id, patient_df
    in three_layer_targets.groupby("patient_id")
}

patient_ids_with_targets = sorted(target_sets)

pairwise_jaccard_records = []

for patient_1, patient_2 in combinations(
    patient_ids_with_targets,
    2,
):
    pairwise_jaccard_records.append({
        "patient_1":
            patient_1,
        "patient_2":
            patient_2,
        "jaccard":
            jaccard(
                target_sets[patient_1],
                target_sets[patient_2],
            ),
    })

pairwise_jaccard = pd.DataFrame(
    pairwise_jaccard_records
)

observed_target_diversity = {
    "mean_pairwise_jaccard":
        float(
            pairwise_jaccard["jaccard"].mean()
        ),
    "median_pairwise_jaccard":
        float(
            pairwise_jaccard["jaccard"].median()
        ),
    "mean_pairwise_dissimilarity":
        float(
            1.0
            -
            pairwise_jaccard["jaccard"].mean()
        ),
}


# %% [CELL 12] ============================================================
# WRITE FULL-POOL PATIENT REPORTS
# ============================================================

def format_target_label(row):
    raw_gene = str(row["raw_gene_id"])
    harmonized_gene = str(row["harmonized_gene_id"])

    if (
        harmonized_gene
        and
        harmonized_gene.lower()
        not in {"nan", "none", "null", ""}
        and
        harmonized_gene != raw_gene
    ):
        gene_label = f"{raw_gene}->{harmonized_gene}"
    else:
        gene_label = raw_gene

    return (
        f"{gene_label}"
        f"[{row['core_module_name']};"
        f"{row['target_layer']};"
        f"score={row['full_pool_specificity_score']:.4f}]"
    )


three_layer_targets[
    "target_label"
] = (
    three_layer_targets.apply(
        format_target_label,
        axis=1,
    )
)

patient_target_text_records = []

for patient_id, patient_df in (
    three_layer_targets
    .sort_values(
        [
            "patient_id",
            "rank_within_patient_final",
        ]
    )
    .groupby("patient_id")
):
    layer_text = {}

    for layer_name, layer_df in (
        patient_df.groupby("target_layer")
    ):
        layer_text[layer_name] = "; ".join(
            layer_df[
                "target_label"
            ].tolist()
        )

    patient_target_text_records.append({
        "patient_id":
            patient_id,
        "cohort_level_sentinel_targets":
            layer_text.get(
                "Layer_A_cohort_representation_expansion",
                "",
            ),
        "recurrent_patient_relevant_targets":
            layer_text.get(
                "Layer_B_recurrent_patient_relevant",
                "",
            ),
        "patient_specific_targets":
            layer_text.get(
                "Layer_C_patient_specific",
                "",
            ),
    })

patient_target_text = pd.DataFrame(
    patient_target_text_records
)

final_full_pool_reports = (
    patient_target_summary.merge(
        patient_target_text,
        on="patient_id",
        how="left",
        validate="one_to_one",
    )
)

for _, row in final_full_pool_reports.iterrows():
    patient_id = str(row["patient_id"])

    report_lines = [
        "AIDO-BBA FULL-POOL MEASUREMENT TARGET RECONSTRUCTION",
        "=" * 72,
        "",
        f"Patient: {patient_id}",
        f"Clinical group: {row.get('true_group', 'NA')}",
        (
            "Clinical-molecular rank state: "
            f"{row.get('clinical_molecular_rank_state', 'NA')}"
        ),
        (
            "Integrated BBA state: "
            f"{row.get('integrated_bba_state', 'NA')}"
        ),
        "",
        "LAYER A — COHORT-LEVEL REPRESENTATION EXPANSION",
        "-" * 72,
        str(
            row.get(
                "cohort_level_sentinel_targets",
                "",
            )
        )
        or "None selected.",
        "",
        "LAYER B — RECURRENT PATIENT-RELEVANT GAPS",
        "-" * 72,
        str(
            row.get(
                "recurrent_patient_relevant_targets",
                "",
            )
        )
        or "None selected.",
        "",
        "LAYER C — PATIENT-SPECIFIC CANDIDATE GAPS",
        "-" * 72,
        str(
            row.get(
                "patient_specific_targets",
                "",
            )
        )
        or "None selected.",
        "",
        "SPECIFICITY SUMMARY",
        "-" * 72,
        (
            "Patient-specific fraction: "
            f"{row.get('patient_specific_fraction', np.nan):.4f}"
        ),
        (
            "Generic fraction: "
            f"{row.get('generic_fraction', np.nan):.4f}"
        ),
        (
            "Full-pool recommendation specificity: "
            f"{row.get('full_pool_recommendation_specificity', np.nan):.4f}"
        ),
        (
            "Specificity tier: "
            f"{row.get('full_pool_specificity_tier', 'NA')}"
        ),
        "",
        "INTERPRETATION BOUNDARY",
        "-" * 72,
        (
            "These are computational measurement priorities. "
            "They are not validated biomarkers, clinical tests, "
            "diagnoses, prognostic conclusions, or treatment "
            "recommendations."
        ),
    ]

    report_path = (
        FULL_POOL_DIR
        / "reports"
        / f"{patient_id}_full_pool_target_report.txt"
    )

    report_path.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )


# %% [CELL 13] ============================================================
# SAVE PART A OUTPUTS
# ============================================================

cohort_gene_frequency.to_csv(
    FULL_POOL_DIR
    / "summaries"
    / "cohort_gap_gene_frequency.tsv",
    sep="\t",
    index=False,
)

full_pool_scored.to_csv(
    FULL_POOL_DIR
    / "summaries"
    / "all_supported_patient_gene_candidates_scored.tsv",
    sep="\t",
    index=False,
)

three_layer_targets.to_csv(
    FULL_POOL_DIR
    / "summaries"
    / "three_layer_patient_measurement_targets.tsv",
    sep="\t",
    index=False,
)

patient_target_summary.to_csv(
    FULL_POOL_DIR
    / "summaries"
    / "patient_full_pool_target_summary.tsv",
    sep="\t",
    index=False,
)

pairwise_jaccard.to_csv(
    FULL_POOL_DIR
    / "summaries"
    / "patient_target_set_pairwise_jaccard.tsv",
    sep="\t",
    index=False,
)

final_full_pool_reports.to_csv(
    FULL_POOL_DIR
    / "summaries"
    / "all_patient_full_pool_target_reports.tsv",
    sep="\t",
    index=False,
)

part_a_summary = pd.DataFrame([
    {
        "metric":
            "n_patients",
        "value":
            n_patients_full,
    },
    {
        "metric":
            "n_supported_candidate_rows",
        "value":
            len(full_pool_scored),
    },
    {
        "metric":
            "n_unique_candidate_genes",
        "value":
            full_pool_scored[
                "raw_gene_id"
            ].nunique(),
    },
    {
        "metric":
            "n_three_layer_target_rows",
        "value":
            len(three_layer_targets),
    },
    {
        "metric":
            "n_cohort_level_sentinel_rows",
        "value":
            int(
                np.sum(
                    three_layer_targets[
                        "target_layer"
                    ]
                    ==
                    "Layer_A_cohort_representation_expansion"
                )
            ),
    },
    {
        "metric":
            "n_recurrent_target_rows",
        "value":
            int(
                np.sum(
                    three_layer_targets[
                        "target_layer"
                    ]
                    ==
                    "Layer_B_recurrent_patient_relevant"
                )
            ),
    },
    {
        "metric":
            "n_patient_specific_target_rows",
        "value":
            int(
                np.sum(
                    three_layer_targets[
                        "target_layer"
                    ]
                    ==
                    "Layer_C_patient_specific"
                )
            ),
    },
    {
        "metric":
            "mean_pairwise_target_jaccard",
        "value":
            observed_target_diversity[
                "mean_pairwise_jaccard"
            ],
    },
    {
        "metric":
            "mean_pairwise_target_dissimilarity",
        "value":
            observed_target_diversity[
                "mean_pairwise_dissimilarity"
            ],
    },
])

part_a_summary.to_csv(
    FULL_POOL_DIR
    / "summaries"
    / "full_pool_target_reconstruction_summary.tsv",
    sep="\t",
    index=False,
)

display(part_a_summary)


# %% [CELL 14] ============================================================
# PART B — BUILD DENSE PATIENT × GENE MATRICES FOR NULL AUDIT
# ============================================================

null_rng = np.random.default_rng(
    NULL_RANDOM_SEED
)

eligible_genes = sorted(
    full_pool_scored[
        "raw_gene_id"
    ].unique()
)

null_patient_ids = sorted(
    patient_states[
        "patient_id"
    ].astype(str).unique()
)

patient_to_index = {
    patient_id: index
    for index, patient_id
    in enumerate(null_patient_ids)
}

gene_to_index = {
    gene: index
    for index, gene
    in enumerate(eligible_genes)
}

n_null_patients = len(null_patient_ids)
n_null_genes = len(eligible_genes)

abs_matrix = np.zeros(
    (
        n_null_patients,
        n_null_genes,
    ),
    dtype=np.float32,
)

repeat_matrix = np.zeros_like(
    abs_matrix
)

sign_consistency_matrix = np.zeros_like(
    abs_matrix
)

present_matrix = np.zeros_like(
    abs_matrix,
    dtype=bool,
)

for row in full_pool_scored.itertuples(
    index=False,
):
    patient_index = patient_to_index[
        str(row.patient_id)
    ]
    gene_index = gene_to_index[
        str(row.raw_gene_id)
    ]

    abs_matrix[
        patient_index,
        gene_index,
    ] = float(
        row.patient_abs_attribution
    )

    repeat_matrix[
        patient_index,
        gene_index,
    ] = float(
        row.repeat_frequency
    )

    sign_consistency_matrix[
        patient_index,
        gene_index,
    ] = float(
        row.sign_consistency
    )

    present_matrix[
        patient_index,
        gene_index,
    ] = True

gene_idf = (
    cohort_gene_frequency
    .set_index("raw_gene_id")
    .reindex(eligible_genes)[
        "idf_normalized"
    ]
    .fillna(0)
    .to_numpy(dtype=np.float32)
)

gene_fold_frequency = (
    full_pool_scored
    .groupby("raw_gene_id")[
        "fold_frequency"
    ]
    .mean()
    .reindex(eligible_genes)
    .fillna(0)
    .to_numpy(dtype=np.float32)
)

gene_consensus_margin = (
    full_pool_scored
    .groupby("raw_gene_id")[
        "consensus_margin"
    ]
    .mean()
    .reindex(eligible_genes)
    .fillna(0)
    .to_numpy(dtype=np.float32)
)

gene_core = (
    full_pool_scored
    .groupby("raw_gene_id")[
        "core_module_name"
    ]
    .first()
    .reindex(eligible_genes)
    .astype(str)
    .to_numpy()
)


# %% [CELL 15] ============================================================
# NULL AUDIT SCORING FUNCTIONS
# ============================================================

def dense_score_matrix(
    absolute_matrix,
    repeat_support_matrix,
    sign_support_matrix,
    idf_vector,
    fold_frequency_vector,
    consensus_margin_vector,
):
    matrix = np.asarray(
        absolute_matrix,
        dtype=float,
    )

    # Patient-wise min-max.
    row_min = np.min(
        matrix,
        axis=1,
        keepdims=True,
    )
    row_max = np.max(
        matrix,
        axis=1,
        keepdims=True,
    )

    patient_abs_scaled = np.divide(
        matrix - row_min,
        row_max - row_min,
        out=np.zeros_like(matrix),
        where=(
            row_max - row_min
            >
            0
        ),
    )

    # Patient-wise percentile ranks.
    order = np.argsort(
        matrix,
        axis=1,
    )
    ranks = np.empty_like(
        order,
        dtype=float,
    )

    base_ranks = (
        np.arange(
            matrix.shape[1],
            dtype=float,
        )
        + 1.0
    ) / matrix.shape[1]

    for patient_index in range(
        matrix.shape[0]
    ):
        ranks[
            patient_index,
            order[
                patient_index
            ],
        ] = base_ranks

    repeat_support = (
        0.50
        * repeat_support_matrix
        +
        0.25
        * sign_support_matrix
        +
        0.25
        * fold_frequency_vector[
            None,
            :
        ]
    )

    consensus_scaled = (
        minmax_scale(
            consensus_margin_vector
        )
        .fillna(0)
        .to_numpy(dtype=float)
    )

    scores = (
        W_PATIENT_ATTRIBUTION
        * patient_abs_scaled
        +
        W_PATIENT_PERCENTILE
        * ranks
        +
        W_GLOBAL_STABILITY
        * repeat_support
        +
        W_CORE_STABILITY
        * consensus_scaled[
            None,
            :
        ]
        +
        W_SPECIFICITY
        * idf_vector[
            None,
            :
        ]
    )

    return scores


def top_k_gene_sets_from_scores(
    scores,
    present_mask,
    top_k,
):
    result = []

    for patient_index in range(
        scores.shape[0]
    ):
        valid_indices = np.where(
            present_mask[
                patient_index
            ]
        )[0]

        if len(valid_indices) == 0:
            result.append(set())
            continue

        patient_scores = scores[
            patient_index,
            valid_indices
        ]

        n_select = min(
            top_k,
            len(valid_indices),
        )

        selected_local = np.argpartition(
            patient_scores,
            -n_select,
        )[
            -n_select:
        ]

        selected_indices = valid_indices[
            selected_local
        ]

        result.append(
            set(
                eligible_genes[index]
                for index
                in selected_indices
            )
        )

    return result


def target_set_metrics(
    target_sets,
):
    recurrence_counter = Counter()

    for target_set in target_sets:
        recurrence_counter.update(
            target_set
        )

    pairwise_jaccards = []

    for index_1, index_2 in combinations(
        range(
            len(target_sets)
        ),
        2,
    ):
        pairwise_jaccards.append(
            jaccard(
                target_sets[index_1],
                target_sets[index_2],
            )
        )

    pairwise_jaccards = np.asarray(
        pairwise_jaccards,
        dtype=float,
    )

    recurrence_values = np.asarray(
        list(
            recurrence_counter.values()
        ),
        dtype=float,
    )

    if len(recurrence_values) == 0:
        max_recurrence_fraction = np.nan
    else:
        max_recurrence_fraction = (
            recurrence_values.max()
            /
            len(target_sets)
        )

    unique_targets_per_patient = [
        len(target_set)
        for target_set
        in target_sets
    ]

    return {
        "mean_pairwise_jaccard":
            float(
                np.nanmean(
                    pairwise_jaccards
                )
            ),
        "mean_pairwise_dissimilarity":
            float(
                1.0
                -
                np.nanmean(
                    pairwise_jaccards
                )
            ),
        "max_target_recurrence_fraction":
            float(
                max_recurrence_fraction
            ),
        "mean_targets_per_patient":
            float(
                np.mean(
                    unique_targets_per_patient
                )
            ),
        "n_unique_targets":
            int(
                len(
                    recurrence_counter
                )
            ),
    }


observed_dense_scores = dense_score_matrix(
    abs_matrix,
    repeat_matrix,
    sign_consistency_matrix,
    gene_idf,
    gene_fold_frequency,
    gene_consensus_margin,
)

observed_top_sets = top_k_gene_sets_from_scores(
    observed_dense_scores,
    present_matrix,
    NULL_TOP_K,
)

observed_null_metrics = target_set_metrics(
    observed_top_sets
)

print("\nObserved target-set metrics:")
display(
    pd.DataFrame([
        observed_null_metrics
    ])
)


# %% [CELL 16] ============================================================
# NULL A — WITHIN-GENE PATIENT ATTRIBUTION PERMUTATION
# ============================================================

null_a_records = []

start_time = time.time()

for permutation_id in range(
    1,
    N_NULL_PERMUTATIONS + 1,
):
    permuted_abs = np.zeros_like(
        abs_matrix
    )
    permuted_repeat = np.zeros_like(
        repeat_matrix
    )
    permuted_sign = np.zeros_like(
        sign_consistency_matrix
    )
    permuted_present = np.zeros_like(
        present_matrix
    )

    for gene_index in range(
        n_null_genes
    ):
        patient_indices = np.where(
            present_matrix[
                :,
                gene_index
            ]
        )[0]

        if len(patient_indices) == 0:
            continue

        shuffled_destinations = (
            null_rng.permutation(
                patient_indices
            )
        )

        permuted_abs[
            shuffled_destinations,
            gene_index,
        ] = abs_matrix[
            patient_indices,
            gene_index,
        ]

        permuted_repeat[
            shuffled_destinations,
            gene_index,
        ] = repeat_matrix[
            patient_indices,
            gene_index,
        ]

        permuted_sign[
            shuffled_destinations,
            gene_index,
        ] = sign_consistency_matrix[
            patient_indices,
            gene_index,
        ]

        permuted_present[
            shuffled_destinations,
            gene_index,
        ] = True

    permuted_scores = dense_score_matrix(
        permuted_abs,
        permuted_repeat,
        permuted_sign,
        gene_idf,
        gene_fold_frequency,
        gene_consensus_margin,
    )

    permuted_sets = (
        top_k_gene_sets_from_scores(
            permuted_scores,
            permuted_present,
            NULL_TOP_K,
        )
    )

    metrics = target_set_metrics(
        permuted_sets
    )

    metrics.update({
        "null_type":
            "within_gene_patient_permutation",
        "permutation_id":
            permutation_id,
    })

    null_a_records.append(
        metrics
    )

    if (
        permutation_id == 1
        or permutation_id % 20 == 0
        or permutation_id
        ==
        N_NULL_PERMUTATIONS
    ):
        elapsed = (
            time.time()
            -
            start_time
        ) / 60

        print(
            f"Null A {permutation_id:>3}/"
            f"{N_NULL_PERMUTATIONS} | "
            f"{elapsed:.2f} min"
        )

null_a = pd.DataFrame(
    null_a_records
)


# %% [CELL 17] ============================================================
# NULL B — WITHIN-PATIENT, WITHIN-CORE GENE-IDENTITY PERMUTATION
# ============================================================

core_to_gene_indices = defaultdict(list)

for gene_index, core_name in enumerate(
    gene_core
):
    core_to_gene_indices[
        str(core_name)
    ].append(
        gene_index
    )

null_b_records = []

for permutation_id in range(
    1,
    N_NULL_PERMUTATIONS + 1,
):
    permuted_abs = abs_matrix.copy()
    permuted_repeat = repeat_matrix.copy()
    permuted_sign = sign_consistency_matrix.copy()
    permuted_present = present_matrix.copy()

    for patient_index in range(
        n_null_patients
    ):
        for core_name, core_indices in (
            core_to_gene_indices.items()
        ):
            core_indices = np.asarray(
                core_indices,
                dtype=int,
            )

            source_indices = core_indices[
                present_matrix[
                    patient_index,
                    core_indices
                ]
            ]

            if len(source_indices) <= 1:
                continue

            destination_indices = (
                null_rng.permutation(
                    source_indices
                )
            )

            original_abs = abs_matrix[
                patient_index,
                source_indices
            ].copy()

            original_repeat = repeat_matrix[
                patient_index,
                source_indices
            ].copy()

            original_sign = sign_consistency_matrix[
                patient_index,
                source_indices
            ].copy()

            permuted_abs[
                patient_index,
                source_indices
            ] = 0

            permuted_repeat[
                patient_index,
                source_indices
            ] = 0

            permuted_sign[
                patient_index,
                source_indices
            ] = 0

            permuted_present[
                patient_index,
                source_indices
            ] = False

            permuted_abs[
                patient_index,
                destination_indices
            ] = original_abs

            permuted_repeat[
                patient_index,
                destination_indices
            ] = original_repeat

            permuted_sign[
                patient_index,
                destination_indices
            ] = original_sign

            permuted_present[
                patient_index,
                destination_indices
            ] = True

    permuted_scores = dense_score_matrix(
        permuted_abs,
        permuted_repeat,
        permuted_sign,
        gene_idf,
        gene_fold_frequency,
        gene_consensus_margin,
    )

    permuted_sets = (
        top_k_gene_sets_from_scores(
            permuted_scores,
            permuted_present,
            NULL_TOP_K,
        )
    )

    metrics = target_set_metrics(
        permuted_sets
    )

    metrics.update({
        "null_type":
            "within_patient_within_core_gene_identity_permutation",
        "permutation_id":
            permutation_id,
    })

    null_b_records.append(
        metrics
    )

    if (
        permutation_id == 1
        or permutation_id % 20 == 0
        or permutation_id
        ==
        N_NULL_PERMUTATIONS
    ):
        elapsed = (
            time.time()
            -
            start_time
        ) / 60

        print(
            f"Null B {permutation_id:>3}/"
            f"{N_NULL_PERMUTATIONS} | "
            f"{elapsed:.2f} min"
        )

null_b = pd.DataFrame(
    null_b_records
)


# %% [CELL 18] ============================================================
# NULL COMPARISON SUMMARY
# ============================================================

null_results = pd.concat(
    [
        null_a,
        null_b,
    ],
    ignore_index=True,
)

null_comparison_records = []

for null_type, null_df in (
    null_results.groupby("null_type")
):
    for metric_name, observed_value in (
        observed_null_metrics.items()
    ):
        null_values = (
            null_df[
                metric_name
            ]
            .to_numpy(dtype=float)
        )

        null_mean = float(
            np.mean(
                null_values
            )
        )

        null_sd = float(
            np.std(
                null_values,
                ddof=1,
            )
        )

        null_comparison_records.append({
            "null_type":
                null_type,
            "metric":
                metric_name,
            "observed_value":
                observed_value,
            "null_mean":
                null_mean,
            "null_sd":
                null_sd,
            "observed_minus_null":
                observed_value
                -
                null_mean,
            "z_score":
                (
                    (
                        observed_value
                        -
                        null_mean
                    )
                    /
                    null_sd
                    if null_sd > 0
                    else np.nan
                ),
            "empirical_upper_p":
                empirical_p_greater_equal(
                    observed_value,
                    null_values,
                ),
            "empirical_lower_p":
                empirical_p_less_equal(
                    observed_value,
                    null_values,
                ),
        })

null_comparison = pd.DataFrame(
    null_comparison_records
)

display(
    null_comparison
    .sort_values(
        [
            "null_type",
            "metric",
        ]
    )
)


# %% [CELL 19] ============================================================
# SAVE PART B OUTPUTS
# ============================================================

null_results.to_csv(
    NULL_DIR
    / "summaries"
    / "recommendation_null_permutation_results.tsv",
    sep="\t",
    index=False,
)

null_comparison.to_csv(
    NULL_DIR
    / "summaries"
    / "recommendation_null_comparison_summary.tsv",
    sep="\t",
    index=False,
)

pd.DataFrame([
    observed_null_metrics
]).to_csv(
    NULL_DIR
    / "summaries"
    / "observed_recommendation_set_metrics.tsv",
    sep="\t",
    index=False,
)

null_manifest = {
    "analysis":
        "AIDO-BBA recommendation null audit",
    "run_directory":
        str(RUN_DIR),
    "n_permutations_per_null":
        N_NULL_PERMUTATIONS,
    "random_seed":
        NULL_RANDOM_SEED,
    "top_k_targets":
        NULL_TOP_K,
    "null_A":
        (
            "Within-gene patient-attribution permutation "
            "preserving each gene's observed attribution "
            "distribution and recurrence."
        ),
    "null_B":
        (
            "Within-patient, within-core gene-identity "
            "permutation preserving each patient's core-level "
            "attribution burden."
        ),
}

with open(
    NULL_DIR
    / "recommendation_null_audit_manifest.json",
    "w",
    encoding="utf-8",
) as handle:
    json.dump(
        null_manifest,
        handle,
        indent=2,
    )


# %% [CELL 20] ============================================================
# PART C — AUTO-DETECT METABRIC FILES
# ============================================================

def resolve_optional_path(
    explicit_path,
    candidates,
    search_root=Path(r"D:\AIDO-Data"),
    search_terms=None,
):
    if explicit_path is not None:
        explicit_path = Path(
            explicit_path
        )

        if explicit_path.exists():
            return explicit_path

    for candidate in candidates:
        if candidate.exists():
            return candidate

    if search_terms:
        matches = []

        for path in search_root.rglob("*"):
            if not path.is_file():
                continue

            path_text = str(path).lower()

            if all(
                term.lower()
                in path_text
                for term in search_terms
            ):
                matches.append(path)

        if matches:
            return sorted(
                matches,
                key=lambda path:
                    path.stat().st_mtime,
                reverse=True,
            )[0]

    return None


resolved_metabric_ge = resolve_optional_path(
    METABRIC_GE_PATH,
    METABRIC_GE_CANDIDATES,
    search_terms=[
        "metabric",
        "expression",
    ],
)

resolved_metabric_clinical = resolve_optional_path(
    METABRIC_CLINICAL_PATH,
    METABRIC_CLINICAL_CANDIDATES,
    search_terms=[
        "metabric",
        "clinical",
    ],
)

print("\nMETABRIC GE:")
print(resolved_metabric_ge)

print("\nMETABRIC clinical:")
print(resolved_metabric_clinical)

METABRIC_AVAILABLE = (
    RUN_METABRIC_STRESS_IF_AVAILABLE
    and
    resolved_metabric_ge is not None
    and
    resolved_metabric_clinical is not None
)


# %% [CELL 21] ============================================================
# GENERIC METABRIC LOADERS
# ============================================================

def read_table_flexible(path):
    path = Path(path)

    # Try tab first.
    try:
        dataframe = pd.read_csv(
            path,
            sep="\t",
            comment="#",
            low_memory=False,
        )

        if dataframe.shape[1] > 1:
            return dataframe
    except Exception:
        pass

    # Try comma.
    dataframe = pd.read_csv(
        path,
        sep=",",
        comment="#",
        low_memory=False,
    )

    return dataframe


def load_expression_matrix(path):
    dataframe = read_table_flexible(
        path
    )

    # Remove obvious description columns.
    candidate_gene_columns = [
        column
        for column in dataframe.columns
        if str(column).strip().lower()
        in {
            "gene",
            "gene_symbol",
            "hugo_symbol",
            "symbol",
            "id",
            "entity",
        }
    ]

    if candidate_gene_columns:
        gene_column = candidate_gene_columns[0]
    else:
        gene_column = dataframe.columns[0]

    gene_ids = dataframe[
        gene_column
    ].astype(str)

    numeric_part = dataframe.drop(
        columns=[
            gene_column
        ]
    )

    # Remove nonnumeric metadata columns.
    numeric_part = numeric_part.apply(
        pd.to_numeric,
        errors="coerce",
    )

    numeric_part = numeric_part.dropna(
        axis=1,
        how="all",
    )

    numeric_part.index = gene_ids

    # Genes × samples expected.
    expression = numeric_part

    # If rows look like samples and columns look like genes,
    # transpose.
    if expression.shape[0] < expression.shape[1]:
        # This heuristic is not always sufficient. Retain the
        # orientation when the first dimension still resembles
        # a gene count.
        if expression.shape[0] < 5000:
            expression = expression.T

    expression.index = (
        expression.index.astype(str)
    )
    expression.columns = (
        expression.columns.astype(str)
    )

    # Aggregate duplicate genes.
    expression = (
        expression
        .groupby(
            expression.index
        )
        .mean()
    )

    return expression


def load_stage_labels(path):
    dataframe = read_table_flexible(
        path
    )

    normalized_columns = {
        str(column).strip().lower():
            column
        for column in dataframe.columns
    }

    sample_candidates = [
        "sample_id",
        "patient_id",
        "sample",
        "patient",
        "sample identifier",
        "patient identifier",
        "patient_id",
    ]

    stage_candidates = [
        "stage",
        "tumor_stage",
        "pathologic_stage",
        "clinical_stage",
        "stage at diagnosis",
        "neoplasm disease stage american joint committee on cancer code",
    ]

    sample_column = None
    stage_column = None

    for candidate in sample_candidates:
        if candidate in normalized_columns:
            sample_column = normalized_columns[
                candidate
            ]
            break

    for candidate in stage_candidates:
        if candidate in normalized_columns:
            stage_column = normalized_columns[
                candidate
            ]
            break

    if sample_column is None:
        sample_column = dataframe.columns[0]

    if stage_column is None:
        stage_like = [
            column
            for column in dataframe.columns
            if "stage"
            in str(column).lower()
        ]

        if stage_like:
            stage_column = stage_like[0]

    if stage_column is None:
        raise KeyError(
            "No stage-like clinical column was found."
        )

    labels = dataframe[
        [
            sample_column,
            stage_column,
        ]
    ].copy()

    labels.columns = [
        "sample_id",
        "stage_raw",
    ]

    labels[
        "stage_binary"
    ] = labels[
        "stage_raw"
    ].map(
        stage_to_binary
    )

    labels = (
        labels.dropna(
            subset=[
                "stage_binary"
            ]
        )
        .drop_duplicates(
            "sample_id"
        )
    )

    labels[
        "sample_id"
    ] = labels[
        "sample_id"
    ].astype(str)

    labels[
        "stage_binary"
    ] = labels[
        "stage_binary"
    ].astype(int)

    return labels


# %% [CELL 22] ============================================================
# RUN OPTIONAL METABRIC DATASET-REPLACEMENT STRESS TEST
# ============================================================

metabric_summary = None
metabric_fold_results = None
metabric_core_projection = None
metabric_gene_overlap = None

if not METABRIC_AVAILABLE:
    print(
        "\nMETABRIC stress test skipped because compatible "
        "GE and clinical files were not found."
    )

else:
    print(
        "\nRunning METABRIC dataset-replacement stress test..."
    )

    metabric_expression = load_expression_matrix(
        resolved_metabric_ge
    )

    metabric_labels = load_stage_labels(
        resolved_metabric_clinical
    )

    # Determine orientation by sample overlap.
    overlap_columns = set(
        metabric_expression.columns
    ) & set(
        metabric_labels["sample_id"]
    )

    overlap_index = set(
        metabric_expression.index
    ) & set(
        metabric_labels["sample_id"]
    )

    if len(overlap_index) > len(overlap_columns):
        metabric_expression = (
            metabric_expression.T
        )

    matched_samples = sorted(
        set(
            metabric_expression.columns
        )
        &
        set(
            metabric_labels["sample_id"]
        )
    )

    if len(matched_samples) < 100:
        raise ValueError(
            "Fewer than 100 METABRIC samples matched "
            "between expression and clinical labels."
        )

    metabric_y = (
        metabric_labels
        .set_index("sample_id")
        .loc[
            matched_samples,
            "stage_binary",
        ]
        .to_numpy(dtype=int)
    )

    metabric_X_gene_by_sample = (
        metabric_expression.loc[
            :,
            matched_samples,
        ]
    )

    # Remove zero-variance genes.
    gene_variance = (
        metabric_X_gene_by_sample.var(
            axis=1
        )
    )

    metabric_X_gene_by_sample = (
        metabric_X_gene_by_sample.loc[
            gene_variance > 0,
            :
        ]
    )

    metabric_X = (
        metabric_X_gene_by_sample.T
        .to_numpy(dtype=float)
    )

    metabric_gene_ids = (
        metabric_X_gene_by_sample.index
        .astype(str)
        .to_numpy()
    )

    cv = RepeatedStratifiedKFold(
        n_splits=METABRIC_N_SPLITS,
        n_repeats=METABRIC_N_REPEATS,
        random_state=METABRIC_RANDOM_SEED,
    )

    fold_records = []
    patient_probability_sum = np.zeros(
        len(matched_samples),
        dtype=float,
    )
    patient_probability_count = np.zeros(
        len(matched_samples),
        dtype=int,
    )

    selected_gene_counter = Counter()

    for fold_number, (
        train_index,
        test_index,
    ) in enumerate(
        cv.split(
            metabric_X,
            metabric_y,
        ),
        start=1,
    ):
        selector = SelectKBest(
            score_func=f_classif,
            k=min(
                METABRIC_TOP_K_GENES,
                metabric_X.shape[1],
            ),
        )

        X_train_selected = selector.fit_transform(
            metabric_X[
                train_index
            ],
            metabric_y[
                train_index
            ],
        )

        X_test_selected = selector.transform(
            metabric_X[
                test_index
            ]
        )

        selected_mask = selector.get_support()
        selected_genes = metabric_gene_ids[
            selected_mask
        ]

        selected_gene_counter.update(
            selected_genes.tolist()
        )

        model = ExtraTreesClassifier(
            n_estimators=METABRIC_N_TREES,
            random_state=(
                METABRIC_RANDOM_SEED
                +
                fold_number
            ),
            class_weight="balanced",
            n_jobs=-1,
        )

        model.fit(
            X_train_selected,
            metabric_y[
                train_index
            ],
        )

        probability = model.predict_proba(
            X_test_selected
        )[:, 1]

        fold_auc = roc_auc_score(
            metabric_y[
                test_index
            ],
            probability,
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
            "auc":
                fold_auc,
        })

        print(
            f"METABRIC fold "
            f"{fold_number:>2}/"
            f"{METABRIC_N_SPLITS * METABRIC_N_REPEATS} "
            f"AUC={fold_auc:.4f}"
        )

    metabric_fold_results = pd.DataFrame(
        fold_records
    )

    metabric_mean_probability = np.divide(
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

    metabric_oof_auc = roc_auc_score(
        metabric_y,
        metabric_mean_probability,
    )

    # Gene recurrence overlap with BRCA gap genes and stable cores.
    metabric_selected_gene_frequency = pd.DataFrame([
        {
            "gene_id":
                gene,
            "n_folds_selected":
                count,
            "selection_frequency":
                count
                /
                (
                    METABRIC_N_SPLITS
                    *
                    METABRIC_N_REPEATS
                ),
        }
        for gene, count
        in selected_gene_counter.items()
    ])

    brca_gap_genes = set(
        cohort_gene_frequency[
            "raw_gene_id"
        ].astype(str)
    )

    stable_core_genes = set(
        core_manifest[
            "gene_id"
        ].astype(str)
    )

    metabric_selected_genes = set(
        metabric_selected_gene_frequency[
            "gene_id"
        ].astype(str)
    )

    gap_overlap = (
        metabric_selected_genes
        &
        brca_gap_genes
    )

    core_overlap = (
        metabric_selected_genes
        &
        stable_core_genes
    )

    metabric_gene_overlap = pd.DataFrame([
        {
            "metric":
                "n_metabric_selected_genes",
            "value":
                len(
                    metabric_selected_genes
                ),
        },
        {
            "metric":
                "n_brca_gap_gene_overlap",
            "value":
                len(
                    gap_overlap
                ),
        },
        {
            "metric":
                "n_stable_core_gene_overlap",
            "value":
                len(
                    core_overlap
                ),
        },
        {
            "metric":
                "brca_gap_overlap_fraction",
            "value":
                (
                    len(gap_overlap)
                    /
                    max(
                        1,
                        len(
                            brca_gap_genes
                        )
                    )
                ),
        },
        {
            "metric":
                "stable_core_overlap_fraction",
            "value":
                (
                    len(core_overlap)
                    /
                    max(
                        1,
                        len(
                            stable_core_genes
                        )
                    )
                ),
        },
    ])

    # Simple stable-core expression projection.
    core_projection_records = []

    metabric_gene_set = set(
        metabric_X_gene_by_sample.index
        .astype(str)
    )

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

        if len(matched_core_genes) == 0:
            continue

        core_expression = (
            metabric_X_gene_by_sample.loc[
                matched_core_genes,
                matched_samples,
            ]
        )

        # Gene-wise z-score, then patient mean.
        gene_mean = core_expression.mean(
            axis=1
        )
        gene_sd = core_expression.std(
            axis=1
        ).replace(
            0,
            np.nan,
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

        core_score = core_z.mean(
            axis=0
        )

        early_values = core_score[
            np.asarray(
                metabric_y
            )
            ==
            0
        ]

        advanced_values = core_score[
            np.asarray(
                metabric_y
            )
            ==
            1
        ]

        u_statistic, p_value = mannwhitneyu(
            early_values,
            advanced_values,
            alternative="two-sided",
        )

        delta = (
            2.0
            *
            u_statistic
            /
            (
                len(
                    early_values
                )
                *
                len(
                    advanced_values
                )
            )
            -
            1.0
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
                    early_values.mean()
                ),
            "mean_advanced":
                float(
                    advanced_values.mean()
                ),
            "mean_difference_advanced_minus_early":
                float(
                    advanced_values.mean()
                    -
                    early_values.mean()
                ),
            "cliffs_delta_early_minus_advanced":
                float(
                    delta
                ),
            "p_value":
                float(
                    p_value
                ),
        })

    metabric_core_projection = pd.DataFrame(
        core_projection_records
    )

    metabric_summary = pd.DataFrame([
        {
            "metric":
                "n_matched_samples",
            "value":
                len(
                    matched_samples
                ),
        },
        {
            "metric":
                "n_early",
            "value":
                int(
                    np.sum(
                        metabric_y
                        ==
                        0
                    )
                ),
        },
        {
            "metric":
                "n_advanced",
            "value":
                int(
                    np.sum(
                        metabric_y
                        ==
                        1
                    )
                ),
        },
        {
            "metric":
                "n_expression_genes",
            "value":
                metabric_X.shape[1],
        },
        {
            "metric":
                "mean_fold_auc",
            "value":
                metabric_fold_results[
                    "auc"
                ].mean(),
        },
        {
            "metric":
                "sd_fold_auc",
            "value":
                metabric_fold_results[
                    "auc"
                ].std(),
        },
        {
            "metric":
                "patient_mean_oof_auc",
            "value":
                metabric_oof_auc,
        },
    ])

    metabric_summary.to_csv(
        STRESS_DIR
        / "summaries"
        / "metabric_stress_summary.tsv",
        sep="\t",
        index=False,
    )

    metabric_fold_results.to_csv(
        STRESS_DIR
        / "summaries"
        / "metabric_fold_results.tsv",
        sep="\t",
        index=False,
    )

    metabric_selected_gene_frequency.to_csv(
        STRESS_DIR
        / "summaries"
        / "metabric_selected_gene_frequency.tsv",
        sep="\t",
        index=False,
    )

    metabric_gene_overlap.to_csv(
        STRESS_DIR
        / "summaries"
        / "metabric_brca_gene_overlap.tsv",
        sep="\t",
        index=False,
    )

    metabric_core_projection.to_csv(
        STRESS_DIR
        / "summaries"
        / "metabric_stable_core_projection.tsv",
        sep="\t",
        index=False,
    )

    display(
        metabric_summary
    )

    display(
        metabric_gene_overlap
    )

    display(
        metabric_core_projection
    )


# %% [CELL 23] ============================================================
# FINAL MANIFEST AND COMPLETION REPORT
# ============================================================

final_manifest = {
    "analysis":
        "AIDO-BBA BRCA 1.0 final experiments",
    "run_directory":
        str(RUN_DIR),
    "part_A":
        {
            "name":
                "Full-pool patient-specific target reconstruction",
            "generic_frequency_threshold":
                GENERIC_FREQUENCY_THRESHOLD,
            "recurrent_frequency_threshold":
                RECURRENT_FREQUENCY_THRESHOLD,
            "top_generic_per_patient":
                TOP_GENERIC_PER_PATIENT,
            "top_recurrent_per_patient":
                TOP_RECURRENT_PER_PATIENT,
            "top_patient_specific_per_patient":
                TOP_PATIENT_SPECIFIC_PER_PATIENT,
            "top_total_per_patient":
                TOP_TOTAL_PER_PATIENT,
        },
    "part_B":
        {
            "name":
                "Recommendation null audit",
            "n_permutations_per_null":
                N_NULL_PERMUTATIONS,
            "random_seed":
                NULL_RANDOM_SEED,
            "top_k":
                NULL_TOP_K,
        },
    "part_C":
        {
            "name":
                "METABRIC dataset-replacement stress test",
            "requested":
                RUN_METABRIC_STRESS_IF_AVAILABLE,
            "available":
                METABRIC_AVAILABLE,
            "expression_path":
                (
                    str(
                        resolved_metabric_ge
                    )
                    if resolved_metabric_ge
                    is not None
                    else None
                ),
            "clinical_path":
                (
                    str(
                        resolved_metabric_clinical
                    )
                    if resolved_metabric_clinical
                    is not None
                    else None
                ),
        },
    "interpretation_boundary":
        (
            "Outputs identify computational explanatory and "
            "measurement priorities. They do not define "
            "biological subtypes, prescribe clinical tests, "
            "infer prognosis, or recommend treatment."
        ),
}

with open(
    RUN_DIR
    / "AIDO_BBA_BRCA_FINAL_EXPERIMENTS_manifest.json",
    "w",
    encoding="utf-8",
) as handle:
    json.dump(
        final_manifest,
        handle,
        indent=2,
    )

print("\n" + "=" * 88)
print("AIDO-BBA BRCA FINAL EXPERIMENTS COMPLETED")
print("=" * 88)

print("\nPART A output:")
print(FULL_POOL_DIR)

print("\nPART B output:")
print(NULL_DIR)

print("\nPART C output:")
print(STRESS_DIR)

print("\nMETABRIC stress executed:")
print(METABRIC_AVAILABLE)

print("\nMain Part A summary:")
display(part_a_summary)

print("\nMain Part B comparison:")
display(null_comparison)

if METABRIC_AVAILABLE:
    print("\nMain Part C summary:")
    display(metabric_summary)
