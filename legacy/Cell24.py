# ============================================================
# AIDO-BBA BRCA 1.0
# CELL 24-FIX
# Corrected stable representation-gap gene audit
#
# Critical correction:
# NCBI rescue is applied ONLY to genes whose original
# HGNC mapping_status == "unresolved_retained".
# ============================================================

from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import re
import json
import warnings

warnings.filterwarnings("ignore")


# ============================================================
# 0. SETTINGS
# ============================================================

OUTPUT_ROOT = Path(
    r"D:\AIDO-Temp\AIDO_BBA_BRCA_1_0"
)

GO_BP_GMT = Path(
    r"D:\AIDO-Data\GSEA\c5.go.bp.v2026.1.Hs.symbols.gmt"
)

NCBI_GENE_INFO = Path(
    r"D:\AIDO-Data\NCBI_Gene\Homo_sapiens.gene_info"
)

N_EXPECTED_REPEATS = 5
N_EXPECTED_FOLDS = 25

MIN_FOLDS_FOR_SUPPORTED_GENE = 5
MIN_PATIENT_REPEATS_FOR_STABLE_ATTRIBUTION = 2

TOP_RESIDUAL_GENES_FOR_MATRIX = 250


# ============================================================
# 1. FIND COMPLETED RUN
# ============================================================

candidate_runs = []

for run_dir in OUTPUT_ROOT.iterdir():

    if not run_dir.is_dir():
        continue

    required_paths = [
        run_dir / "05_attribution" / "shap_matrices",

        (
            run_dir
            / "06_bp_reconstruction"
            / "summaries"
            / "model_gene_harmonization_audit.tsv"
        ),

        (
            run_dir
            / "06_bp_reconstruction"
            / "summaries"
            / "eligible_go_bp_universe.tsv"
        ),

        (
            run_dir
            / "10_null_corrected_completeness"
            / "patient_null_corrected_completeness.tsv"
        ),

        (
            run_dir
            / "10_null_corrected_completeness"
            / "null_corrected_state_omnibus_tests.tsv"
        ),

        (
            run_dir
            / "10_null_corrected_completeness"
            / "null_corrected_state_pairwise_tests.tsv"
        )
    ]

    if all(
        path.exists()
        for path in required_paths
    ):
        candidate_runs.append(
            run_dir
        )

if len(candidate_runs) == 0:

    raise FileNotFoundError(
        "No completed null-corrected AIDO-BBA run found."
    )

RUN_DIR = sorted(
    candidate_runs,
    key=lambda path: path.stat().st_mtime,
    reverse=True
)[0]

SHAP_MATRIX_DIR = (
    RUN_DIR
    / "05_attribution"
    / "shap_matrices"
)

BP_SUMMARY_DIR = (
    RUN_DIR
    / "06_bp_reconstruction"
    / "summaries"
)

NULL_CORRECTED_DIR = (
    RUN_DIR
    / "10_null_corrected_completeness"
)

GAP_DIR = (
    RUN_DIR
    / "11_representation_gap_genes_corrected"
)

GAP_SUMMARY_DIR = (
    GAP_DIR
    / "summaries"
)

GAP_MATRIX_DIR = (
    GAP_DIR
    / "matrices"
)

for directory in [
    GAP_DIR,
    GAP_SUMMARY_DIR,
    GAP_MATRIX_DIR
]:
    directory.mkdir(
        parents=True,
        exist_ok=True
    )


print("=" * 80)
print("AIDO-BBA CORRECTED REPRESENTATION-GAP GENE AUDIT")
print("=" * 80)

print("\nRun:")
print(RUN_DIR)


# ============================================================
# 2. UTILITIES
# ============================================================

def clean_symbol(value):

    if pd.isna(value):
        return None

    text = str(value).strip().upper()

    if text in {
        "",
        "-",
        "NA",
        "NAN",
        "NONE",
        "NULL"
    }:
        return None

    return text


def benjamini_hochberg(p_values):

    p_values = np.asarray(
        p_values,
        dtype=float
    )

    adjusted = np.full(
        len(p_values),
        np.nan,
        dtype=float
    )

    valid_mask = np.isfinite(
        p_values
    )

    valid_indices = np.where(
        valid_mask
    )[0]

    valid_p = p_values[
        valid_mask
    ]

    if len(valid_p) == 0:
        return adjusted

    order = np.argsort(
        valid_p
    )

    ordered_p = valid_p[
        order
    ]

    ranks = np.arange(
        1,
        len(ordered_p) + 1
    )

    ordered_adjusted = (
        ordered_p
        * len(ordered_p)
        / ranks
    )

    ordered_adjusted = np.minimum.accumulate(
        ordered_adjusted[::-1]
    )[::-1]

    ordered_adjusted = np.clip(
        ordered_adjusted,
        0,
        1
    )

    restored = np.empty(
        len(valid_p),
        dtype=float
    )

    restored[
        order
    ] = ordered_adjusted

    adjusted[
        valid_indices
    ] = restored

    return adjusted


def sign_consistency(values):

    values = np.asarray(
        values,
        dtype=float
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return np.nan

    positive_fraction = np.mean(
        values > 0
    )

    negative_fraction = np.mean(
        values < 0
    )

    return float(
        max(
            positive_fraction,
            negative_fraction
        )
    )


def dominant_direction(values):

    values = np.asarray(
        values,
        dtype=float
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return "unknown"

    positive_fraction = np.mean(
        values > 0
    )

    negative_fraction = np.mean(
        values < 0
    )

    if positive_fraction > negative_fraction:
        return "toward_advanced"

    if negative_fraction > positive_fraction:
        return "toward_early"

    return "mixed"


# ============================================================
# 3. LOAD ORIGINAL HGNC AUDIT
# ============================================================

gene_harmonization = pd.read_csv(
    BP_SUMMARY_DIR
    / "model_gene_harmonization_audit.tsv",
    sep="\t",
    dtype=str
)

required_columns = [
    "normalized_symbol",
    "harmonized_symbol",
    "mapping_status"
]

missing_columns = [
    column
    for column in required_columns
    if column not in gene_harmonization.columns
]

if missing_columns:

    raise ValueError(
        "Missing harmonization columns: "
        + ", ".join(
            missing_columns
        )
    )


# ============================================================
# 4. BUILD BASE HGNC MAPPING
# ============================================================

raw_to_harmonized = {}
raw_to_mapping_status = {}

for _, row in gene_harmonization.iterrows():

    raw_symbol = clean_symbol(
        row.get("normalized_symbol")
    )

    harmonized_symbol = clean_symbol(
        row.get("harmonized_symbol")
    )

    mapping_status = str(
        row.get("mapping_status")
    )

    if raw_symbol is None:
        continue

    raw_to_harmonized[
        raw_symbol
    ] = harmonized_symbol

    raw_to_mapping_status[
        raw_symbol
    ] = mapping_status


# ============================================================
# 5. LOAD NCBI SYNONYMS
# ============================================================

with open(
    NCBI_GENE_INFO,
    "r",
    encoding="utf-8"
) as handle:

    ncbi_header = (
        handle.readline()
        .rstrip("\n")
        .lstrip("#")
        .split("\t")
    )

ncbi = pd.read_csv(
    NCBI_GENE_INFO,
    sep="\t",
    skiprows=1,
    names=ncbi_header,
    dtype=str,
    low_memory=False
)

ncbi_candidate_map = defaultdict(set)

for _, row in ncbi.iterrows():

    official_symbol = clean_symbol(
        row.get("Symbol")
    )

    if official_symbol is None:
        continue

    synonym_field = row.get(
        "Synonyms"
    )

    if pd.isna(
        synonym_field
    ):
        continue

    for synonym in str(
        synonym_field
    ).split("|"):

        synonym = clean_symbol(
            synonym
        )

        if (
            synonym is not None
            and synonym != official_symbol
        ):

            ncbi_candidate_map[
                synonym
            ].add(
                official_symbol
            )

ncbi_unique_map = {
    alias: next(iter(targets))
    for alias, targets
    in ncbi_candidate_map.items()
    if len(targets) == 1
}

ncbi_ambiguous_map = {
    alias: sorted(targets)
    for alias, targets
    in ncbi_candidate_map.items()
    if len(targets) > 1
}


# ============================================================
# 6. CORRECT NCBI RESCUE
# ============================================================

ncbi_rescue_records = []

for raw_symbol, mapping_status in (
    raw_to_mapping_status.items()
):

    if mapping_status != "unresolved_retained":
        continue

    if raw_symbol in ncbi_unique_map:

        rescued_symbol = ncbi_unique_map[
            raw_symbol
        ]

        raw_to_harmonized[
            raw_symbol
        ] = rescued_symbol

        rescue_status = (
            "ncbi_unique_rescue"
        )

    elif raw_symbol in ncbi_ambiguous_map:

        rescued_symbol = None
        rescue_status = "ncbi_ambiguous"

    else:

        rescued_symbol = None
        rescue_status = "still_unresolved"

    ncbi_rescue_records.append({
        "raw_symbol":
            raw_symbol,

        "original_mapping_status":
            mapping_status,

        "ncbi_rescued_symbol":
            rescued_symbol,

        "ncbi_rescue_status":
            rescue_status
    })

ncbi_rescue_audit = pd.DataFrame(
    ncbi_rescue_records
)

ncbi_rescue_summary = (
    ncbi_rescue_audit[
        "ncbi_rescue_status"
    ]
    .value_counts(
        dropna=False
    )
    .rename_axis(
        "ncbi_rescue_status"
    )
    .reset_index(
        name="n"
    )
)

n_ncbi_rescued = int(
    (
        ncbi_rescue_audit[
            "ncbi_rescue_status"
        ]
        == "ncbi_unique_rescue"
    ).sum()
)

print("\nCorrected NCBI rescue:")
display(
    ncbi_rescue_summary
)

print(
    "Unique NCBI rescues integrated:",
    n_ncbi_rescued
)

if n_ncbi_rescued > 100:

    raise RuntimeError(
        "NCBI rescue count is unexpectedly high. "
        "Stop and inspect mapping logic."
    )


# ============================================================
# 7. REBUILD ELIGIBLE GO-BP GENE UNIVERSE
# ============================================================

eligible_bp = pd.read_csv(
    BP_SUMMARY_DIR
    / "eligible_go_bp_universe.tsv",
    sep="\t"
)

eligible_bp_names = set(
    eligible_bp[
        "term_name"
    ].astype(str)
)

eligible_go_gene_set = set()

with open(
    GO_BP_GMT,
    "r",
    encoding="utf-8"
) as handle:

    for line in handle:

        fields = line.rstrip(
            "\n"
        ).split("\t")

        if len(fields) < 3:
            continue

        term_name = fields[0].strip()

        if term_name not in eligible_bp_names:
            continue

        for gene in fields[2:]:

            cleaned_gene = clean_symbol(
                gene
            )

            if cleaned_gene is not None:

                eligible_go_gene_set.add(
                    cleaned_gene
                )

print(
    "\nGenes represented in eligible GO-BP space:",
    len(
        eligible_go_gene_set
    )
)


# ============================================================
# 8. LOAD PATIENT DATA
# ============================================================

patient_null_corrected = pd.read_csv(
    NULL_CORRECTED_DIR
    / "patient_null_corrected_completeness.tsv",
    sep="\t"
)

patient_ids_all = (
    patient_null_corrected[
        "patient_id"
    ]
    .astype(str)
    .tolist()
)

print(
    "Null-corrected patients:",
    len(
        patient_ids_all
    )
)


# ============================================================
# 9. EXTRACT HELD-OUT GAP-GENE ATTRIBUTIONS
# ============================================================

shap_files = sorted(
    SHAP_MATRIX_DIR.glob(
        "extratrees_shap_repeat_*_fold_*.npz"
    )
)

if len(shap_files) != N_EXPECTED_FOLDS:

    raise ValueError(
        f"Expected {N_EXPECTED_FOLDS} SHAP files; "
        f"found {len(shap_files)}."
    )

patient_repeat_gene_records = []
fold_gene_records = []
fold_summary_records = []

print(
    "\nExtracting corrected held-out "
    "representation-gap attributions..."
)

for file_number, shap_file in enumerate(
    shap_files,
    start=1
):

    match = re.search(
        r"repeat_(\d+)_fold_(\d+)",
        shap_file.name
    )

    if match is None:

        raise ValueError(
            f"Cannot parse fold file:\n{shap_file}"
        )

    repeat_id = int(
        match.group(1)
    )

    fold_id = int(
        match.group(2)
    )

    with np.load(
        shap_file,
        allow_pickle=False
    ) as data:

        fold_patient_ids = (
            data["patient_ids"]
            .astype(str)
        )

        selected_genes_raw = (
            data["selected_genes"]
            .astype(str)
        )

        shap_values = (
            data[
                "shap_values_advanced"
            ]
            .astype(
                np.float64
            )
        )

    harmonized_genes = []

    for raw_gene in selected_genes_raw:

        cleaned_raw = clean_symbol(
            raw_gene
        )

        harmonized_gene = (
            raw_to_harmonized.get(
                cleaned_raw,
                cleaned_raw
            )
        )

        harmonized_genes.append(
            harmonized_gene
        )

    harmonized_genes = np.asarray(
        harmonized_genes,
        dtype=object
    )

    gap_feature_mask = np.asarray(
        [
            gene not in eligible_go_gene_set
            if gene is not None
            else True

            for gene in harmonized_genes
        ],
        dtype=bool
    )

    gap_feature_indices = np.where(
        gap_feature_mask
    )[0]

    n_gap_features = int(
        gap_feature_mask.sum()
    )

    fold_summary_records.append({
        "repeat_id":
            repeat_id,

        "fold_id":
            fold_id,

        "n_selected_genes":
            len(
                selected_genes_raw
            ),

        "n_representation_gap_genes":
            n_gap_features,

        "fraction_representation_gap_genes":
            float(
                n_gap_features
                /
                len(
                    selected_genes_raw
                )
            )
    })

    for feature_index in gap_feature_indices:

        raw_gene = str(
            selected_genes_raw[
                feature_index
            ]
        )

        harmonized_gene = (
            harmonized_genes[
                feature_index
            ]
        )

        gene_shap_values = (
            shap_values[
                :,
                feature_index
            ]
        )

        fold_gene_records.append({
            "repeat_id":
                repeat_id,

            "fold_id":
                fold_id,

            "raw_gene_id":
                raw_gene,

            "harmonized_gene_id":
                harmonized_gene,

            "n_test_patients":
                len(
                    fold_patient_ids
                ),

            "mean_absolute_shap_when_selected":
                float(
                    np.mean(
                        np.abs(
                            gene_shap_values
                        )
                    )
                ),

            "mean_signed_shap_when_selected":
                float(
                    np.mean(
                        gene_shap_values
                    )
                ),

            "sign_consistency_when_selected":
                sign_consistency(
                    gene_shap_values
                ),

            "dominant_direction_when_selected":
                dominant_direction(
                    gene_shap_values
                )
        })

        for patient_id, shap_value in zip(
            fold_patient_ids,
            gene_shap_values
        ):

            patient_repeat_gene_records.append({
                "patient_id":
                    patient_id,

                "repeat_id":
                    repeat_id,

                "fold_id":
                    fold_id,

                "raw_gene_id":
                    raw_gene,

                "harmonized_gene_id":
                    harmonized_gene,

                "shap_value":
                    float(
                        shap_value
                    ),

                "absolute_shap_value":
                    float(
                        abs(
                            shap_value
                        )
                    )
            })

    print(
        f"[{file_number:>2}/{len(shap_files)}] "
        f"repeat {repeat_id}, fold {fold_id}: "
        f"{n_gap_features} corrected gap genes"
    )


patient_repeat_gene = pd.DataFrame(
    patient_repeat_gene_records
)

fold_gene_audit = pd.DataFrame(
    fold_gene_records
)

fold_gap_summary = pd.DataFrame(
    fold_summary_records
)


# ============================================================
# 10. GLOBAL GAP-GENE STABILITY
# ============================================================

global_gap_gene_stability = (
    fold_gene_audit
    .groupby(
        [
            "raw_gene_id",
            "harmonized_gene_id"
        ],
        dropna=False,
        as_index=False
    )
    .agg(
        n_folds_selected_as_gap=(
            "fold_id",
            "size"
        ),

        n_repeats_selected_as_gap=(
            "repeat_id",
            "nunique"
        ),

        conditional_mean_absolute_shap=(
            "mean_absolute_shap_when_selected",
            "mean"
        ),

        conditional_median_absolute_shap=(
            "mean_absolute_shap_when_selected",
            "median"
        ),

        conditional_sd_absolute_shap=(
            "mean_absolute_shap_when_selected",
            "std"
        ),

        conditional_mean_signed_shap=(
            "mean_signed_shap_when_selected",
            "mean"
        ),

        fold_direction_consistency=(
            "mean_signed_shap_when_selected",
            sign_consistency
        ),

        dominant_fold_direction=(
            "mean_signed_shap_when_selected",
            dominant_direction
        )
    )
)

global_gap_gene_stability[
    "fold_selection_frequency"
] = (
    global_gap_gene_stability[
        "n_folds_selected_as_gap"
    ]
    / N_EXPECTED_FOLDS
)

global_gap_gene_stability[
    "repeat_selection_frequency"
] = (
    global_gap_gene_stability[
        "n_repeats_selected_as_gap"
    ]
    / N_EXPECTED_REPEATS
)

global_gap_gene_stability[
    "pipeline_mean_absolute_shap"
] = (
    global_gap_gene_stability[
        "conditional_mean_absolute_shap"
    ]
    *
    global_gap_gene_stability[
        "fold_selection_frequency"
    ]
)

global_gap_gene_stability[
    "pipeline_mean_signed_shap"
] = (
    global_gap_gene_stability[
        "conditional_mean_signed_shap"
    ]
    *
    global_gap_gene_stability[
        "fold_selection_frequency"
    ]
)

global_gap_gene_stability[
    "conditional_absolute_shap_cv"
] = (
    global_gap_gene_stability[
        "conditional_sd_absolute_shap"
    ]
    /
    global_gap_gene_stability[
        "conditional_mean_absolute_shap"
    ].replace(
        0,
        np.nan
    )
)


# ============================================================
# 11. SUPPORT TIERS
# ============================================================

supported_mask = (
    global_gap_gene_stability[
        "n_folds_selected_as_gap"
    ]
    >= MIN_FOLDS_FOR_SUPPORTED_GENE
)

supported_values = (
    global_gap_gene_stability.loc[
        supported_mask,
        "pipeline_mean_absolute_shap"
    ]
)

pipeline_q75 = (
    float(
        supported_values.quantile(
            0.75
        )
    )
    if len(
        supported_values
    ) > 0
    else np.nan
)

pipeline_q90 = (
    float(
        supported_values.quantile(
            0.90
        )
    )
    if len(
        supported_values
    ) > 0
    else np.nan
)

global_gap_gene_stability[
    "representation_gap_support_tier"
] = np.select(
    [
        (
            global_gap_gene_stability[
                "n_folds_selected_as_gap"
            ] >= 10
        )
        &
        (
            global_gap_gene_stability[
                "pipeline_mean_absolute_shap"
            ] >= pipeline_q90
        )
        &
        (
            global_gap_gene_stability[
                "fold_direction_consistency"
            ] >= 0.80
        ),

        (
            global_gap_gene_stability[
                "n_folds_selected_as_gap"
            ] >= 5
        )
        &
        (
            global_gap_gene_stability[
                "pipeline_mean_absolute_shap"
            ] >= pipeline_q75
        )
    ],
    [
        "high_support_gap_gene",
        "moderate_support_gap_gene"
    ],
    default="limited_support_gap_gene"
)

global_gap_gene_stability = (
    global_gap_gene_stability
    .sort_values(
        [
            "pipeline_mean_absolute_shap",
            "fold_selection_frequency",
            "fold_direction_consistency"
        ],
        ascending=[
            False,
            False,
            False
        ]
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# 12. PATIENT-GENE STABILITY
# ============================================================

patient_gene_selected = (
    patient_repeat_gene
    .groupby(
        [
            "patient_id",
            "raw_gene_id",
            "harmonized_gene_id"
        ],
        dropna=False,
        as_index=False
    )
    .agg(
        n_repeats_selected=(
            "repeat_id",
            "nunique"
        ),

        conditional_mean_signed_shap=(
            "shap_value",
            "mean"
        ),

        conditional_median_signed_shap=(
            "shap_value",
            "median"
        ),

        conditional_mean_absolute_shap=(
            "absolute_shap_value",
            "mean"
        ),

        conditional_sd_signed_shap=(
            "shap_value",
            "std"
        ),

        attribution_sign_consistency=(
            "shap_value",
            sign_consistency
        ),

        dominant_attribution_direction=(
            "shap_value",
            dominant_direction
        )
    )
)

patient_gene_selected[
    "repeat_selection_frequency"
] = (
    patient_gene_selected[
        "n_repeats_selected"
    ]
    / N_EXPECTED_REPEATS
)

patient_gene_selected[
    "pipeline_mean_signed_shap"
] = (
    patient_gene_selected[
        "conditional_mean_signed_shap"
    ]
    *
    patient_gene_selected[
        "repeat_selection_frequency"
    ]
)

patient_gene_selected[
    "pipeline_mean_absolute_shap"
] = (
    patient_gene_selected[
        "conditional_mean_absolute_shap"
    ]
    *
    patient_gene_selected[
        "repeat_selection_frequency"
    ]
)

patient_gene_selected[
    "stable_patient_gene_attribution"
] = (
    (
        patient_gene_selected[
            "n_repeats_selected"
        ]
        >=
        MIN_PATIENT_REPEATS_FOR_STABLE_ATTRIBUTION
    )
    &
    (
        patient_gene_selected[
            "attribution_sign_consistency"
        ]
        >= 0.80
    )
)


# ============================================================
# 13. BUILD DOWNSTREAM MATRIX
# ============================================================

supported_gap_genes = (
    global_gap_gene_stability[
        global_gap_gene_stability[
            "n_folds_selected_as_gap"
        ]
        >=
        MIN_FOLDS_FOR_SUPPORTED_GENE
    ]
    .copy()
)

selected_matrix_genes = (
    supported_gap_genes
    .head(
        TOP_RESIDUAL_GENES_FOR_MATRIX
    )[
        "raw_gene_id"
    ]
    .astype(str)
    .tolist()
)

patient_gene_matrix_long = (
    patient_gene_selected[
        patient_gene_selected[
            "raw_gene_id"
        ]
        .astype(str)
        .isin(
            selected_matrix_genes
        )
    ][
        [
            "patient_id",
            "raw_gene_id",
            "pipeline_mean_signed_shap"
        ]
    ]
)

patient_gene_matrix = (
    patient_gene_matrix_long
    .pivot_table(
        index="patient_id",
        columns="raw_gene_id",
        values="pipeline_mean_signed_shap",
        aggfunc="sum",
        fill_value=0.0
    )
)

patient_gene_matrix = (
    patient_gene_matrix
    .reindex(
        index=patient_ids_all,
        columns=selected_matrix_genes,
        fill_value=0.0
    )
)

patient_gene_matrix.index.name = (
    "patient_id"
)

patient_gene_matrix.columns.name = (
    "representation_gap_gene"
)


# ============================================================
# 14. FDR CORRECTION
# ============================================================

corrected_omnibus_tests = pd.read_csv(
    NULL_CORRECTED_DIR
    / "null_corrected_state_omnibus_tests.tsv",
    sep="\t"
)

corrected_pairwise_tests = pd.read_csv(
    NULL_CORRECTED_DIR
    / "null_corrected_state_pairwise_tests.tsv",
    sep="\t"
)

corrected_omnibus_tests[
    "fdr_bh_global"
] = benjamini_hochberg(
    corrected_omnibus_tests[
        "p_value"
    ].to_numpy()
)

corrected_pairwise_tests[
    "fdr_within_state_metric"
] = np.nan

for (
    state_variable,
    metric
), row_indices in (
    corrected_pairwise_tests
    .groupby(
        [
            "state_variable",
            "metric"
        ]
    )
    .groups
    .items()
):

    row_indices = list(
        row_indices
    )

    corrected_pairwise_tests.loc[
        row_indices,
        "fdr_within_state_metric"
    ] = benjamini_hochberg(
        corrected_pairwise_tests.loc[
            row_indices,
            "p_value"
        ].to_numpy()
    )

corrected_pairwise_tests[
    "fdr_global"
] = benjamini_hochberg(
    corrected_pairwise_tests[
        "p_value"
    ].to_numpy()
)


# ============================================================
# 15. SAVE OUTPUTS
# ============================================================

ncbi_rescue_audit.to_csv(
    GAP_SUMMARY_DIR
    / "corrected_ncbi_rescue_audit.tsv",
    sep="\t",
    index=False
)

ncbi_rescue_summary.to_csv(
    GAP_SUMMARY_DIR
    / "corrected_ncbi_rescue_summary.tsv",
    sep="\t",
    index=False
)

fold_gap_summary.to_csv(
    GAP_SUMMARY_DIR
    / "fold_representation_gap_summary.tsv",
    sep="\t",
    index=False
)

fold_gene_audit.to_csv(
    GAP_SUMMARY_DIR
    / "fold_gap_gene_attribution_audit.tsv",
    sep="\t",
    index=False
)

patient_repeat_gene.to_csv(
    GAP_SUMMARY_DIR
    / "patient_repeat_gap_gene_attributions.tsv",
    sep="\t",
    index=False
)

global_gap_gene_stability.to_csv(
    GAP_SUMMARY_DIR
    / "global_representation_gap_gene_stability.tsv",
    sep="\t",
    index=False
)

patient_gene_selected.to_csv(
    GAP_SUMMARY_DIR
    / "patient_gap_gene_attribution_stability.tsv",
    sep="\t",
    index=False
)

supported_gap_genes.to_csv(
    GAP_SUMMARY_DIR
    / "supported_representation_gap_genes.tsv",
    sep="\t",
    index=False
)

patient_gene_matrix.to_csv(
    GAP_MATRIX_DIR
    / "patient_by_gap_gene_pipeline_shap.tsv",
    sep="\t",
    index=True
)

np.savez_compressed(
    GAP_MATRIX_DIR
    / "patient_by_gap_gene_pipeline_shap.npz",

    patient_ids=np.asarray(
        patient_gene_matrix.index,
        dtype=str
    ),

    genes=np.asarray(
        patient_gene_matrix.columns,
        dtype=str
    ),

    pipeline_shap_matrix=np.asarray(
        patient_gene_matrix.to_numpy(
            dtype=np.float32
        ),
        dtype=np.float32
    )
)

corrected_omnibus_tests.to_csv(
    GAP_SUMMARY_DIR
    / "null_corrected_state_omnibus_tests_with_fdr.tsv",
    sep="\t",
    index=False
)

corrected_pairwise_tests.to_csv(
    GAP_SUMMARY_DIR
    / "null_corrected_state_pairwise_tests_with_fdr.tsv",
    sep="\t",
    index=False
)


# ============================================================
# 16. SUMMARY
# ============================================================

summary_table = pd.DataFrame([
    {
        "metric":
            "corrected_ncbi_unique_rescues",

        "value":
            n_ncbi_rescued
    },
    {
        "metric":
            "n_unique_gap_genes",

        "value":
            global_gap_gene_stability[
                "raw_gene_id"
            ].nunique()
    },
    {
        "metric":
            "n_supported_gap_genes_five_or_more_folds",

        "value":
            int(
                (
                    global_gap_gene_stability[
                        "n_folds_selected_as_gap"
                    ]
                    >= 5
                ).sum()
            )
    },
    {
        "metric":
            "n_high_support_gap_genes",

        "value":
            int(
                (
                    global_gap_gene_stability[
                        "representation_gap_support_tier"
                    ]
                    == "high_support_gap_gene"
                ).sum()
            )
    },
    {
        "metric":
            "n_moderate_support_gap_genes",

        "value":
            int(
                (
                    global_gap_gene_stability[
                        "representation_gap_support_tier"
                    ]
                    == "moderate_support_gap_gene"
                ).sum()
            )
    },
    {
        "metric":
            "patient_gene_stability_rows",

        "value":
            len(
                patient_gene_selected
            )
    },
    {
        "metric":
            "stable_patient_gene_attributions",

        "value":
            int(
                patient_gene_selected[
                    "stable_patient_gene_attribution"
                ].sum()
            )
    },
    {
        "metric":
            "matrix_patients",

        "value":
            patient_gene_matrix.shape[0]
    },
    {
        "metric":
            "matrix_genes",

        "value":
            patient_gene_matrix.shape[1]
    },
    {
        "metric":
            "significant_corrected_omnibus_fdr_0_05",

        "value":
            int(
                (
                    corrected_omnibus_tests[
                        "fdr_bh_global"
                    ]
                    <= 0.05
                ).sum()
            )
    },
    {
        "metric":
            "significant_corrected_pairwise_fdr_0_05",

        "value":
            int(
                (
                    corrected_pairwise_tests[
                        "fdr_within_state_metric"
                    ]
                    <= 0.05
                ).sum()
            )
    }
])

summary_table.to_csv(
    GAP_SUMMARY_DIR
    / "corrected_representation_gap_gene_audit_summary.tsv",
    sep="\t",
    index=False
)


# ============================================================
# 17. DISPLAY — SORT BEFORE SUBSETTING
# ============================================================

print("\n" + "=" * 80)
print("CELL 24-FIX COMPLETED")
print("=" * 80)

display(
    summary_table
)

print("\nTop corrected representation-gap genes:")

display(
    global_gap_gene_stability[
        [
            "raw_gene_id",
            "harmonized_gene_id",
            "n_folds_selected_as_gap",
            "fold_selection_frequency",
            "conditional_mean_absolute_shap",
            "pipeline_mean_absolute_shap",
            "pipeline_mean_signed_shap",
            "fold_direction_consistency",
            "dominant_fold_direction",
            "representation_gap_support_tier"
        ]
    ].head(40)
)

print("\nFold-level corrected gap burden:")

display(
    fold_gap_summary.round(5)
)

print(
    "\nStrongest FDR-corrected "
    "null-adjusted state effects:"
)

pairwise_sorted = (
    corrected_pairwise_tests
    .sort_values(
        [
            "fdr_within_state_metric",
            "absolute_cliffs_delta"
        ],
        ascending=[
            True,
            False
        ]
    )
)

display(
    pairwise_sorted[
        [
            "state_variable",
            "metric",
            "group_1",
            "group_2",
            "mean_group_1",
            "mean_group_2",
            "mean_difference",
            "cliffs_delta",
            "absolute_cliffs_delta",
            "p_value",
            "fdr_within_state_metric",
            "fdr_global"
        ]
    ].head(30)
)

print("\nCorrected output directory:")
print(GAP_DIR)