# ============================================================
# AIDO-BBA BRCA 1.0
# ONE-CELL GO-BP STRUCTURED SHAP RECONSTRUCTION
#
# 功能：
# 1. 自動找到已完成 SHAP 的原始 run
# 2. 讀取 GO Biological Process GMT
# 3. HGNC + NCBI gene-symbol harmonization
# 4. 建立 eligible GO-BP universe（K >= 10）
# 5. 將 held-out gene SHAP 分配至 GO BP
# 6. 保留 overlapping BP，但按 gene membership degree 分攤
# 7. 精確重建 ExtraTrees held-out probability
# 8. 計算 patient-level explanatory completeness
# 9. 輸出 top BP explanations、BP stability 與 unmapped residual
# 10. 支援中斷後 resume
# ============================================================

from pathlib import Path
from datetime import datetime
from collections import defaultdict
import json
import re
import time
import warnings

import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix

warnings.filterwarnings("ignore")


# ============================================================
# 0. SETTINGS
# ============================================================

OUTPUT_ROOT = Path(
    r"D:\AIDO-Temp\AIDO_BBA_BRCA_1_0"
)

DATA_ROOT = Path(
    r"D:\AIDO-Data"
)

GO_BP_GMT = (
    DATA_ROOT
    / "GSEA"
    / "c5.go.bp.v2026.1.Hs.symbols.gmt"
)

HGNC_COMPLETE_SET = (
    DATA_ROOT
    / "HGNC"
    / "hgnc_complete_set.txt"
)

NCBI_HUMAN_GENE_INFO = (
    DATA_ROOT
    / "NCBI_Gene"
    / "Homo_sapiens.gene_info"
)

MIN_BP_GENES = 10
TOP_BP_PER_PATIENT = 50

TOP_K_VALUES = [
    5,
    10,
    20,
    30,
    50,
    100
]

RESUME_EXISTING_FOLDS = True
SAVE_FULL_BP_MATRICES = True


# ============================================================
# 1. FIND ORIGINAL COMPLETED SHAP RUN
# ============================================================

candidate_runs = []

for run_dir in OUTPUT_ROOT.iterdir():

    if not run_dir.is_dir():
        continue

    shap_summary = (
        run_dir
        / "05_attribution"
        / "summaries"
        / "shap_fold_additivity_audit.tsv"
    )

    state_file = (
        run_dir
        / "04_blackbox"
        / "bba_patient_state_taxonomy.tsv"
    )

    if shap_summary.exists():

        candidate_runs.append({
            "run_dir": run_dir,
            "has_patient_states": state_file.exists(),
            "modified_time": run_dir.stat().st_mtime
        })

if len(candidate_runs) == 0:

    raise FileNotFoundError(
        "No completed SHAP run was found under:\n"
        f"{OUTPUT_ROOT}"
    )

candidate_runs = sorted(
    candidate_runs,
    key=lambda record: (
        record["has_patient_states"],
        record["modified_time"]
    ),
    reverse=True
)

RUN_DIR = candidate_runs[0]["run_dir"]

ATTRIBUTION_DIR = (
    RUN_DIR
    / "05_attribution"
)

SHAP_MATRIX_DIR = (
    ATTRIBUTION_DIR
    / "shap_matrices"
)

BLACKBOX_DIR = (
    RUN_DIR
    / "04_blackbox"
)

STRUCTURED_DIR = (
    RUN_DIR
    / "06_bp_reconstruction"
)

BP_MATRIX_DIR = (
    STRUCTURED_DIR
    / "bp_matrices"
)

BP_FOLD_DIR = (
    STRUCTURED_DIR
    / "fold_tables"
)

BP_SUMMARY_DIR = (
    STRUCTURED_DIR
    / "summaries"
)

for directory in [
    STRUCTURED_DIR,
    BP_MATRIX_DIR,
    BP_FOLD_DIR,
    BP_SUMMARY_DIR
]:

    directory.mkdir(
        parents=True,
        exist_ok=True
    )

print("=" * 78)
print("AIDO-BBA GO-BP STRUCTURED RECONSTRUCTION")
print("=" * 78)

print("\nSelected run:")
print(RUN_DIR)


# ============================================================
# 2. INPUT CHECK
# ============================================================

required_files = [
    GO_BP_GMT,
    HGNC_COMPLETE_SET,
    NCBI_HUMAN_GENE_INFO
]

for file_path in required_files:

    if not file_path.exists():

        raise FileNotFoundError(
            f"Required file not found:\n{file_path}"
        )

shap_matrix_files = sorted(
    SHAP_MATRIX_DIR.glob(
        "extratrees_shap_repeat_*_fold_*.npz"
    )
)

if len(shap_matrix_files) != 25:

    raise ValueError(
        "Expected 25 SHAP matrices, found "
        f"{len(shap_matrix_files)}."
    )

print("\nSHAP fold matrices:", len(shap_matrix_files))
print("GO BP GMT:", GO_BP_GMT)
print("HGNC:", HGNC_COMPLETE_SET)
print("NCBI:", NCBI_HUMAN_GENE_INFO)


# ============================================================
# 3. GENERAL SYMBOL UTILITIES
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


def split_symbol_field(value):
    """
    HGNC aliases and previous symbols are commonly separated
    by |, comma, or semicolon.
    """

    if pd.isna(value):
        return []

    text = str(value).strip()

    if text == "":
        return []

    parts = re.split(
        r"[|,;]"
        ,
        text
    )

    cleaned = []

    for part in parts:

        symbol = clean_symbol(part)

        if symbol is not None:
            cleaned.append(symbol)

    return cleaned


# ============================================================
# 4. LOAD HGNC
# ============================================================

print("\nLoading HGNC harmonization resources...")

hgnc = pd.read_csv(
    HGNC_COMPLETE_SET,
    sep="\t",
    low_memory=False,
    dtype=str
)

hgnc.columns = [
    str(column).strip()
    for column in hgnc.columns
]

required_hgnc_columns = [
    "symbol"
]

missing_hgnc_columns = [
    column
    for column in required_hgnc_columns
    if column not in hgnc.columns
]

if missing_hgnc_columns:

    raise ValueError(
        "HGNC file lacks required column(s): "
        + ", ".join(missing_hgnc_columns)
    )

approved_symbols = set(
    hgnc["symbol"]
    .dropna()
    .map(clean_symbol)
    .dropna()
)

hgnc_candidate_map = defaultdict(set)

for _, row in hgnc.iterrows():

    approved = clean_symbol(
        row.get("symbol")
    )

    if approved is None:
        continue

    for field_name in [
        "prev_symbol",
        "alias_symbol"
    ]:

        if field_name not in hgnc.columns:
            continue

        for alias in split_symbol_field(
            row.get(field_name)
        ):

            if alias != approved:
                hgnc_candidate_map[
                    alias
                ].add(approved)

hgnc_unique_alias_map = {
    alias: next(iter(targets))
    for alias, targets in hgnc_candidate_map.items()
    if len(targets) == 1
}

hgnc_ambiguous_aliases = {
    alias
    for alias, targets in hgnc_candidate_map.items()
    if len(targets) > 1
}

print("HGNC approved symbols:", len(approved_symbols))
print("HGNC unique aliases:", len(hgnc_unique_alias_map))
print("HGNC ambiguous aliases:", len(hgnc_ambiguous_aliases))


# ============================================================
# 5. LOAD NCBI GENE INFO
# ============================================================

print("\nLoading NCBI gene-info resource...")

ncbi = pd.read_csv(
    NCBI_HUMAN_GENE_INFO,
    sep="\t",
    comment="#",
    dtype=str,
    low_memory=False
)

ncbi.columns = [
    str(column).strip()
    for column in ncbi.columns
]

ncbi_candidate_map = defaultdict(set)

if (
    "Symbol" in ncbi.columns
    and "Synonyms" in ncbi.columns
):

    for _, row in ncbi.iterrows():

        official = clean_symbol(
            row.get("Symbol")
        )

        if official is None:
            continue

        synonyms = split_symbol_field(
            str(
                row.get("Synonyms", "")
            ).replace("|", ";")
        )

        for synonym in synonyms:

            if synonym != official:
                ncbi_candidate_map[
                    synonym
                ].add(official)

ncbi_unique_alias_map = {
    alias: next(iter(targets))
    for alias, targets in ncbi_candidate_map.items()
    if len(targets) == 1
}

ncbi_ambiguous_aliases = {
    alias
    for alias, targets in ncbi_candidate_map.items()
    if len(targets) > 1
}

print("NCBI unique synonyms:", len(ncbi_unique_alias_map))
print("NCBI ambiguous synonyms:", len(ncbi_ambiguous_aliases))


# ============================================================
# 6. SYMBOL HARMONIZATION FUNCTION
# ============================================================

def harmonize_symbol(value):

    original = clean_symbol(value)

    if original is None:

        return {
            "original_symbol": value,
            "normalized_symbol": None,
            "harmonized_symbol": None,
            "mapping_status": "invalid"
        }

    if original in approved_symbols:

        return {
            "original_symbol": value,
            "normalized_symbol": original,
            "harmonized_symbol": original,
            "mapping_status": "hgnc_approved"
        }

    if original in hgnc_ambiguous_aliases:

        return {
            "original_symbol": value,
            "normalized_symbol": original,
            "harmonized_symbol": None,
            "mapping_status": "hgnc_ambiguous"
        }

    if original in hgnc_unique_alias_map:

        return {
            "original_symbol": value,
            "normalized_symbol": original,
            "harmonized_symbol":
                hgnc_unique_alias_map[original],
            "mapping_status": "hgnc_alias_or_previous"
        }

    if original in ncbi_ambiguous_aliases:

        return {
            "original_symbol": value,
            "normalized_symbol": original,
            "harmonized_symbol": None,
            "mapping_status": "ncbi_ambiguous"
        }

    if original in ncbi_unique_alias_map:

        mapped = ncbi_unique_alias_map[
            original
        ]

        return {
            "original_symbol": value,
            "normalized_symbol": original,
            "harmonized_symbol": mapped,
            "mapping_status": "ncbi_synonym"
        }

    return {
        "original_symbol": value,
        "normalized_symbol": original,
        "harmonized_symbol": original,
        "mapping_status": "unresolved_retained"
    }


# ============================================================
# 7. MODEL GENE UNIVERSE
# ============================================================

model_gene_manifest_file = (
    BLACKBOX_DIR
    / "model_gene_manifest.tsv"
)

if model_gene_manifest_file.exists():

    model_gene_manifest = pd.read_csv(
        model_gene_manifest_file,
        sep="\t",
        dtype=str
    )

    model_gene_column = (
        "gene_id"
        if "gene_id" in model_gene_manifest.columns
        else model_gene_manifest.columns[0]
    )

    model_genes = (
        model_gene_manifest[
            model_gene_column
        ]
        .dropna()
        .astype(str)
        .tolist()
    )

else:

    all_selected_genes = []

    for matrix_file in shap_matrix_files:

        with np.load(
            matrix_file,
            allow_pickle=False
        ) as data:

            all_selected_genes.extend(
                data["selected_genes"]
                .astype(str)
                .tolist()
            )

    model_genes = sorted(
        set(all_selected_genes)
    )

gene_harmonization_records = [
    harmonize_symbol(gene)
    for gene in model_genes
]

gene_harmonization_audit = pd.DataFrame(
    gene_harmonization_records
)

gene_harmonization_audit.to_csv(
    BP_SUMMARY_DIR
    / "model_gene_harmonization_audit.tsv",
    sep="\t",
    index=False
)

harmonization_summary = (
    gene_harmonization_audit[
        "mapping_status"
    ]
    .value_counts(dropna=False)
    .rename_axis("mapping_status")
    .reset_index(name="n")
)

print("\nModel-gene harmonization:")
display(harmonization_summary)

raw_to_harmonized = dict(
    zip(
        gene_harmonization_audit[
            "normalized_symbol"
        ],
        gene_harmonization_audit[
            "harmonized_symbol"
        ]
    )
)

harmonized_model_gene_set = set(
    gene_harmonization_audit[
        "harmonized_symbol"
    ]
    .dropna()
    .astype(str)
)


# ============================================================
# 8. LOAD GO BP GMT
# ============================================================

print("\nLoading GO Biological Process GMT...")

go_bp_raw = []

with open(
    GO_BP_GMT,
    "r",
    encoding="utf-8"
) as file:

    for line_number, line in enumerate(
        file,
        start=1
    ):

        fields = line.rstrip("\n").split("\t")

        if len(fields) < 3:
            continue

        term_name = fields[0].strip()
        description = fields[1].strip()

        genes = {
            clean_symbol(gene)
            for gene in fields[2:]
        }

        genes.discard(None)

        go_bp_raw.append({
            "term_name": term_name,
            "description": description,
            "gmt_gene_count": len(genes),
            "genes": genes
        })

print("Raw GO BP terms:", len(go_bp_raw))


# ============================================================
# 9. BUILD ELIGIBLE GO BP UNIVERSE
# ============================================================

eligible_bp_records = []

for record in go_bp_raw:

    matched_genes = sorted(
        record["genes"]
        & harmonized_model_gene_set
    )

    if len(matched_genes) < MIN_BP_GENES:
        continue

    eligible_bp_records.append({
        "term_name": record["term_name"],
        "description": record["description"],
        "gmt_gene_count":
            record["gmt_gene_count"],
        "matched_gene_count":
            len(matched_genes),
        "matched_genes":
            matched_genes
    })

eligible_bp_records = sorted(
    eligible_bp_records,
    key=lambda record: record["term_name"]
)

bp_names = np.asarray(
    [
        record["term_name"]
        for record in eligible_bp_records
    ],
    dtype=str
)

bp_descriptions = np.asarray(
    [
        record["description"]
        for record in eligible_bp_records
    ],
    dtype=str
)

bp_matched_gene_counts = np.asarray(
    [
        record["matched_gene_count"]
        for record in eligible_bp_records
    ],
    dtype=int
)

bp_index_lookup = {
    bp_name: index
    for index, bp_name in enumerate(bp_names)
}

gene_to_bp_indices = defaultdict(list)

for bp_index, record in enumerate(
    eligible_bp_records
):

    for gene in record["matched_genes"]:

        gene_to_bp_indices[
            gene
        ].append(bp_index)

bp_universe_table = pd.DataFrame({
    "bp_index": np.arange(
        len(bp_names)
    ),
    "term_name": bp_names,
    "description": bp_descriptions,
    "matched_gene_count":
        bp_matched_gene_counts
})

bp_universe_table.to_csv(
    BP_SUMMARY_DIR
    / "eligible_go_bp_universe.tsv",
    sep="\t",
    index=False
)

print("\nEligible GO BP terms:", len(bp_names))
print("Minimum matched genes:", MIN_BP_GENES)

if len(bp_names) == 0:

    raise ValueError(
        "No eligible GO BP terms were constructed."
    )


# ============================================================
# 10. FOLD-LEVEL GO BP RECONSTRUCTION
# ============================================================

fold_audit_records = []
global_bp_records = []
patient_completeness_records = []

overall_start = time.time()

print("\n" + "=" * 78)
print("STARTING GO-BP SHAP RECONSTRUCTION")
print("=" * 78)

for matrix_number, matrix_file in enumerate(
    shap_matrix_files,
    start=1
):

    match = re.search(
        r"repeat_(\d+)_fold_(\d+)",
        matrix_file.name
    )

    if match is None:

        raise ValueError(
            f"Cannot parse fold identity:\n{matrix_file}"
        )

    repeat_id = int(
        match.group(1)
    )

    fold_id = int(
        match.group(2)
    )

    fold_prefix = (
        f"repeat_{repeat_id:02d}_"
        f"fold_{fold_id:02d}"
    )

    bp_matrix_file = (
        BP_MATRIX_DIR
        / f"bp_shap_{fold_prefix}.npz"
    )

    completeness_file = (
        BP_FOLD_DIR
        / f"patient_completeness_{fold_prefix}.tsv"
    )

    top_bp_file = (
        BP_FOLD_DIR
        / f"patient_top_bp_{fold_prefix}.tsv"
    )

    global_bp_file = (
        BP_FOLD_DIR
        / f"global_bp_shap_{fold_prefix}.tsv"
    )

    audit_file = (
        BP_FOLD_DIR
        / f"bp_reconstruction_audit_{fold_prefix}.json"
    )

    outputs_exist = all([
        completeness_file.exists(),
        top_bp_file.exists(),
        global_bp_file.exists(),
        audit_file.exists(),
        (
            bp_matrix_file.exists()
            if SAVE_FULL_BP_MATRICES
            else True
        )
    ])

    if (
        RESUME_EXISTING_FOLDS
        and outputs_exist
    ):

        print(
            f"[{matrix_number:>2}/{len(shap_matrix_files)}] "
            f"{fold_prefix} | SKIPPED"
        )

        continue

    fold_start = time.time()

    print(
        f"[{matrix_number:>2}/{len(shap_matrix_files)}] "
        f"{fold_prefix} | RUNNING"
    )

    with np.load(
        matrix_file,
        allow_pickle=False
    ) as data:

        patient_ids_fold = (
            data["patient_ids"]
            .astype(str)
        )

        true_labels_fold = (
            data["true_labels"]
            .astype(int)
        )

        probabilities_fold = (
            data[
                "predicted_probability_advanced"
            ]
            .astype(float)
        )

        expected_value_fold = float(
            data[
                "expected_value_advanced"
            ][0]
        )

        selected_genes_raw = (
            data["selected_genes"]
            .astype(str)
        )

        shap_values_gene = (
            data[
                "shap_values_advanced"
            ]
            .astype(float)
        )

    n_patients_fold = (
        shap_values_gene.shape[0]
    )

    n_selected_features = (
        shap_values_gene.shape[1]
    )

    selected_genes_harmonized = []

    mapping_statuses = []

    for raw_gene in selected_genes_raw:

        mapping_record = harmonize_symbol(
            raw_gene
        )

        selected_genes_harmonized.append(
            mapping_record[
                "harmonized_symbol"
            ]
        )

        mapping_statuses.append(
            mapping_record[
                "mapping_status"
            ]
        )

    selected_genes_harmonized = np.asarray(
        selected_genes_harmonized,
        dtype=object
    )

    # --------------------------------------------------------
    # Construct feature-to-BP allocation matrix
    #
    # Each gene attribution is divided equally across all
    # eligible BPs containing that gene.
    #
    # Therefore mapped SHAP mass is not duplicated merely
    # because GO terms overlap.
    # --------------------------------------------------------

    allocation_rows = []
    allocation_columns = []
    allocation_values = []

    mapped_feature_mask = np.zeros(
        n_selected_features,
        dtype=bool
    )

    feature_bp_degree = np.zeros(
        n_selected_features,
        dtype=int
    )

    for feature_index, harmonized_gene in enumerate(
        selected_genes_harmonized
    ):

        if harmonized_gene is None:
            continue

        memberships = gene_to_bp_indices.get(
            str(harmonized_gene),
            []
        )

        degree = len(memberships)

        feature_bp_degree[
            feature_index
        ] = degree

        if degree == 0:
            continue

        mapped_feature_mask[
            feature_index
        ] = True

        allocation_weight = (
            1.0 / degree
        )

        for bp_index in memberships:

            allocation_rows.append(
                feature_index
            )

            allocation_columns.append(
                bp_index
            )

            allocation_values.append(
                allocation_weight
            )

    allocation_matrix = csr_matrix(
        (
            allocation_values,
            (
                allocation_rows,
                allocation_columns
            )
        ),
        shape=(
            n_selected_features,
            len(bp_names)
        ),
        dtype=np.float64
    )

    # BP SHAP:
    # patients x BPs
    bp_shap = (
        allocation_matrix.T
        .dot(
            shap_values_gene.T
        )
        .T
    )

    bp_shap = np.asarray(
        bp_shap,
        dtype=float
    )

    mapped_signed_contribution = (
        shap_values_gene[
            :,
            mapped_feature_mask
        ]
        .sum(axis=1)
    )

    unmapped_signed_residual = (
        shap_values_gene[
            :,
            ~mapped_feature_mask
        ]
        .sum(axis=1)
    )

    total_signed_contribution = (
        shap_values_gene.sum(axis=1)
    )

    reconstructed_probability = (
        expected_value_fold
        + bp_shap.sum(axis=1)
        + unmapped_signed_residual
    )

    exact_reconstruction_error = (
        reconstructed_probability
        - probabilities_fold
    )

    total_absolute_gene_shap = (
        np.abs(
            shap_values_gene
        )
        .sum(axis=1)
    )

    mapped_absolute_gene_shap = (
        np.abs(
            shap_values_gene[
                :,
                mapped_feature_mask
            ]
        )
        .sum(axis=1)
    )

    unmapped_absolute_gene_shap = (
        np.abs(
            shap_values_gene[
                :,
                ~mapped_feature_mask
            ]
        )
        .sum(axis=1)
    )

    attribution_mass_coverage = np.divide(
        mapped_absolute_gene_shap,
        total_absolute_gene_shap,
        out=np.zeros_like(
            mapped_absolute_gene_shap
        ),
        where=(
            total_absolute_gene_shap > 0
        )
    )

    selected_gene_coverage = float(
        mapped_feature_mask.mean()
    )

    bp_absolute_mass = (
        np.abs(
            bp_shap
        )
        .sum(axis=1)
    )

    # --------------------------------------------------------
    # Top-K BP completeness
    # --------------------------------------------------------

    top_k_metrics = {}

    sorted_bp_indices = np.argsort(
        np.abs(bp_shap),
        axis=1
    )[:, ::-1]

    for top_k in TOP_K_VALUES:

        effective_top_k = min(
            top_k,
            len(bp_names)
        )

        selected_indices = (
            sorted_bp_indices[
                :,
                :effective_top_k
            ]
        )

        row_indices = np.arange(
            n_patients_fold
        )[:, None]

        selected_bp_values = (
            bp_shap[
                row_indices,
                selected_indices
            ]
        )

        top_k_signed_sum = (
            selected_bp_values.sum(
                axis=1
            )
        )

        top_k_absolute_mass = (
            np.abs(
                selected_bp_values
            )
            .sum(axis=1)
        )

        top_k_bp_mass_fraction = np.divide(
            top_k_absolute_mass,
            bp_absolute_mass,
            out=np.zeros_like(
                top_k_absolute_mass
            ),
            where=(
                bp_absolute_mass > 0
            )
        )

        top_k_probability = (
            expected_value_fold
            + top_k_signed_sum
        )

        top_k_absolute_error = np.abs(
            probabilities_fold
            - top_k_probability
        )

        top_k_metrics[
            top_k
        ] = {
            "bp_mass_fraction":
                top_k_bp_mass_fraction,

            "reconstructed_probability":
                top_k_probability,

            "absolute_probability_error":
                top_k_absolute_error
        }

    # --------------------------------------------------------
    # Patient-level completeness table
    # --------------------------------------------------------

    patient_completeness = pd.DataFrame({
        "patient_id":
            patient_ids_fold,

        "repeat_id":
            repeat_id,

        "fold_id":
            fold_id,

        "true_label":
            true_labels_fold,

        "true_group":
            np.where(
                true_labels_fold == 1,
                "Advanced",
                "Early"
            ),

        "predicted_probability_advanced":
            probabilities_fold,

        "expected_value_advanced":
            expected_value_fold,

        "n_selected_genes":
            n_selected_features,

        "n_mapped_selected_genes":
            int(
                mapped_feature_mask.sum()
            ),

        "selected_gene_coverage_fraction":
            selected_gene_coverage,

        "total_absolute_gene_shap":
            total_absolute_gene_shap,

        "mapped_absolute_gene_shap":
            mapped_absolute_gene_shap,

        "unmapped_absolute_gene_shap":
            unmapped_absolute_gene_shap,

        "attribution_mass_coverage":
            attribution_mass_coverage,

        "mapped_signed_contribution":
            mapped_signed_contribution,

        "unmapped_signed_residual":
            unmapped_signed_residual,

        "total_signed_contribution":
            total_signed_contribution,

        "bp_signed_contribution":
            bp_shap.sum(axis=1),

        "bp_absolute_mass":
            bp_absolute_mass,

        "exact_reconstructed_probability":
            reconstructed_probability,

        "exact_reconstruction_error":
            exact_reconstruction_error
    })

    for top_k in TOP_K_VALUES:

        patient_completeness[
            f"top{top_k}_bp_mass_fraction"
        ] = (
            top_k_metrics[
                top_k
            ][
                "bp_mass_fraction"
            ]
        )

        patient_completeness[
            f"top{top_k}_reconstructed_probability"
        ] = (
            top_k_metrics[
                top_k
            ][
                "reconstructed_probability"
            ]
        )

        patient_completeness[
            f"top{top_k}_absolute_probability_error"
        ] = (
            top_k_metrics[
                top_k
            ][
                "absolute_probability_error"
            ]
        )

    patient_completeness.to_csv(
        completeness_file,
        sep="\t",
        index=False
    )

    # --------------------------------------------------------
    # Patient top BP explanations
    # --------------------------------------------------------

    top_bp_records = []

    top_n = min(
        TOP_BP_PER_PATIENT,
        len(bp_names)
    )

    for patient_position, patient_id in enumerate(
        patient_ids_fold
    ):

        top_indices = (
            sorted_bp_indices[
                patient_position,
                :top_n
            ]
        )

        for bp_rank, bp_index in enumerate(
            top_indices,
            start=1
        ):

            contribution = float(
                bp_shap[
                    patient_position,
                    bp_index
                ]
            )

            top_bp_records.append({
                "patient_id":
                    patient_id,

                "repeat_id":
                    repeat_id,

                "fold_id":
                    fold_id,

                "true_label":
                    int(
                        true_labels_fold[
                            patient_position
                        ]
                    ),

                "true_group":
                    (
                        "Advanced"
                        if true_labels_fold[
                            patient_position
                        ] == 1
                        else "Early"
                    ),

                "predicted_probability_advanced":
                    float(
                        probabilities_fold[
                            patient_position
                        ]
                    ),

                "bp_rank":
                    bp_rank,

                "bp_index":
                    int(bp_index),

                "term_name":
                    bp_names[
                        bp_index
                    ],

                "description":
                    bp_descriptions[
                        bp_index
                    ],

                "bp_matched_gene_count":
                    int(
                        bp_matched_gene_counts[
                            bp_index
                        ]
                    ),

                "bp_shap_value":
                    contribution,

                "absolute_bp_shap_value":
                    abs(
                        contribution
                    ),

                "attribution_direction":
                    (
                        "toward_advanced"
                        if contribution > 0
                        else (
                            "toward_early"
                            if contribution < 0
                            else "neutral"
                        )
                    )
            })

    patient_top_bp = pd.DataFrame(
        top_bp_records
    )

    patient_top_bp.to_csv(
        top_bp_file,
        sep="\t",
        index=False
    )

    # --------------------------------------------------------
    # Global BP attribution for this held-out fold
    # --------------------------------------------------------

    global_bp_table = pd.DataFrame({
        "repeat_id":
            repeat_id,

        "fold_id":
            fold_id,

        "bp_index":
            np.arange(
                len(bp_names)
            ),

        "term_name":
            bp_names,

        "description":
            bp_descriptions,

        "matched_gene_count":
            bp_matched_gene_counts,

        "mean_absolute_bp_shap":
            np.mean(
                np.abs(
                    bp_shap
                ),
                axis=0
            ),

        "median_absolute_bp_shap":
            np.median(
                np.abs(
                    bp_shap
                ),
                axis=0
            ),

        "mean_signed_bp_shap":
            np.mean(
                bp_shap,
                axis=0
            ),

        "n_test_patients":
            n_patients_fold
    })

    global_bp_table.to_csv(
        global_bp_file,
        sep="\t",
        index=False
    )

    # --------------------------------------------------------
    # Save full BP matrix
    # --------------------------------------------------------

    if SAVE_FULL_BP_MATRICES:

        np.savez_compressed(
            bp_matrix_file,

            patient_ids=np.asarray(
                patient_ids_fold,
                dtype=str
            ),

            true_labels=np.asarray(
                true_labels_fold,
                dtype=np.int8
            ),

            predicted_probability_advanced=np.asarray(
                probabilities_fold,
                dtype=np.float32
            ),

            expected_value_advanced=np.asarray(
                [expected_value_fold],
                dtype=np.float32
            ),

            bp_names=np.asarray(
                bp_names,
                dtype=str
            ),

            bp_shap_values=np.asarray(
                bp_shap,
                dtype=np.float32
            ),

            unmapped_signed_residual=np.asarray(
                unmapped_signed_residual,
                dtype=np.float32
            ),

            attribution_mass_coverage=np.asarray(
                attribution_mass_coverage,
                dtype=np.float32
            )
        )

    # --------------------------------------------------------
    # Fold audit
    # --------------------------------------------------------

    fold_duration = (
        time.time()
        - fold_start
    )

    fold_audit = {
        "repeat_id":
            repeat_id,

        "fold_id":
            fold_id,

        "n_patients":
            int(
                n_patients_fold
            ),

        "n_selected_genes":
            int(
                n_selected_features
            ),

        "n_mapped_selected_genes":
            int(
                mapped_feature_mask.sum()
            ),

        "selected_gene_coverage_fraction":
            float(
                selected_gene_coverage
            ),

        "n_eligible_bp":
            int(
                len(bp_names)
            ),

        "mean_attribution_mass_coverage":
            float(
                np.mean(
                    attribution_mass_coverage
                )
            ),

        "median_attribution_mass_coverage":
            float(
                np.median(
                    attribution_mass_coverage
                )
            ),

        "mean_absolute_unmapped_residual":
            float(
                np.mean(
                    np.abs(
                        unmapped_signed_residual
                    )
                )
            ),

        "mean_absolute_exact_reconstruction_error":
            float(
                np.mean(
                    np.abs(
                        exact_reconstruction_error
                    )
                )
            ),

        "maximum_absolute_exact_reconstruction_error":
            float(
                np.max(
                    np.abs(
                        exact_reconstruction_error
                    )
                )
            ),

        "reconstruction_correlation":
            float(
                np.corrcoef(
                    reconstructed_probability,
                    probabilities_fold
                )[0, 1]
            ),

        "duration_seconds":
            float(
                fold_duration
            )
    }

    with open(
        audit_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            fold_audit,
            file,
            indent=2
        )


# ============================================================
# 11. COMBINE FOLD OUTPUTS
# ============================================================

print("\n" + "=" * 78)
print("COMBINING GO-BP OUTPUTS")
print("=" * 78)

completeness_files = sorted(
    BP_FOLD_DIR.glob(
        "patient_completeness_repeat_*_fold_*.tsv"
    )
)

top_bp_files = sorted(
    BP_FOLD_DIR.glob(
        "patient_top_bp_repeat_*_fold_*.tsv"
    )
)

global_bp_files = sorted(
    BP_FOLD_DIR.glob(
        "global_bp_shap_repeat_*_fold_*.tsv"
    )
)

audit_files = sorted(
    BP_FOLD_DIR.glob(
        "bp_reconstruction_audit_repeat_*_fold_*.json"
    )
)

if len(completeness_files) != 25:

    raise ValueError(
        "Expected 25 completeness files, found "
        f"{len(completeness_files)}."
    )

patient_completeness_all = pd.concat(
    [
        pd.read_csv(
            file,
            sep="\t"
        )
        for file in completeness_files
    ],
    ignore_index=True
)

patient_top_bp_all = pd.concat(
    [
        pd.read_csv(
            file,
            sep="\t"
        )
        for file in top_bp_files
    ],
    ignore_index=True
)

global_bp_by_fold = pd.concat(
    [
        pd.read_csv(
            file,
            sep="\t"
        )
        for file in global_bp_files
    ],
    ignore_index=True
)

fold_audit_records = []

for file in audit_files:

    with open(
        file,
        "r",
        encoding="utf-8"
    ) as handle:

        fold_audit_records.append(
            json.load(handle)
        )

bp_reconstruction_fold_audit = pd.DataFrame(
    fold_audit_records
)


# ============================================================
# 12. PATIENT-LEVEL COMPLETENESS ACROSS REPEATS
# ============================================================

aggregation_dictionary = {
    "true_label":
        "first",

    "true_group":
        "first",

    "predicted_probability_advanced":
        ["mean", "std"],

    "selected_gene_coverage_fraction":
        ["mean", "std"],

    "attribution_mass_coverage":
        ["mean", "std"],

    "unmapped_signed_residual":
        ["mean", "std"],

    "unmapped_absolute_gene_shap":
        ["mean", "std"],

    "bp_absolute_mass":
        ["mean", "std"],

    "exact_reconstruction_error":
        [
            lambda values:
                np.mean(
                    np.abs(values)
                ),
            lambda values:
                np.max(
                    np.abs(values)
                )
        ]
}

for top_k in TOP_K_VALUES:

    aggregation_dictionary[
        f"top{top_k}_bp_mass_fraction"
    ] = [
        "mean",
        "std"
    ]

    aggregation_dictionary[
        f"top{top_k}_absolute_probability_error"
    ] = [
        "mean",
        "std"
    ]

patient_completeness_summary = (
    patient_completeness_all
    .groupby(
        "patient_id"
    )
    .agg(
        aggregation_dictionary
    )
)

patient_completeness_summary.columns = [
    "_".join(
        [
            str(part)
            for part in column
            if str(part) != ""
        ]
    )
    for column in patient_completeness_summary.columns
]

patient_completeness_summary = (
    patient_completeness_summary
    .reset_index()
)

patient_completeness_summary = (
    patient_completeness_summary
    .rename(
        columns={
            "true_label_first":
                "true_label",

            "true_group_first":
                "true_group",

            "predicted_probability_advanced_mean":
                "mean_probability_advanced",

            "predicted_probability_advanced_std":
                "sd_probability_advanced",

            "selected_gene_coverage_fraction_mean":
                "mean_selected_gene_coverage",

            "selected_gene_coverage_fraction_std":
                "sd_selected_gene_coverage",

            "attribution_mass_coverage_mean":
                "mean_attribution_mass_coverage",

            "attribution_mass_coverage_std":
                "sd_attribution_mass_coverage",

            "unmapped_signed_residual_mean":
                "mean_unmapped_signed_residual",

            "unmapped_signed_residual_std":
                "sd_unmapped_signed_residual",

            "unmapped_absolute_gene_shap_mean":
                "mean_unmapped_absolute_shap",

            "unmapped_absolute_gene_shap_std":
                "sd_unmapped_absolute_shap"
        }
    )
)


# ============================================================
# 13. GLOBAL BP STABILITY
# ============================================================

global_bp_stability = (
    global_bp_by_fold
    .groupby(
        [
            "bp_index",
            "term_name",
            "description",
            "matched_gene_count"
        ],
        as_index=False
    )
    .agg(
        n_folds=(
            "fold_id",
            "size"
        ),

        mean_absolute_bp_shap=(
            "mean_absolute_bp_shap",
            "mean"
        ),

        median_absolute_bp_shap=(
            "mean_absolute_bp_shap",
            "median"
        ),

        sd_absolute_bp_shap=(
            "mean_absolute_bp_shap",
            "std"
        ),

        mean_signed_bp_shap=(
            "mean_signed_bp_shap",
            "mean"
        ),

        median_signed_bp_shap=(
            "mean_signed_bp_shap",
            "median"
        )
    )
)

global_bp_stability[
    "absolute_bp_shap_cv"
] = (
    global_bp_stability[
        "sd_absolute_bp_shap"
    ]
    /
    global_bp_stability[
        "mean_absolute_bp_shap"
    ].replace(
        0,
        np.nan
    )
)

global_bp_stability = (
    global_bp_stability
    .sort_values(
        [
            "mean_absolute_bp_shap",
            "absolute_bp_shap_cv"
        ],
        ascending=[
            False,
            True
        ]
    )
    .reset_index(drop=True)
)


# ============================================================
# 14. PATIENT–BP STABILITY
# ============================================================

patient_bp_stability = (
    patient_top_bp_all
    .groupby(
        [
            "patient_id",
            "term_name",
            "description"
        ],
        as_index=False
    )
    .agg(
        n_repeats_top50=(
            "repeat_id",
            "nunique"
        ),

        mean_absolute_bp_shap=(
            "absolute_bp_shap_value",
            "mean"
        ),

        median_absolute_bp_shap=(
            "absolute_bp_shap_value",
            "median"
        ),

        mean_signed_bp_shap=(
            "bp_shap_value",
            "mean"
        ),

        median_bp_rank=(
            "bp_rank",
            "median"
        ),

        fraction_toward_advanced=(
            "bp_shap_value",
            lambda values:
                float(
                    np.mean(
                        values > 0
                    )
                )
        )
    )
)

patient_bp_stability[
    "top50_repeat_frequency"
] = (
    patient_bp_stability[
        "n_repeats_top50"
    ]
    / 5
)

patient_bp_stability = (
    patient_bp_stability
    .sort_values(
        [
            "patient_id",
            "top50_repeat_frequency",
            "mean_absolute_bp_shap"
        ],
        ascending=[
            True,
            False,
            False
        ]
    )
)


# ============================================================
# 15. MERGE WITH EXISTING BBA PATIENT STATES
# ============================================================

bba_state_file = (
    BLACKBOX_DIR
    / "bba_patient_state_taxonomy.tsv"
)

if bba_state_file.exists():

    bba_patient_states = pd.read_csv(
        bba_state_file,
        sep="\t"
    )

    patient_completeness_with_states = (
        bba_patient_states.merge(
            patient_completeness_summary,
            on=[
                "patient_id",
                "true_label",
                "true_group"
            ],
            how="left",
            validate="one_to_one"
        )
    )

else:

    patient_completeness_with_states = (
        patient_completeness_summary.copy()
    )


# ============================================================
# 16. SAVE FINAL OUTPUTS
# ============================================================

bp_reconstruction_fold_audit.to_csv(
    BP_SUMMARY_DIR
    / "bp_reconstruction_fold_audit.tsv",
    sep="\t",
    index=False
)

patient_completeness_all.to_csv(
    BP_SUMMARY_DIR
    / "patient_completeness_all_repeats.tsv",
    sep="\t",
    index=False
)

patient_completeness_summary.to_csv(
    BP_SUMMARY_DIR
    / "patient_completeness_summary.tsv",
    sep="\t",
    index=False
)

patient_completeness_with_states.to_csv(
    BP_SUMMARY_DIR
    / "patient_completeness_with_bba_states.tsv",
    sep="\t",
    index=False
)

patient_top_bp_all.to_csv(
    BP_SUMMARY_DIR
    / "heldout_patient_top50_bp_attributions.tsv",
    sep="\t",
    index=False
)

global_bp_by_fold.to_csv(
    BP_SUMMARY_DIR
    / "global_bp_shap_by_fold.tsv",
    sep="\t",
    index=False
)

global_bp_stability.to_csv(
    BP_SUMMARY_DIR
    / "global_bp_shap_stability.tsv",
    sep="\t",
    index=False
)

patient_bp_stability.to_csv(
    BP_SUMMARY_DIR
    / "patient_bp_attribution_stability.tsv",
    sep="\t",
    index=False
)

harmonization_summary.to_csv(
    BP_SUMMARY_DIR
    / "gene_harmonization_summary.tsv",
    sep="\t",
    index=False
)


# ============================================================
# 17. MANIFEST
# ============================================================

bp_manifest = {
    "project":
        "AIDO-BBA BRCA 1.0",

    "analysis":
        "GO_BP_structured_SHAP_reconstruction",

    "datetime":
        datetime.now().isoformat(),

    "run_directory":
        str(RUN_DIR),

    "gmt_file":
        str(GO_BP_GMT),

    "hgnc_file":
        str(HGNC_COMPLETE_SET),

    "ncbi_gene_info_file":
        str(NCBI_HUMAN_GENE_INFO),

    "minimum_bp_genes":
        MIN_BP_GENES,

    "n_eligible_bp":
        int(
            len(bp_names)
        ),

    "n_shap_folds":
        int(
            len(shap_matrix_files)
        ),

    "top_bp_per_patient":
        TOP_BP_PER_PATIENT,

    "overlap_handling":
        (
            "Each gene SHAP value is divided equally "
            "among all eligible GO BP terms containing "
            "the harmonized gene."
        ),

    "exact_reconstruction_formula":
        (
            "expected_value + sum(BP_SHAP) "
            "+ unmapped_gene_SHAP_residual"
        )
}

with open(
    STRUCTURED_DIR
    / "bp_reconstruction_manifest.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        bp_manifest,
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
print("GO-BP STRUCTURED RECONSTRUCTION COMPLETED")
print("=" * 78)

print(
    "Eligible GO BP terms:",
    len(bp_names)
)

print(
    "Patient-repeat completeness rows:",
    len(patient_completeness_all)
)

print(
    "Patient summary rows:",
    len(patient_completeness_summary)
)

print(
    "Patient top-BP rows:",
    len(patient_top_bp_all)
)

print(
    "Patient-BP stability rows:",
    len(patient_bp_stability)
)

print(
    "Duration:",
    round(
        total_minutes,
        2
    ),
    "minutes"
)

print("\nFold-level reconstruction audit:")

display(
    bp_reconstruction_fold_audit[
        [
            "repeat_id",
            "fold_id",
            "n_mapped_selected_genes",
            "selected_gene_coverage_fraction",
            "mean_attribution_mass_coverage",
            "mean_absolute_unmapped_residual",
            "mean_absolute_exact_reconstruction_error",
            "maximum_absolute_exact_reconstruction_error",
            "reconstruction_correlation"
        ]
    ].round(8)
)

print("\nOverall completeness summary:")

overall_completeness_summary = pd.DataFrame([
    {
        "metric":
            "mean_selected_gene_coverage",

        "value":
            patient_completeness_all[
                "selected_gene_coverage_fraction"
            ].mean()
    },
    {
        "metric":
            "mean_attribution_mass_coverage",

        "value":
            patient_completeness_all[
                "attribution_mass_coverage"
            ].mean()
    },
    {
        "metric":
            "median_attribution_mass_coverage",

        "value":
            patient_completeness_all[
                "attribution_mass_coverage"
            ].median()
    },
    {
        "metric":
            "mean_absolute_unmapped_signed_residual",

        "value":
            np.mean(
                np.abs(
                    patient_completeness_all[
                        "unmapped_signed_residual"
                    ]
                )
            )
    },
    {
        "metric":
            "mean_exact_reconstruction_error",

        "value":
            np.mean(
                np.abs(
                    patient_completeness_all[
                        "exact_reconstruction_error"
                    ]
                )
            )
    }
])

for top_k in TOP_K_VALUES:

    overall_completeness_summary.loc[
        len(
            overall_completeness_summary
        )
    ] = {
        "metric":
            f"mean_top{top_k}_bp_mass_fraction",

        "value":
            patient_completeness_all[
                f"top{top_k}_bp_mass_fraction"
            ].mean()
    }

display(
    overall_completeness_summary
)

print("\nTop 30 stable GO BP explanations:")

display(
    global_bp_stability.head(30)
)

print("\nOutput directory:")
print(STRUCTURED_DIR)