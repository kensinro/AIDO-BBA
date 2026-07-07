
# %% [CELL 1] ============================================================
# AIDO-BBA KIRC PORTABILITY AUDIT — FULL INTEGRATED SCRIPT V3 STRICT ENDPOINT
#
# Purpose
# -------
# Cross-cancer portability stress test using TCGA-KIRC:
# 1. Build Early (I/II) vs Advanced (III/IV) stage endpoint.
# 2. Match primary-tumour GE with patient-level stage.
# 3. Run repeated CV with ElasticNet logistic and ExtraTrees.
# 4. Audit cross-model disagreement and ambiguous patients.
# 5. Audit GO-BP representation completeness.
# 6. Identify KIRC representation-gap genes.
# 7. Project BRCA stable cores into KIRC.
# 8. Generate manuscript-ready figures and tables.
#
# This is a portability stress test, not a full second AIDO-BBA study.
# ============================================================

import os
import re
import time
import math
import json
import glob
import warnings
from pathlib import Path
from aido_bba.config import (
    data_root, brca_output_root, brca_run_dir, go_bp_gmt,
    hgnc_file, ncbi_gene_info, metabric_dir, gse96058_dir,
    kirc_dir, kirc_output_root,
)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    confusion_matrix
)
from sklearn.feature_selection import f_classif

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# -----------------------------
# PATHS
# -----------------------------
KIRC_DIR = str(kirc_dir())
BRCA_RUN_DIR = str(brca_run_dir())
GO_BP_GMT = str(go_bp_gmt())

OUTPUT_ROOT = str(kirc_output_root())

KIRC_GE_FILE = os.path.join(KIRC_DIR, "GE.tsv")
KIRC_PHENO_FILE = os.path.join(KIRC_DIR, "Phenotype.tsv")
KIRC_CLINICAL_MATRIX_FILE = os.path.join(
    KIRC_DIR,
    "TCGA.KIRC.sampleMap_KIRC_clinicalMatrix"
)

# -----------------------------
# MODEL SETTINGS
# -----------------------------
RANDOM_SEED = 20260701
N_REPEATS = 5
N_SPLITS = 5
N_SELECTED_GENES = 1500
TOP_IMPORTANCE_GENES_PER_FOLD = 100
TOP_GAP_GENES_TO_PLOT = 20
AMBIGUITY_MARGIN = 0.05

# -----------------------------
# OUTPUT
# -----------------------------
RUN_STAMP = time.strftime("%Y%m%d_%H%M%S")
RUN_DIR = os.path.join(
    OUTPUT_ROOT,
    f"RUN_KIRC_PORTABILITY_{RUN_STAMP}"
)
TABLE_DIR = os.path.join(RUN_DIR, "tables")
FIG_DIR = os.path.join(RUN_DIR, "figures")
LOG_DIR = os.path.join(RUN_DIR, "logs")

for directory in [
    RUN_DIR,
    TABLE_DIR,
    FIG_DIR,
    LOG_DIR
]:
    os.makedirs(directory, exist_ok=True)

print("=" * 88)
print("AIDO-BBA KIRC PORTABILITY AUDIT")
print("=" * 88)
print("KIRC directory:", KIRC_DIR)
print("BRCA reference run:", BRCA_RUN_DIR)
print("GO-BP GMT:", GO_BP_GMT)
print("Output:", RUN_DIR)


# %% [CELL 2] ============================================================
# UTILITIES
# ============================================================

def normalize_sample_id(value):

    if pd.isna(value):
        return np.nan

    return str(value).strip()


def patient_id_from_sample(sample_id):

    if pd.isna(sample_id):
        return np.nan

    parts = str(sample_id).split("-")

    if len(parts) >= 3:
        return "-".join(parts[:3])

    return str(sample_id)


def sample_type_code_from_sample(sample_id):

    if pd.isna(sample_id):
        return np.nan

    parts = str(sample_id).split("-")

    if len(parts) >= 4:
        return parts[3][:2]

    return np.nan


def is_primary_tumour(sample_id):

    return (
        sample_type_code_from_sample(sample_id)
        ==
        "01"
    )


def detect_text_encoding(path):

    path = Path(path)

    with open(path, "rb") as handle:
        prefix = handle.read(4)

    if prefix.startswith(b"\xff\xfe"):
        return "utf-16-le"

    if prefix.startswith(b"\xfe\xff"):
        return "utf-16-be"

    if prefix.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"

    return None


def safe_read_table(path):

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    detected_encoding = detect_text_encoding(path)

    candidate_encodings = []

    if detected_encoding is not None:
        candidate_encodings.append(detected_encoding)

    candidate_encodings.extend([
        "utf-8-sig",
        "utf-8",
        "utf-16",
        "utf-16-le",
        "utf-16-be",
        "cp1252",
        "latin1",
    ])

    # Preserve order while removing duplicates.
    candidate_encodings = list(
        dict.fromkeys(candidate_encodings)
    )

    read_errors = []

    for encoding in candidate_encodings:

        try:
            dataframe = pd.read_csv(
                path,
                sep="\t",
                encoding=encoding,
                encoding_errors="strict",
                low_memory=False,
            )

            # Reject clearly mis-decoded single-column tables.
            if dataframe.shape[1] <= 1:
                read_errors.append(
                    f"{encoding}: parsed only "
                    f"{dataframe.shape[1]} column"
                )
                continue

            print(
                f"Loaded {Path(path).name} "
                f"with encoding={encoding}, "
                f"shape={dataframe.shape}"
            )

            return dataframe

        except Exception as error:
            read_errors.append(
                f"{encoding}: {type(error).__name__}: {error}"
            )

    raise ValueError(
        "Unable to read table with the tested encodings:\n"
        + "\n".join(read_errors)
    )


def pick_sample_id_column(dataframe):

    candidates = [
        "sampleID",
        "sample",
        "SampleID",
        "Sample",
        "Tumor_Sample_Barcode",
        "bcr_patient_barcode",
        "_SAMPLE",
        "Hybridization REF",
        "Hybridization_REF",
        "sample_id",
        "patient_id"
    ]

    for column in candidates:

        if column in dataframe.columns:
            return column

    for column in dataframe.columns:

        values = dataframe[column].astype(str).head(30)

        fraction_tcga = (
            values.str.contains(
                "TCGA-",
                regex=False,
                na=False
            ).mean()
        )

        if fraction_tcga >= 0.30:
            return column

    return dataframe.columns[0]


def normalize_stage_text(value):

    if pd.isna(value):
        return np.nan

    text = str(value).strip().upper()

    if text in {
        "",
        "NA",
        "NAN",
        "NONE",
        "NULL",
        "NOT REPORTED",
        "UNKNOWN"
    }:
        return np.nan

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

    ordered_patterns = [
        "IIIC",
        "IIIB",
        "IIIA",
        "III",
        "IIC",
        "IIB",
        "IIA",
        "II",
        "IVC",
        "IVB",
        "IVA",
        "IV",
        "IB",
        "IA",
        "IS",
        "I"
    ]

    # Handle stage IV before stage I ambiguity.
    for pattern in [
        "IVC",
        "IVB",
        "IVA",
        "IV",
        "IIIC",
        "IIIB",
        "IIIA",
        "III",
        "IIC",
        "IIB",
        "IIA",
        "II",
        "IB",
        "IA",
        "IS",
        "I"
    ]:
        if re.search(
            rf"\b{pattern}\b",
            text
        ):
            return pattern

    numeric_tokens = re.findall(
        r"(?<!\d)[1-4](?!\d)",
        text
    )

    if numeric_tokens:

        numeric_stage = int(
            numeric_tokens[0]
        )

        return {
            1: "I",
            2: "II",
            3: "III",
            4: "IV"
        }.get(
            numeric_stage,
            np.nan
        )

    return np.nan


def stage_group_from_raw(stage_raw):

    if pd.isna(stage_raw):
        return np.nan

    stage = str(stage_raw).upper().strip()

    early = {
        "I",
        "IA",
        "IB",
        "IS",
        "II",
        "IIA",
        "IIB",
        "IIC"
    }

    advanced = {
        "III",
        "IIIA",
        "IIIB",
        "IIIC",
        "IV",
        "IVA",
        "IVB",
        "IVC"
    }

    if stage in early:
        return "Early"

    if stage in advanced:
        return "Advanced"

    return np.nan


def stage_column_priority(column_name):

    name = str(column_name).strip().lower()

    # Explicit exclusions: these may contain roman numerals or numbers
    # but are not clinical stage fields.
    excluded_terms = [
        "date",
        "time",
        "rppa",
        "protein",
        "pancan",
        "plate",
        "batch",
        "form_completion",
        "year",
        "month",
        "day",
    ]

    if any(
        term in name
        for term in excluded_terms
    ):
        return None

    # Only columns explicitly denoting clinical/pathologic/AJCC stage
    # are permitted. No fallback scan over arbitrary text columns.
    exact_priority = {
        "pathologic_stage": 0,
        "pathological_stage": 1,
        "ajcc_pathologic_tumor_stage": 2,
        "ajcc_pathologic_stage": 3,
        "tumor_stage": 4,
        "clinical_stage": 5,
        "ajcc_clinical_stage": 6,
        "stage": 7,
    }

    # Strip merge suffixes generated by pandas.
    base_name = re.sub(
        r"_(pheno|clinical)$",
        "",
        name,
    )

    if base_name in exact_priority:
        return exact_priority[
            base_name
        ]

    # Controlled keyword fallback is limited to stage-labelled fields.
    if "stage" not in base_name:
        return None

    if (
        "pathologic" in base_name
        or
        "pathological" in base_name
    ):
        return 20

    if "ajcc" in base_name:
        return 30

    if "clinical" in base_name:
        return 40

    if "tumor" in base_name:
        return 50

    return 60


def get_strict_stage_columns(dataframe):

    ranked_columns = []

    for column in dataframe.columns:

        priority = stage_column_priority(
            column
        )

        if priority is not None:
            ranked_columns.append(
                (
                    priority,
                    str(column),
                )
            )

    ranked_columns = sorted(
        ranked_columns,
        key=lambda item: (
            item[0],
            item[1],
        ),
    )

    return [
        column
        for _, column
        in ranked_columns
    ]


def build_stage_from_dataframe(dataframe):

    stage_candidate_columns = (
        get_strict_stage_columns(
            dataframe
        )
    )

    if len(stage_candidate_columns) == 0:
        raise KeyError(
            "No explicitly stage-labelled clinical column was found. "
            "The strict endpoint parser does not scan arbitrary fields."
        )

    print(
        "Strict stage columns, in priority order:"
    )

    for column in stage_candidate_columns:
        print(
            " -",
            column,
        )

    stage_raw_values = []
    stage_source_values = []
    stage_source_original_values = []

    for _, row in dataframe.iterrows():

        found_stage = np.nan
        found_source = np.nan
        found_original_value = np.nan

        for column in stage_candidate_columns:

            original_value = row[
                column
            ]

            normalized = normalize_stage_text(
                original_value
            )

            if pd.notna(normalized):

                found_stage = normalized
                found_source = column
                found_original_value = original_value
                break

        stage_raw_values.append(
            found_stage
        )

        stage_source_values.append(
            found_source
        )

        stage_source_original_values.append(
            found_original_value
        )

    output = dataframe.copy()

    output[
        "stage_raw"
    ] = stage_raw_values

    output[
        "stage_group"
    ] = output[
        "stage_raw"
    ].map(
        stage_group_from_raw
    )

    output[
        "stage_source_column"
    ] = stage_source_values

    output[
        "stage_source_original_value"
    ] = stage_source_original_values

    output[
        "stage_label"
    ] = output[
        "stage_group"
    ].map(
        {
            "Early": 0,
            "Advanced": 1,
        }
    )

    # Hard contamination guard: every resolved endpoint must originate
    # from a permitted stage-labelled field.
    invalid_sources = (
        output.loc[
            output[
                "stage_group"
            ].notna(),
            "stage_source_column"
        ]
        .dropna()
        .astype(str)
        .map(
            stage_column_priority
        )
        .isna()
        .sum()
    )

    if invalid_sources != 0:
        raise RuntimeError(
            "Strict stage-source validation failed: "
            f"{invalid_sources} resolved rows came from non-stage fields."
        )

    return output


def read_gmt_gene_universe(
    gmt_path,
    min_genes=10
):

    eligible_sets = 0
    gene_universe = set()

    with open(
        gmt_path,
        "r",
        encoding="utf-8"
    ) as handle:

        for line in handle:

            parts = line.rstrip(
                "\n"
            ).split(
                "\t"
            )

            if len(parts) < 3:
                continue

            genes = [
                gene.strip()
                for gene in parts[2:]
                if str(gene).strip()
            ]

            genes = list(
                dict.fromkeys(
                    genes
                )
            )

            if len(genes) >= min_genes:

                eligible_sets += 1
                gene_universe.update(
                    genes
                )

    return gene_universe, eligible_sets


def cliffs_delta(
    group_a,
    group_b
):

    group_a = np.asarray(
        group_a,
        dtype=float
    )

    group_b = np.asarray(
        group_b,
        dtype=float
    )

    if (
        len(group_a) == 0
        or
        len(group_b) == 0
    ):
        return np.nan

    u_statistic, _ = stats.mannwhitneyu(
        group_a,
        group_b,
        alternative="two-sided"
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


def cliffs_delta_magnitude(delta):

    if pd.isna(delta):
        return np.nan

    absolute_delta = abs(delta)

    if absolute_delta < 0.147:
        return "negligible"

    if absolute_delta < 0.33:
        return "small"

    if absolute_delta < 0.474:
        return "medium"

    return "large"


def best_balanced_accuracy_threshold(
    y_true,
    probability,
    thresholds=None
):

    if thresholds is None:

        thresholds = np.arange(
            0.05,
            0.951,
            0.01
        )

    rows = []

    for threshold in thresholds:

        predicted = (
            probability
            >=
            threshold
        ).astype(int)

        balanced_accuracy = (
            balanced_accuracy_score(
                y_true,
                predicted
            )
        )

        sensitivity = recall_score(
            y_true,
            predicted,
            pos_label=1,
            zero_division=0
        )

        specificity = recall_score(
            y_true,
            predicted,
            pos_label=0,
            zero_division=0
        )

        rows.append({
            "threshold":
                threshold,
            "balanced_accuracy":
                balanced_accuracy,
            "sensitivity":
                sensitivity,
            "specificity":
                specificity
        })

    scan = pd.DataFrame(
        rows
    ).sort_values(
        [
            "balanced_accuracy",
            "sensitivity",
            "specificity",
            "threshold"
        ],
        ascending=[
            False,
            False,
            False,
            True
        ]
    ).reset_index(
        drop=True
    )

    return (
        scan.iloc[0].to_dict(),
        scan
    )


def find_brca_core_file(
    brca_run_dir
):

    preferred_candidates = [
        os.path.join(
            brca_run_dir,
            "13_gap_module_cores",
            "summaries",
            "stable_core_gene_manifest.tsv"
        ),
        os.path.join(
            brca_run_dir,
            "12_gap_gene_modules",
            "summaries",
            "stable_core_gene_manifest.tsv"
        )
    ]

    for path in preferred_candidates:

        if os.path.exists(path):
            return path

    candidate_files = []

    for extension in [
        "*.tsv",
        "*.csv"
    ]:

        candidate_files.extend(
            glob.glob(
                os.path.join(
                    brca_run_dir,
                    "**",
                    extension
                ),
                recursive=True
            )
        )

    for path in candidate_files:

        try:

            if path.lower().endswith(
                ".csv"
            ):

                dataframe = pd.read_csv(
                    path,
                    low_memory=False
                )

            else:

                dataframe = pd.read_csv(
                    path,
                    sep="\t",
                    low_memory=False
                )

        except Exception:
            continue

        columns = set(
            dataframe.columns
        )

        valid_gene_column = (
            "gene_id" in columns
            or
            "raw_gene_id" in columns
        )

        valid_core_column = (
            "core_module_name" in columns
        )

        if (
            valid_gene_column
            and
            valid_core_column
        ):
            return path

    return None


def load_brca_core_reference(
    path
):

    if path is None:
        return None

    if path.lower().endswith(
        ".csv"
    ):

        dataframe = pd.read_csv(
            path,
            low_memory=False
        )

    else:

        dataframe = pd.read_csv(
            path,
            sep="\t",
            low_memory=False
        )

    if "gene_id" in dataframe.columns:

        raw_gene_column = "gene_id"

    elif "raw_gene_id" in dataframe.columns:

        raw_gene_column = "raw_gene_id"

    else:

        raise KeyError(
            "No gene_id/raw_gene_id column "
            "found in BRCA core file."
        )

    if "harmonized_gene_id" not in dataframe.columns:

        dataframe[
            "harmonized_gene_id"
        ] = dataframe[
            raw_gene_column
        ]

    output = dataframe[
        [
            raw_gene_column,
            "harmonized_gene_id",
            "core_module_name"
        ]
    ].copy()

    output.columns = [
        "raw_gene_id",
        "harmonized_gene_id",
        "core_module_name"
    ]

    for column in [
        "raw_gene_id",
        "harmonized_gene_id",
        "core_module_name"
    ]:

        output[column] = (
            output[column]
            .astype(str)
        )

    return (
        output.drop_duplicates()
    )


def save_df(
    dataframe,
    filename,
    index=False
):

    output_path = os.path.join(
        TABLE_DIR,
        filename
    )

    if filename.lower().endswith(
        ".csv"
    ):

        dataframe.to_csv(
            output_path,
            index=index
        )

    else:

        dataframe.to_csv(
            output_path,
            sep="\t",
            index=index
        )

    return output_path


def save_fig(
    filename,
    dpi=200
):

    output_path = os.path.join(
        FIG_DIR,
        filename
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight"
    )

    plt.close()

    return output_path


# %% [CELL 3] ============================================================
# LOAD PHENOTYPE AND CLINICAL DATA
# ============================================================

print("=" * 88)
print("LOADING KIRC PHENOTYPE AND CLINICAL DATA")
print("=" * 88)

pheno_df = safe_read_table(
    KIRC_PHENO_FILE
)

clinical_df = safe_read_table(
    KIRC_CLINICAL_MATRIX_FILE
)

print(
    "Phenotype shape:",
    pheno_df.shape
)

print(
    "Clinical matrix shape:",
    clinical_df.shape
)

pheno_sample_column = pick_sample_id_column(
    pheno_df
)

clinical_sample_column = pick_sample_id_column(
    clinical_df
)

pheno_df = pheno_df.copy()
clinical_df = clinical_df.copy()

pheno_df["sample_id"] = (
    pheno_df[
        pheno_sample_column
    ].map(
        normalize_sample_id
    )
)

clinical_df["sample_id"] = (
    clinical_df[
        clinical_sample_column
    ].map(
        normalize_sample_id
    )
)

pheno_df["patient_id"] = (
    pheno_df[
        "sample_id"
    ].map(
        patient_id_from_sample
    )
)

clinical_df["patient_id"] = (
    clinical_df[
        "sample_id"
    ].map(
        patient_id_from_sample
    )
)

clinical_columns_to_add = [
    column
    for column in clinical_df.columns
    if column
    not in {
        "sample_id",
        "patient_id"
    }
]

merged_metadata = (
    pheno_df.merge(
        clinical_df[
            [
                "sample_id",
                "patient_id",
                *clinical_columns_to_add
            ]
        ],
        on=[
            "sample_id",
            "patient_id"
        ],
        how="outer",
        suffixes=(
            "_pheno",
            "_clinical"
        )
    )
)

stage_df = build_stage_from_dataframe(
    merged_metadata
)

print("\nStage counts:")

display(
    stage_df[
        "stage_group"
    ]
    .value_counts(
        dropna=False
    )
    .rename_axis(
        "stage_group"
    )
    .reset_index(
        name="n"
    )
)

print("\nDetected stage-source columns:")

display(
    stage_df[
        "stage_source_column"
    ]
    .value_counts(
        dropna=False
    )
    .head(20)
    .rename_axis(
        "stage_source_column"
    )
    .reset_index(
        name="n"
    )
)


stage_endpoint_audit = (
    stage_df[
        [
            "sample_id",
            "patient_id",
            "stage_raw",
            "stage_group",
            "stage_label",
            "stage_source_column",
            "stage_source_original_value",
        ]
    ]
    .copy()
)

stage_source_summary = (
    stage_endpoint_audit
    .groupby(
        [
            "stage_source_column",
            "stage_raw",
            "stage_group",
        ],
        dropna=False,
        as_index=False,
    )
    .size()
    .rename(
        columns={
            "size": "n_rows",
        }
    )
)

resolved_nonstage_sources = (
    stage_endpoint_audit.loc[
        stage_endpoint_audit[
            "stage_group"
        ].notna()
    ]
    .assign(
        source_is_permitted=lambda frame:
            frame[
                "stage_source_column"
            ].map(
                stage_column_priority
            ).notna()
    )
    .query(
        "source_is_permitted == False"
    )
)

if len(
    resolved_nonstage_sources
) > 0:
    raise RuntimeError(
        "Endpoint contamination detected: resolved stages arose "
        "from non-permitted columns."
    )

save_df(
    stage_endpoint_audit,
    "KIRC_strict_stage_endpoint_audit.tsv",
    index=False,
)

save_df(
    stage_source_summary,
    "KIRC_strict_stage_source_summary.tsv",
    index=False,
)

print(
    "\nStrict endpoint audit passed: "
    "all resolved stages originated from permitted stage fields."
)

stage_df[
    "sample_type_code"
] = stage_df[
    "sample_id"
].map(
    sample_type_code_from_sample
)

stage_df[
    "is_primary_tumour"
] = stage_df[
    "sample_id"
].map(
    is_primary_tumour
)

stage_primary = (
    stage_df.loc[
        stage_df[
            "is_primary_tumour"
        ]
        &
        stage_df[
            "stage_group"
        ].isin([
            "Early",
            "Advanced"
        ])
    ]
    .copy()
)

patient_group_counts = (
    stage_primary
    .groupby(
        "patient_id"
    )[
        "stage_group"
    ]
    .nunique()
    .rename(
        "n_unique_stage_groups"
    )
    .reset_index()
)

stage_primary = (
    stage_primary.merge(
        patient_group_counts,
        on="patient_id",
        how="left"
    )
)

conflicting_patients = set(
    stage_primary.loc[
        stage_primary[
            "n_unique_stage_groups"
        ]
        >
        1,
        "patient_id"
    ].unique()
)

stage_primary_clean = (
    stage_primary.loc[
        ~stage_primary[
            "patient_id"
        ].isin(
            conflicting_patients
        )
    ]
    .copy()
)

stage_patient = (
    stage_primary_clean
    .sort_values(
        [
            "patient_id",
            "sample_id"
        ]
    )
    .groupby(
        "patient_id",
        as_index=False
    )
    .first()
)

stage_patient[
    "stage_label"
] = stage_patient[
    "stage_group"
].map(
    {
        "Early": 0,
        "Advanced": 1
    }
)

print(
    "\nConflicting stage patients excluded:",
    len(
        conflicting_patients
    )
)

print(
    "Final patient-level stage cohort:",
    stage_patient.shape
)

display(
    stage_patient[
        "stage_group"
    ]
    .value_counts()
    .rename_axis(
        "stage_group"
    )
    .reset_index(
        name="n"
    )
)

unresolved_stage_rows = (
    stage_df.loc[
        stage_df[
            "stage_group"
        ].isna(),
        [
            "sample_id",
            "patient_id",
            "stage_raw",
            "stage_source_column",
            "stage_source_original_value",
        ]
    ]
    .copy()
)

save_df(
    unresolved_stage_rows,
    "KIRC_unresolved_stage_rows.tsv",
    index=False,
)

save_df(
    stage_patient,
    "KIRC_stage_patient_table.tsv",
    index=False
)


# %% [CELL 4] ============================================================
# LOAD KIRC GENE EXPRESSION
# ============================================================

print("=" * 88)
print("LOADING KIRC GENE EXPRESSION")
print("=" * 88)

ge_raw = pd.read_csv(
    KIRC_GE_FILE,
    sep="\t",
    low_memory=False
)

print(
    "Raw GE shape:",
    ge_raw.shape
)

gene_id_column = ge_raw.columns[0]

print(
    "Gene identifier column:",
    gene_id_column
)

expression = (
    ge_raw
    .set_index(
        gene_id_column
    )
    .T
)

expression.index.name = (
    "sample_id"
)

expression.columns = (
    expression.columns.astype(str)
)

expression.index = expression.index.map(
    normalize_sample_id
)

expression = expression.apply(
    pd.to_numeric,
    errors="coerce"
)

# Aggregate duplicate gene identifiers.
expression = (
    expression.T
    .groupby(
        level=0
    )
    .mean()
    .T
)

print(
    "Expression matrix after transpose:",
    expression.shape
)

expression_metadata = pd.DataFrame({
    "sample_id":
        expression.index.astype(str)
})

expression_metadata[
    "patient_id"
] = expression_metadata[
    "sample_id"
].map(
    patient_id_from_sample
)

expression_metadata[
    "sample_type_code"
] = expression_metadata[
    "sample_id"
].map(
    sample_type_code_from_sample
)

expression_metadata[
    "is_primary_tumour"
] = expression_metadata[
    "sample_id"
].map(
    is_primary_tumour
)

expression_metadata[
    "expression_row"
] = np.arange(
    len(
        expression_metadata
    )
)

primary_metadata = (
    expression_metadata.loc[
        expression_metadata[
            "is_primary_tumour"
        ]
    ]
    .copy()
)

primary_metadata = (
    primary_metadata
    .sort_values(
        [
            "patient_id",
            "sample_id"
        ]
    )
    .groupby(
        "patient_id",
        as_index=False
    )
    .first()
)

primary_expression = (
    expression.iloc[
        primary_metadata[
            "expression_row"
        ].to_numpy(
            dtype=int
        )
    ]
    .copy()
)

primary_expression.index = (
    primary_metadata[
        "patient_id"
    ].astype(str)
)

print(
    "Final unique primary-tumour GE patients:",
    primary_expression.shape[0]
)

print(
    "Primary GE matrix shape:",
    primary_expression.shape
)

save_df(
    primary_metadata,
    "KIRC_primary_expression_metadata.tsv",
    index=False
)


# %% [CELL 5] ============================================================
# MATCH EXPRESSION WITH STAGE ENDPOINT
# ============================================================

print("=" * 88)
print("MATCHING GE WITH STAGE ENDPOINT")
print("=" * 88)

matched_patient_ids = sorted(
    set(
        stage_patient[
            "patient_id"
        ].astype(str)
    )
    &
    set(
        primary_expression.index.astype(str)
    )
)

if len(matched_patient_ids) < 100:

    raise ValueError(
        "Fewer than 100 KIRC patients matched "
        "between stage and expression data."
    )

matched = (
    stage_patient
    .set_index(
        "patient_id"
    )
    .loc[
        matched_patient_ids
    ]
    .reset_index()
)

X_df = (
    primary_expression.loc[
        matched_patient_ids
    ]
    .copy()
)

y = matched[
    "stage_label"
].astype(int).to_numpy()

X_df = X_df.replace(
    [
        np.inf,
        -np.inf
    ],
    np.nan
)

X_df = X_df.loc[
    :,
    ~X_df.isna().all(
        axis=0
    )
]

if X_df.isna().sum().sum() > 0:

    X_df = X_df.fillna(
        X_df.median(
            axis=0
        )
    )

variances = X_df.var(
    axis=0
)

X_df = X_df.loc[
    :,
    variances > 0
]

print(
    "Matched cohort:",
    matched.shape
)

display(
    matched[
        "stage_group"
    ]
    .value_counts()
    .rename_axis(
        "stage_group"
    )
    .reset_index(
        name="n"
    )
)

print(
    "Final modeling matrix:",
    X_df.shape
)

cohort_summary = pd.DataFrame([
    {
        "metric":
            "matched_patients",
        "value":
            X_df.shape[0]
    },
    {
        "metric":
            "final_genes",
        "value":
            X_df.shape[1]
    },
    {
        "metric":
            "early_patients",
        "value":
            int(
                np.sum(
                    y == 0
                )
            )
    },
    {
        "metric":
            "advanced_patients",
        "value":
            int(
                np.sum(
                    y == 1
                )
            )
    }
])

display(
    cohort_summary
)

save_df(
    matched,
    "KIRC_matched_analysis_cohort.tsv",
    index=False
)

save_df(
    cohort_summary,
    "KIRC_cohort_summary.tsv",
    index=False
)


# %% [CELL 6] ============================================================
# REPEATED CV: ELASTICNET + EXTRATREES
# ============================================================

print("=" * 88)
print("REPEATED CV MODELING")
print("=" * 88)

X = X_df.to_numpy(
    dtype=np.float32
)

feature_names = np.asarray(
    X_df.columns.astype(str)
)

patient_ids = matched[
    "patient_id"
].astype(str).to_numpy()

true_groups = matched[
    "stage_group"
].astype(str).to_numpy()

oof_rows = []
selected_gene_rows = []
importance_rows = []

fold_counter = 0
start_time = time.time()

for repeat_id in range(
    1,
    N_REPEATS + 1
):

    splitter = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=(
            RANDOM_SEED
            +
            repeat_id
        )
    )

    for fold_id, (
        train_index,
        test_index
    ) in enumerate(
        splitter.split(
            X,
            y
        ),
        start=1
    ):

        fold_counter += 1

        print(
            f"[{fold_counter:02d}/"
            f"{N_REPEATS * N_SPLITS}] "
            f"repeat {repeat_id}, "
            f"fold {fold_id}"
        )

        X_train = X[
            train_index
        ]

        X_test = X[
            test_index
        ]

        y_train = y[
            train_index
        ]

        y_test = y[
            test_index
        ]

        f_statistics, p_values = f_classif(
            X_train,
            y_train
        )

        f_statistics = np.nan_to_num(
            f_statistics,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        n_select = min(
            N_SELECTED_GENES,
            X_train.shape[1]
        )

        selected_indices = np.argsort(
            f_statistics
        )[
            ::-1
        ][
            :n_select
        ]

        selected_features = (
            feature_names[
                selected_indices
            ]
        )

        selected_gene_rows.append(
            pd.DataFrame({
                "repeat_id":
                    repeat_id,
                "fold_id":
                    fold_id,
                "gene_id":
                    selected_features,
                "f_stat":
                    f_statistics[
                        selected_indices
                    ],
                "rank_within_fold":
                    np.arange(
                        1,
                        len(
                            selected_indices
                        )
                        +
                        1
                    )
            })
        )

        X_train_selected = (
            X_train[
                :,
                selected_indices
            ]
        )

        X_test_selected = (
            X_test[
                :,
                selected_indices
            ]
        )

        elasticnet_model = Pipeline([
            (
                "scaler",
                StandardScaler()
            ),
            (
                "classifier",
                LogisticRegression(
                    penalty="elasticnet",
                    solver="saga",
                    l1_ratio=0.5,
                    C=1.0,
                    max_iter=5000,
                    class_weight="balanced",
                    random_state=(
                        RANDOM_SEED
                        +
                        repeat_id
                        *
                        100
                        +
                        fold_id
                    ),
                    n_jobs=-1
                )
            )
        ])

        elasticnet_model.fit(
            X_train_selected,
            y_train
        )

        probability_elasticnet = (
            elasticnet_model.predict_proba(
                X_test_selected
            )[:, 1]
        )

        extratrees_model = ExtraTreesClassifier(
            n_estimators=500,
            criterion="gini",
            max_features="sqrt",
            min_samples_leaf=1,
            bootstrap=False,
            class_weight="balanced",
            random_state=(
                RANDOM_SEED
                +
                repeat_id
                *
                1000
                +
                fold_id
            ),
            n_jobs=-1
        )

        extratrees_model.fit(
            X_train_selected,
            y_train
        )

        probability_extratrees = (
            extratrees_model.predict_proba(
                X_test_selected
            )[:, 1]
        )

        importances = (
            extratrees_model.feature_importances_
        )

        importance_order = np.argsort(
            importances
        )[
            ::-1
        ][
            :TOP_IMPORTANCE_GENES_PER_FOLD
        ]

        importance_rows.append(
            pd.DataFrame({
                "repeat_id":
                    repeat_id,
                "fold_id":
                    fold_id,
                "gene_id":
                    selected_features[
                        importance_order
                    ],
                "feature_importance":
                    importances[
                        importance_order
                    ],
                "importance_rank":
                    np.arange(
                        1,
                        len(
                            importance_order
                        )
                        +
                        1
                    )
            })
        )

        oof_rows.append(
            pd.DataFrame({
                "repeat_id":
                    repeat_id,
                "fold_id":
                    fold_id,
                "patient_id":
                    patient_ids[
                        test_index
                    ],
                "true_label":
                    y_test,
                "true_group":
                    true_groups[
                        test_index
                    ],
                "elasticnet_probability_advanced":
                    probability_elasticnet,
                "extratrees_probability_advanced":
                    probability_extratrees
            })
        )

elapsed_minutes = (
    time.time()
    -
    start_time
) / 60.0

print(
    f"\nCompleted repeated CV in "
    f"{elapsed_minutes:.2f} minutes."
)

oof_long = pd.concat(
    oof_rows,
    ignore_index=True
)

selected_genes_long = pd.concat(
    selected_gene_rows,
    ignore_index=True
)

importance_long = pd.concat(
    importance_rows,
    ignore_index=True
)

save_df(
    oof_long,
    "KIRC_oof_predictions_by_repeat_fold.tsv",
    index=False
)

save_df(
    selected_genes_long,
    "KIRC_selected_genes_by_fold.tsv",
    index=False
)

save_df(
    importance_long,
    "KIRC_extratrees_top_importance_genes_by_fold.tsv",
    index=False
)


# %% [CELL 7] ============================================================
# PATIENT-LEVEL OOF AGGREGATION
# ============================================================

print("=" * 88)
print("PATIENT-LEVEL OOF AGGREGATION")
print("=" * 88)

patient_oof = (
    oof_long
    .groupby(
        [
            "patient_id",
            "true_label",
            "true_group"
        ],
        as_index=False
    )
    .agg(
        elasticnet_probability_advanced=(
            "elasticnet_probability_advanced",
            "mean"
        ),
        extratrees_probability_advanced=(
            "extratrees_probability_advanced",
            "mean"
        ),
        n_repeats=(
            "repeat_id",
            "size"
        )
    )
)

performance_rows = []

for model_name, probability_column in [
    (
        "ElasticNet_Logistic",
        "elasticnet_probability_advanced"
    ),
    (
        "ExtraTrees_BlackBox",
        "extratrees_probability_advanced"
    )
]:

    y_true = patient_oof[
        "true_label"
    ].to_numpy(
        dtype=int
    )

    probability = patient_oof[
        probability_column
    ].to_numpy(
        dtype=float
    )

    auc = roc_auc_score(
        y_true,
        probability
    )

    average_precision = (
        average_precision_score(
            y_true,
            probability
        )
    )

    best_threshold, threshold_scan = (
        best_balanced_accuracy_threshold(
            y_true,
            probability
        )
    )

    threshold = best_threshold[
        "threshold"
    ]

    predicted = (
        probability
        >=
        threshold
    ).astype(int)

    balanced_accuracy = (
        balanced_accuracy_score(
            y_true,
            predicted
        )
    )

    accuracy = accuracy_score(
        y_true,
        predicted
    )

    precision = precision_score(
        y_true,
        predicted,
        zero_division=0
    )

    sensitivity = recall_score(
        y_true,
        predicted,
        pos_label=1,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        predicted,
        zero_division=0
    )

    confusion = confusion_matrix(
        y_true,
        predicted
    )

    if confusion.shape == (2, 2):

        tn, fp, fn, tp = confusion.ravel()

    else:

        tn = fp = fn = tp = np.nan

    specificity = (
        tn
        /
        (
            tn
            +
            fp
        )
        if (
            pd.notna(tn)
            and
            (
                tn
                +
                fp
            )
            >
            0
        )
        else np.nan
    )

    performance_rows.append({
        "model_name":
            model_name,
        "auc":
            auc,
        "average_precision":
            average_precision,
        "optimal_threshold_balanced_accuracy":
            threshold,
        "balanced_accuracy":
            balanced_accuracy,
        "accuracy":
            accuracy,
        "precision":
            precision,
        "recall_sensitivity":
            sensitivity,
        "specificity":
            specificity,
        "f1_score":
            f1,
        "tn":
            tn,
        "fp":
            fp,
        "fn":
            fn,
        "tp":
            tp
    })

    save_df(
        threshold_scan,
        f"KIRC_threshold_scan_{model_name}.tsv",
        index=False
    )

performance_df = (
    pd.DataFrame(
        performance_rows
    )
    .sort_values(
        "auc",
        ascending=False
    )
    .reset_index(
        drop=True
    )
)

display(
    performance_df.round(4)
)

save_df(
    performance_df,
    "KIRC_model_performance_summary.tsv",
    index=False
)

threshold_map = dict(
    zip(
        performance_df[
            "model_name"
        ],
        performance_df[
            "optimal_threshold_balanced_accuracy"
        ]
    )
)

patient_oof[
    "elasticnet_threshold"
] = threshold_map[
    "ElasticNet_Logistic"
]

patient_oof[
    "extratrees_threshold"
] = threshold_map[
    "ExtraTrees_BlackBox"
]

patient_oof[
    "elasticnet_pred"
] = (
    patient_oof[
        "elasticnet_probability_advanced"
    ]
    >=
    patient_oof[
        "elasticnet_threshold"
    ]
).astype(int)

patient_oof[
    "extratrees_pred"
] = (
    patient_oof[
        "extratrees_probability_advanced"
    ]
    >=
    patient_oof[
        "extratrees_threshold"
    ]
).astype(int)

save_df(
    patient_oof,
    "KIRC_patient_level_oof_predictions.tsv",
    index=False
)


# %% [CELL 8] ============================================================
# MODEL DISAGREEMENT AUDIT
# ============================================================

print("=" * 88)
print("MODEL DISAGREEMENT AUDIT")
print("=" * 88)

patient_oof[
    "elasticnet_correct"
] = (
    patient_oof[
        "true_label"
    ]
    ==
    patient_oof[
        "elasticnet_pred"
    ]
)

patient_oof[
    "extratrees_correct"
] = (
    patient_oof[
        "true_label"
    ]
    ==
    patient_oof[
        "extratrees_pred"
    ]
)

def cross_model_state(row):

    if (
        row["elasticnet_correct"]
        and
        row["extratrees_correct"]
    ):
        return "both_correct"

    if (
        not row["elasticnet_correct"]
        and
        not row["extratrees_correct"]
    ):
        return "both_wrong"

    if (
        row["elasticnet_correct"]
        and
        not row["extratrees_correct"]
    ):
        return "elasticnet_only_correct"

    return "extratrees_only_correct"


patient_oof[
    "cross_model_state"
] = patient_oof.apply(
    cross_model_state,
    axis=1
)

patient_oof[
    "absolute_probability_difference"
] = np.abs(
    patient_oof[
        "elasticnet_probability_advanced"
    ]
    -
    patient_oof[
        "extratrees_probability_advanced"
    ]
)

patient_oof[
    "elasticnet_near_threshold"
] = (
    np.abs(
        patient_oof[
            "elasticnet_probability_advanced"
        ]
        -
        patient_oof[
            "elasticnet_threshold"
        ]
    )
    <=
    AMBIGUITY_MARGIN
)

patient_oof[
    "extratrees_near_threshold"
] = (
    np.abs(
        patient_oof[
            "extratrees_probability_advanced"
        ]
        -
        patient_oof[
            "extratrees_threshold"
        ]
    )
    <=
    AMBIGUITY_MARGIN
)

def integrated_state(row):

    if (
        row["elasticnet_pred"]
        !=
        row["extratrees_pred"]
    ):
        return "model_dependent"

    if (
        row["elasticnet_near_threshold"]
        and
        row["extratrees_near_threshold"]
    ):
        return "intermediate_ambiguous"

    if (
        row["true_label"] == 0
        and
        row["elasticnet_pred"] == 1
        and
        row["extratrees_pred"] == 1
    ):
        return "clinical_early_molecular_advanced_like"

    if (
        row["true_label"] == 1
        and
        row["elasticnet_pred"] == 0
        and
        row["extratrees_pred"] == 0
    ):
        return "clinical_advanced_molecular_early_like"

    return "clinical_molecular_concordant"


patient_oof[
    "integrated_portability_state"
] = patient_oof.apply(
    integrated_state,
    axis=1
)

state_counts = (
    patient_oof
    .groupby(
        [
            "true_group",
            "integrated_portability_state"
        ],
        as_index=False
    )
    .size()
    .rename(
        columns={
            "size": "n"
        }
    )
)

cross_model_counts = (
    patient_oof
    .groupby(
        [
            "true_group",
            "cross_model_state"
        ],
        as_index=False
    )
    .size()
    .rename(
        columns={
            "size": "n"
        }
    )
)

most_discordant = (
    patient_oof
    .sort_values(
        "absolute_probability_difference",
        ascending=False
    )
    .head(30)
)

display(
    state_counts
)

display(
    cross_model_counts
)

save_df(
    state_counts,
    "KIRC_integrated_portability_state_counts.tsv",
    index=False
)

save_df(
    cross_model_counts,
    "KIRC_cross_model_state_counts.tsv",
    index=False
)

save_df(
    most_discordant,
    "KIRC_most_discordant_patients.tsv",
    index=False
)

save_df(
    patient_oof,
    "KIRC_patient_level_oof_predictions_with_states.tsv",
    index=False
)


# %% [CELL 9] ============================================================
# GO-BP COMPLETENESS AND GAP GENES
# ============================================================

print("=" * 88)
print("GO-BP COMPLETENESS AND REPRESENTATION-GAP AUDIT")
print("=" * 88)

go_bp_gene_universe, n_eligible_sets = (
    read_gmt_gene_universe(
        GO_BP_GMT,
        min_genes=10
    )
)

print(
    "Eligible GO-BP terms:",
    n_eligible_sets
)

print(
    "Genes represented in eligible GO-BP space:",
    len(
        go_bp_gene_universe
    )
)

selected_genes_long = (
    selected_genes_long.copy()
)

selected_genes_long[
    "represented_in_go_bp"
] = selected_genes_long[
    "gene_id"
].isin(
    go_bp_gene_universe
)

fold_completeness = (
    selected_genes_long
    .groupby(
        [
            "repeat_id",
            "fold_id"
        ],
        as_index=False
    )
    .agg(
        n_selected_genes=(
            "gene_id",
            "size"
        ),
        n_represented_genes=(
            "represented_in_go_bp",
            "sum"
        )
    )
)

fold_completeness[
    "representation_fraction"
] = (
    fold_completeness[
        "n_represented_genes"
    ]
    /
    fold_completeness[
        "n_selected_genes"
    ]
)

fold_completeness[
    "gap_fraction"
] = (
    1.0
    -
    fold_completeness[
        "representation_fraction"
    ]
)

completeness_summary = pd.DataFrame([
    {
        "metric":
            "mean_representation_fraction_selected_1500",
        "value":
            fold_completeness[
                "representation_fraction"
            ].mean()
    },
    {
        "metric":
            "sd_representation_fraction_selected_1500",
        "value":
            fold_completeness[
                "representation_fraction"
            ].std(
                ddof=1
            )
    },
    {
        "metric":
            "mean_gap_fraction_selected_1500",
        "value":
            fold_completeness[
                "gap_fraction"
            ].mean()
    },
    {
        "metric":
            "n_eligible_go_bp_terms",
        "value":
            n_eligible_sets
    },
    {
        "metric":
            "n_go_bp_gene_universe",
        "value":
            len(
                go_bp_gene_universe
            )
    }
])

importance_long = (
    importance_long.copy()
)

importance_long[
    "represented_in_go_bp"
] = importance_long[
    "gene_id"
].isin(
    go_bp_gene_universe
)

importance_gap_summary = (
    importance_long
    .groupby(
        "gene_id",
        as_index=False
    )
    .agg(
        n_folds_top100=(
            "gene_id",
            "size"
        ),
        mean_feature_importance=(
            "feature_importance",
            "mean"
        ),
        median_feature_importance=(
            "feature_importance",
            "median"
        ),
        represented_in_go_bp=(
            "represented_in_go_bp",
            "max"
        )
    )
)

importance_gap_summary[
    "fold_selection_frequency"
] = (
    importance_gap_summary[
        "n_folds_top100"
    ]
    /
    (
        N_REPEATS
        *
        N_SPLITS
    )
)

gap_gene_summary = (
    importance_gap_summary.loc[
        ~importance_gap_summary[
            "represented_in_go_bp"
        ]
    ]
    .copy()
    .sort_values(
        [
            "fold_selection_frequency",
            "mean_feature_importance"
        ],
        ascending=[
            False,
            False
        ]
    )
    .reset_index(
        drop=True
    )
)

represented_gene_summary = (
    importance_gap_summary.loc[
        importance_gap_summary[
            "represented_in_go_bp"
        ]
    ]
    .copy()
    .sort_values(
        [
            "fold_selection_frequency",
            "mean_feature_importance"
        ],
        ascending=[
            False,
            False
        ]
    )
    .reset_index(
        drop=True
    )
)

display(
    completeness_summary.round(4)
)

display(
    gap_gene_summary.head(40)
)

save_df(
    fold_completeness,
    "KIRC_fold_level_go_bp_completeness.tsv",
    index=False
)

save_df(
    completeness_summary,
    "KIRC_go_bp_completeness_summary.tsv",
    index=False
)

save_df(
    importance_gap_summary,
    "KIRC_top100_importance_gene_summary_all.tsv",
    index=False
)

save_df(
    gap_gene_summary,
    "KIRC_representation_gap_genes_summary.tsv",
    index=False
)

save_df(
    represented_gene_summary.head(100),
    "KIRC_top_represented_importance_genes.tsv",
    index=False
)


# %% [CELL 10] ============================================================
# LOAD BRCA STABLE CORES
# ============================================================

print("=" * 88)
print("LOADING BRCA STABLE-CORE REFERENCE")
print("=" * 88)

brca_core_file = find_brca_core_file(
    BRCA_RUN_DIR
)

brca_core_reference = load_brca_core_reference(
    brca_core_file
)

if brca_core_reference is None:

    print(
        "No BRCA core file found. "
        "Core projection will be skipped."
    )

else:

    print(
        "BRCA core file found:"
    )

    print(
        brca_core_file
    )

    display(
        brca_core_reference.head()
    )

    display(
        brca_core_reference
        .groupby(
            "core_module_name"
        )
        .size()
        .reset_index(
            name="n_genes"
        )
    )

    save_df(
        brca_core_reference,
        "BRCA_reference_core_genes_loaded.tsv",
        index=False
    )


# %% [CELL 11] ============================================================
# PROJECT BRCA CORES INTO KIRC
# ============================================================

print("=" * 88)
print("BRCA CORE PROJECTION INTO KIRC")
print("=" * 88)

core_projection_df = None
core_overlap_df = None
kirc_core_score_df = None

if brca_core_reference is None:

    print(
        "Skipped: BRCA core file unavailable."
    )

else:

    kirc_gene_set = set(
        X_df.columns.astype(str)
    )

    patient_metadata = matched[
        [
            "patient_id",
            "stage_group",
            "stage_label"
        ]
    ].copy()

    core_projection_records = []
    core_overlap_records = []

    kirc_core_score_df = (
        patient_metadata.copy()
    )

    for core_name, core_dataframe in (
        brca_core_reference.groupby(
            "core_module_name"
        )
    ):

        candidate_genes = set(
            core_dataframe[
                "raw_gene_id"
            ].astype(str)
        ) | set(
            core_dataframe[
                "harmonized_gene_id"
            ].astype(str)
        )

        matched_genes = sorted(
            candidate_genes
            &
            kirc_gene_set
        )

        core_overlap_records.append({
            "core_module_name":
                core_name,
            "n_candidate_genes_from_brca":
                len(
                    candidate_genes
                ),
            "n_matched_genes_in_kirc":
                len(
                    matched_genes
                ),
            "matched_fraction":
                (
                    len(
                        matched_genes
                    )
                    /
                    len(
                        candidate_genes
                    )
                    if len(
                        candidate_genes
                    )
                    >
                    0
                    else np.nan
                )
        })

        if len(matched_genes) < 3:

            core_projection_records.append({
                "core_module_name":
                    core_name,
                "n_matched_genes":
                    len(
                        matched_genes
                    ),
                "mean_core_score_early":
                    np.nan,
                "mean_core_score_advanced":
                    np.nan,
                "difference_advanced_minus_early":
                    np.nan,
                "cliffs_delta_advanced_vs_early":
                    np.nan,
                "cliffs_delta_magnitude":
                    np.nan,
                "mannwhitney_u_p_value":
                    np.nan
            })

            continue

        core_expression = (
            X_df[
                matched_genes
            ]
            .copy()
        )

        core_expression_z = (
            core_expression
            -
            core_expression.mean(
                axis=0
            )
        ) / core_expression.std(
            axis=0,
            ddof=0
        )

        core_expression_z = (
            core_expression_z
            .replace(
                [
                    np.inf,
                    -np.inf
                ],
                np.nan
            )
            .fillna(
                0.0
            )
        )

        core_score = (
            core_expression_z.mean(
                axis=1
            )
        )

        early_scores = (
            core_score.loc[
                matched[
                    "stage_group"
                ].to_numpy()
                ==
                "Early"
            ]
            .to_numpy()
        )

        advanced_scores = (
            core_score.loc[
                matched[
                    "stage_group"
                ].to_numpy()
                ==
                "Advanced"
            ]
            .to_numpy()
        )

        mannwhitney_result = (
            stats.mannwhitneyu(
                advanced_scores,
                early_scores,
                alternative="two-sided"
            )
        )

        delta = cliffs_delta(
            advanced_scores,
            early_scores
        )

        core_projection_records.append({
            "core_module_name":
                core_name,
            "n_matched_genes":
                len(
                    matched_genes
                ),
            "mean_core_score_early":
                float(
                    np.mean(
                        early_scores
                    )
                ),
            "mean_core_score_advanced":
                float(
                    np.mean(
                        advanced_scores
                    )
                ),
            "difference_advanced_minus_early":
                float(
                    np.mean(
                        advanced_scores
                    )
                    -
                    np.mean(
                        early_scores
                    )
                ),
            "cliffs_delta_advanced_vs_early":
                float(
                    delta
                ),
            "cliffs_delta_magnitude":
                cliffs_delta_magnitude(
                    delta
                ),
            "mannwhitney_u_p_value":
                float(
                    mannwhitney_result.pvalue
                )
        })

        kirc_core_score_df[
            f"{core_name}_score"
        ] = core_score.to_numpy()

    core_overlap_df = pd.DataFrame(
        core_overlap_records
    )

    core_projection_df = pd.DataFrame(
        core_projection_records
    )

    display(
        core_overlap_df
    )

    display(
        core_projection_df.round(6)
    )

    save_df(
        core_overlap_df,
        "KIRC_BRCA_core_overlap_summary.tsv",
        index=False
    )

    save_df(
        core_projection_df,
        "KIRC_BRCA_core_projection_results.tsv",
        index=False
    )

    save_df(
        kirc_core_score_df,
        "KIRC_BRCA_core_scores_per_patient.tsv",
        index=False
    )


# %% [CELL 12] ============================================================
# FIGURES
# ============================================================

print("=" * 88)
print("GENERATING FIGURES")
print("=" * 88)

# Figure 1: ROC curves.
plt.figure(
    figsize=(
        7,
        6
    )
)

for model_name, probability_column in [
    (
        "ElasticNet",
        "elasticnet_probability_advanced"
    ),
    (
        "ExtraTrees",
        "extratrees_probability_advanced"
    )
]:

    y_true = patient_oof[
        "true_label"
    ].to_numpy(
        dtype=int
    )

    probability = patient_oof[
        probability_column
    ].to_numpy(
        dtype=float
    )

    auc = roc_auc_score(
        y_true,
        probability
    )

    fpr, tpr, _ = roc_curve(
        y_true,
        probability
    )

    plt.plot(
        fpr,
        tpr,
        linewidth=2,
        label=(
            f"{model_name} "
            f"(AUC={auc:.3f})"
        )
    )

plt.plot(
    [
        0,
        1
    ],
    [
        0,
        1
    ],
    linestyle="--",
    linewidth=1
)

plt.xlabel(
    "False positive rate"
)

plt.ylabel(
    "True positive rate"
)

plt.title(
    "KIRC portability audit: patient-level OOF ROC"
)

plt.legend()

save_fig(
    "Figure_01_KIRC_OOF_ROC.png"
)


# Figure 2: Cross-model probability scatter.
plt.figure(
    figsize=(
        7,
        6
    )
)

for group_name in [
    "Early",
    "Advanced"
]:

    subset = patient_oof.loc[
        patient_oof[
            "true_group"
        ]
        ==
        group_name
    ]

    plt.scatter(
        subset[
            "elasticnet_probability_advanced"
        ],
        subset[
            "extratrees_probability_advanced"
        ],
        alpha=0.70,
        label=group_name
    )

elasticnet_threshold = float(
    patient_oof[
        "elasticnet_threshold"
    ].iloc[0]
)

extratrees_threshold = float(
    patient_oof[
        "extratrees_threshold"
    ].iloc[0]
)

plt.axvline(
    elasticnet_threshold,
    linestyle="--",
    linewidth=1
)

plt.axhline(
    extratrees_threshold,
    linestyle=":",
    linewidth=1
)

plt.xlabel(
    "ElasticNet probability of Advanced"
)

plt.ylabel(
    "ExtraTrees probability of Advanced"
)

plt.title(
    "KIRC cross-model probability geometry"
)

plt.legend()

save_fig(
    "Figure_02_KIRC_cross_model_probability_scatter.png"
)


# Figure 3: Fold-level GO-BP completeness.
plt.figure(
    figsize=(
        8,
        5
    )
)

values = fold_completeness[
    "representation_fraction"
].to_numpy()

x_values = np.arange(
    1,
    len(
        values
    )
    +
    1
)

plt.bar(
    x_values,
    values
)

plt.axhline(
    np.mean(
        values
    ),
    linestyle="--",
    linewidth=1,
    label=(
        f"Mean="
        f"{np.mean(values):.3f}"
    )
)

plt.xlabel(
    "Fold index"
)

plt.ylabel(
    "GO-BP representation fraction"
)

plt.title(
    "KIRC selected-gene representation completeness"
)

plt.legend()

save_fig(
    "Figure_03_KIRC_fold_level_go_bp_completeness.png"
)


# Figure 4: Top gap genes.
top_gap_plot = gap_gene_summary.head(
    TOP_GAP_GENES_TO_PLOT
)

if len(top_gap_plot) > 0:

    plt.figure(
        figsize=(
            10,
            6
        )
    )

    plt.barh(
        top_gap_plot[
            "gene_id"
        ][
            ::-1
        ],
        top_gap_plot[
            "fold_selection_frequency"
        ][
            ::-1
        ]
    )

    plt.xlabel(
        "Fold selection frequency in ExtraTrees top-100"
    )

    plt.ylabel(
        "Gene"
    )

    plt.title(
        "Top KIRC representation-gap genes"
    )

    save_fig(
        "Figure_04_KIRC_top_representation_gap_genes.png"
    )


# Figure 5: Core projection boxplots.
if kirc_core_score_df is not None:

    score_columns = [
        column
        for column in kirc_core_score_df.columns
        if column.endswith(
            "_score"
        )
    ]

    for score_column in score_columns:

        core_name = score_column.replace(
            "_score",
            ""
        )

        early_values = (
            kirc_core_score_df.loc[
                kirc_core_score_df[
                    "stage_group"
                ]
                ==
                "Early",
                score_column
            ]
            .dropna()
            .to_numpy()
        )

        advanced_values = (
            kirc_core_score_df.loc[
                kirc_core_score_df[
                    "stage_group"
                ]
                ==
                "Advanced",
                score_column
            ]
            .dropna()
            .to_numpy()
        )

        plt.figure(
            figsize=(
                5,
                5
            )
        )

        plt.boxplot(
            [
                early_values,
                advanced_values
            ],
            labels=[
                "Early",
                "Advanced"
            ]
        )

        plt.ylabel(
            "Core score"
        )

        plt.title(
            f"KIRC projection of {core_name}"
        )

        save_fig(
            f"Figure_05_{core_name}_projection_boxplot.png"
        )

print(
    "Figures saved to:",
    FIG_DIR
)


# %% [CELL 13] ============================================================
# MANUSCRIPT-READY TABLES
# ============================================================

print("=" * 88)
print("BUILDING MANUSCRIPT-READY TABLES")
print("=" * 88)

table_a = pd.DataFrame([
    {
        "metric":
            "patients",
        "value":
            X_df.shape[0]
    },
    {
        "metric":
            "genes_after_filtering",
        "value":
            X_df.shape[1]
    },
    {
        "metric":
            "early_patients",
        "value":
            int(
                np.sum(
                    y == 0
                )
            )
    },
    {
        "metric":
            "advanced_patients",
        "value":
            int(
                np.sum(
                    y == 1
                )
            )
    },
    {
        "metric":
            "repeats",
        "value":
            N_REPEATS
    },
    {
        "metric":
            "folds_per_repeat",
        "value":
            N_SPLITS
    },
    {
        "metric":
            "selected_genes_per_fold",
        "value":
            N_SELECTED_GENES
    }
])

table_b = performance_df.copy()
table_c = state_counts.copy()
table_d = completeness_summary.copy()
table_e = gap_gene_summary.head(
    40
).copy()

save_df(
    table_a,
    "Table_A_KIRC_cohort_and_design.tsv",
    index=False
)

save_df(
    table_b,
    "Table_B_KIRC_model_performance.tsv",
    index=False
)

save_df(
    table_c,
    "Table_C_KIRC_integrated_portability_states.tsv",
    index=False
)

save_df(
    table_d,
    "Table_D_KIRC_go_bp_completeness.tsv",
    index=False
)

save_df(
    table_e,
    "Table_E_KIRC_top_gap_genes.tsv",
    index=False
)

if core_projection_df is not None:

    save_df(
        core_projection_df,
        "Table_F_KIRC_BRCA_core_projection.tsv",
        index=False
    )

display(
    table_a
)

display(
    table_b.round(4)
)

display(
    table_c
)

display(
    table_d.round(6)
)

display(
    table_e.head(20)
)

if core_projection_df is not None:

    display(
        core_projection_df.round(6)
    )


# %% [CELL 14] ============================================================
# FINAL SUMMARY AND MANIFEST
# ============================================================

print("=" * 88)
print("AIDO-BBA KIRC PORTABILITY AUDIT COMPLETED")
print("=" * 88)

final_summary = pd.DataFrame([
    {
        "metric":
            "n_patients",
        "value":
            X_df.shape[0]
    },
    {
        "metric":
            "n_genes_after_filtering",
        "value":
            X_df.shape[1]
    },
    {
        "metric":
            "n_early",
        "value":
            int(
                np.sum(
                    y == 0
                )
            )
    },
    {
        "metric":
            "n_advanced",
        "value":
            int(
                np.sum(
                    y == 1
                )
            )
    },
    {
        "metric":
            "n_cv_folds_total",
        "value":
            N_REPEATS
            *
            N_SPLITS
    },
    {
        "metric":
            "mean_go_bp_representation_fraction",
        "value":
            fold_completeness[
                "representation_fraction"
            ].mean()
    },
    {
        "metric":
            "n_gap_genes_in_top100_fold_summary",
        "value":
            gap_gene_summary.shape[0]
    },
    {
        "metric":
            "elasticnet_auc",
        "value":
            float(
                performance_df
                .set_index(
                    "model_name"
                )
                .loc[
                    "ElasticNet_Logistic",
                    "auc"
                ]
            )
    },
    {
        "metric":
            "extratrees_auc",
        "value":
            float(
                performance_df
                .set_index(
                    "model_name"
                )
                .loc[
                    "ExtraTrees_BlackBox",
                    "auc"
                ]
            )
    }
])

save_df(
    final_summary,
    "KIRC_final_summary.tsv",
    index=False
)

manifest = {
    "analysis":
        "AIDO-BBA KIRC cross-cancer portability audit",
    "run_directory":
        RUN_DIR,
    "kirc_directory":
        KIRC_DIR,
    "brca_reference_run":
        BRCA_RUN_DIR,
    "go_bp_gmt":
        GO_BP_GMT,
    "n_repeats":
        N_REPEATS,
    "n_splits":
        N_SPLITS,
    "n_selected_genes":
        N_SELECTED_GENES,
    "top_importance_genes_per_fold":
        TOP_IMPORTANCE_GENES_PER_FOLD,
    "random_seed":
        RANDOM_SEED,
    "stage_endpoint_policy":
        (
            "Early versus Advanced stage was parsed only from explicitly "
            "stage-labelled clinical/pathologic/AJCC fields. Arbitrary "
            "object-column fallback was prohibited."
        ),
    "interpretation_boundary":
        (
            "This analysis evaluates portability across cancer type. "
            "It does not define KIRC-specific latent states or provide "
            "clinical recommendations."
        )
}

with open(
    os.path.join(
        RUN_DIR,
        "KIRC_portability_audit_manifest.json"
    ),
    "w",
    encoding="utf-8"
) as handle:

    json.dump(
        manifest,
        handle,
        indent=2
    )

display(
    final_summary.round(6)
)

print("\nMain output directory:")
print(RUN_DIR)

print("\nTables:")
for path in sorted(
    glob.glob(
        os.path.join(
            TABLE_DIR,
            "*"
        )
    )
):
    print(
        " -",
        os.path.basename(
            path
        )
    )

print("\nFigures:")
for path in sorted(
    glob.glob(
        os.path.join(
            FIG_DIR,
            "*"
        )
    )
):
    print(
        " -",
        os.path.basename(
            path
        )
    )
