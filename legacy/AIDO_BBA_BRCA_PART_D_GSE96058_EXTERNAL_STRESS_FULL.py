
# %% [CELL 1] ============================================================
# AIDO-BBA BRCA 1.0
# PART D ONLY — GSE96058 EXTERNAL DATASET-REPLACEMENT STRESS TEST
#
# Primary endpoint:
#   Lymph-node status: node-negative vs node-positive
#
# Main tasks:
#   1. Parse GEO series-matrix clinical metadata.
#   2. Match external expression samples to GEO metadata.
#   3. Stream the large expression CSV and retain a high-variance
#      modeling universe without loading the complete 1.8-GB file
#      into pandas memory.
#   4. Rebuild an ExtraTrees lymph-node classifier under repeated CV.
#   5. Audit fold-level and patient-level OOF performance.
#   6. Test overlap with BRCA representation-gap and stable-core genes.
#   7. Project the three BRCA stable cores into GSE96058.
#
# Interpretation boundary:
# This is an endpoint-swapped, dataset-replacement stress test.
# It is not definitive external validation of the TCGA stage model.
# ============================================================

from pathlib import Path
from collections import Counter
import heapq
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

print("=" * 92)
print("AIDO-BBA BRCA 1.0 — PART D GSE96058 EXTERNAL STRESS TEST")
print("=" * 92)


# %% [CELL 2] ============================================================
# PATHS AND SETTINGS
# ============================================================

OUTPUT_ROOT = Path(
    r"D:\AIDO-Temp\AIDO_BBA_BRCA_1_0"
)

GSE96058_ROOT = Path(
    r"D:\AIDO-Data\External\GSE96058"
)

EXPRESSION_PATH = (
    GSE96058_ROOT
    / "GSE96058_gene_expression_3273_samples_and_136_replicates_transformed.csv"
)

SERIES_MATRIX_PATHS = [
    GSE96058_ROOT
    / "GSE96058-GPL11154_series_matrix.txt",

    GSE96058_ROOT
    / "GSE96058-GPL18573_series_matrix.txt",
]

N_SPLITS = 5
N_REPEATS = 5
TOP_K_GENES = 1500
N_TREES = 400
RANDOM_SEED = 20260701

# Streaming settings for the 1.8-GB expression file.
CSV_CHUNK_SIZE = 250
MAX_VARIANCE_GENES = 8000

MIN_MATCHED_SAMPLES = 300
MIN_CLASS_COUNT = 30

# For core projection, all stable-core genes are retained even if they
# are not among the high-variance modeling genes.
ORIENTATION_CORRECT_AUC = True

print("\nExpression:")
print(EXPRESSION_PATH)

print("\nSeries matrices:")
for path in SERIES_MATRIX_PATHS:
    print(path)


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
        candidate_runs.append(run_dir)

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
    / "21_gse96058_external_stress"
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
# INPUT CHECKS
# ============================================================

missing_files = []

for path in [
    EXPRESSION_PATH,
    *SERIES_MATRIX_PATHS,
]:
    if not path.exists():
        missing_files.append(str(path))

if missing_files:
    raise FileNotFoundError(
        "Required GSE96058 files were not found:\n"
        + "\n".join(missing_files)
    )

print("\nAll required input files were found.")


# %% [CELL 5] ============================================================
# GENERAL UTILITIES
# ============================================================

def clean_token(value):

    if value is None:
        return None

    text = str(value).strip().strip('"')

    if text == "":
        return None

    return text


def normalize_identifier(value):

    text = clean_token(value)

    if text is None:
        return None

    return re.sub(
        r"[^A-Z0-9]+",
        "",
        text.upper(),
    )


def normalize_gene_symbol(value):

    text = clean_token(value)

    if text is None:
        return None

    return text.upper()


def orientation_corrected_auc(y_true, probability):

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


def cliffs_delta_from_mannwhitney(group_a, group_b):

    group_a = np.asarray(group_a, dtype=float)
    group_b = np.asarray(group_b, dtype=float)

    if len(group_a) == 0 or len(group_b) == 0:
        return np.nan

    u_statistic, _ = mannwhitneyu(
        group_a,
        group_b,
        alternative="two-sided",
    )

    return (
        2.0
        * u_statistic
        /
        (
            len(group_a)
            *
            len(group_b)
        )
        -
        1.0
    )


def parse_node_status(value):

    if value is None or pd.isna(value):
        return np.nan

    text = str(value).strip().lower()

    if text in {
        "",
        "na",
        "nan",
        "none",
        "null",
        "unknown",
        "not available",
        "not reported",
        "missing",
    }:
        return np.nan

    # Strong negative patterns.
    negative_patterns = [
        r"\bnode[\s_-]*negative\b",
        r"\bnodal[\s_-]*negative\b",
        r"\blymph[\s_-]*node[\s_-]*negative\b",
        r"\bnegative\b",
        r"\bn0\b",
        r"\b0\s*(positive\s*)?(nodes?|lymph nodes?)\b",
        r"\bno\s+(positive\s+)?(nodes?|lymph nodes?)\b",
    ]

    for pattern in negative_patterns:
        if re.search(pattern, text):
            return 0

    # Strong positive patterns.
    positive_patterns = [
        r"\bnode[\s_-]*positive\b",
        r"\bnodal[\s_-]*positive\b",
        r"\blymph[\s_-]*node[\s_-]*positive\b",
        r"\bpositive\b",
        r"\bn[1-9]\b",
        r"\b[1-9]\d*\s*(positive\s*)?(nodes?|lymph nodes?)\b",
    ]

    for pattern in positive_patterns:
        if re.search(pattern, text):
            return 1

    # Numeric count fallback.
    numbers = re.findall(
        r"(?<!\d)\d+(?!\d)",
        text,
    )

    if len(numbers) == 1:
        count = int(numbers[0])

        if count == 0:
            return 0

        if count > 0:
            return 1

    return np.nan


# %% [CELL 6] ============================================================
# PARSE GEO SERIES-MATRIX METADATA
# ============================================================

def split_geo_line(line):

    line = line.rstrip("\n\r")

    parts = line.split("\t")

    key = parts[0].strip()

    values = [
        clean_token(part)
        for part in parts[1:]
    ]

    return key, values


def parse_series_matrix_metadata(path):

    metadata_rows = {}

    with open(
        path,
        "r",
        encoding="utf-8",
        errors="replace",
    ) as handle:

        for line in handle:

            if not line.startswith("!Sample_"):
                continue

            key, values = split_geo_line(line)

            metadata_rows.setdefault(
                key,
                [],
            ).append(values)

    accession_rows = metadata_rows.get(
        "!Sample_geo_accession",
        [],
    )

    if len(accession_rows) == 0:
        raise ValueError(
            f"No !Sample_geo_accession row found in {path}"
        )

    accessions = accession_rows[0]
    n_samples = len(accessions)

    records = [
        {
            "geo_accession":
                accessions[index],
            "series_matrix_file":
                str(path),
        }
        for index in range(n_samples)
    ]

    simple_fields = {
        "!Sample_title":
            "sample_title",
        "!Sample_source_name_ch1":
            "source_name",
        "!Sample_description":
            "sample_description",
    }

    for geo_key, output_column in simple_fields.items():

        rows = metadata_rows.get(
            geo_key,
            [],
        )

        if len(rows) == 0:
            continue

        values = rows[0]

        for index in range(
            min(
                n_samples,
                len(values),
            )
        ):
            records[index][
                output_column
            ] = values[index]

    # Characteristics can occur on multiple rows.
    characteristic_rows = metadata_rows.get(
        "!Sample_characteristics_ch1",
        [],
    )

    for row_number, values in enumerate(
        characteristic_rows,
        start=1,
    ):
        for index in range(
            min(
                n_samples,
                len(values),
            )
        ):
            records[index][
                f"characteristic_{row_number}"
            ] = values[index]

    return pd.DataFrame(records)


metadata_frames = [
    parse_series_matrix_metadata(path)
    for path in SERIES_MATRIX_PATHS
]

geo_metadata = pd.concat(
    metadata_frames,
    ignore_index=True,
)

geo_metadata = (
    geo_metadata.drop_duplicates(
        "geo_accession"
    )
    .reset_index(drop=True)
)

print("\nParsed GEO metadata:")
print(geo_metadata.shape)

print("\nMetadata columns:")
print(geo_metadata.columns.tolist())


# %% [CELL 7] ============================================================
# EXTRACT CHARACTERISTIC KEY–VALUE PAIRS
# ============================================================

characteristic_columns = [
    column
    for column in geo_metadata.columns
    if column.startswith(
        "characteristic_"
    )
]

long_characteristics = []

for row in geo_metadata.itertuples(
    index=False,
):

    row_dict = row._asdict()

    for column in characteristic_columns:

        value = row_dict.get(column)

        if value is None or pd.isna(value):
            continue

        text = str(value).strip()

        if text == "":
            continue

        if ":" in text:
            key, characteristic_value = (
                text.split(
                    ":",
                    1,
                )
            )
        elif "=" in text:
            key, characteristic_value = (
                text.split(
                    "=",
                    1,
                )
            )
        else:
            key = column
            characteristic_value = text

        long_characteristics.append({
            "geo_accession":
                row_dict[
                    "geo_accession"
                ],
            "characteristic_key":
                key.strip().lower(),
            "characteristic_value":
                characteristic_value.strip(),
            "raw_characteristic":
                text,
        })

long_characteristics = pd.DataFrame(
    long_characteristics
)

characteristic_key_summary = (
    long_characteristics[
        "characteristic_key"
    ]
    .value_counts()
    .rename_axis(
        "characteristic_key"
    )
    .reset_index(
        name="n_samples"
    )
)

print("\nMost common characteristic keys:")
display(
    characteristic_key_summary.head(
        50
    )
)


# %% [CELL 8] ============================================================
# IDENTIFY LYMPH-NODE CHARACTERISTICS
# ============================================================

node_key_mask = (
    long_characteristics[
        "characteristic_key"
    ]
    .str.contains(
        r"lymph|node|nodal|axillary",
        case=False,
        regex=True,
        na=False,
    )
)

node_characteristics = (
    long_characteristics[
        node_key_mask
    ]
    .copy()
)

print("\nCandidate node-related characteristics:")
display(
    node_characteristics[
        [
            "characteristic_key",
            "characteristic_value",
        ]
    ]
    .drop_duplicates()
    .head(100)
)

if len(node_characteristics) == 0:
    characteristic_key_summary.to_csv(
        SUMMARY_DIR
        / "available_characteristic_keys.tsv",
        sep="\t",
        index=False,
    )

    raise ValueError(
        "No lymph-node-related characteristic was found. "
        "Available keys were saved for inspection."
    )

node_characteristics[
    "node_status_binary"
] = node_characteristics[
    "characteristic_value"
].map(
    parse_node_status
)

node_characteristics_valid = (
    node_characteristics.dropna(
        subset=[
            "node_status_binary"
        ]
    )
    .copy()
)

node_characteristics_valid[
    "node_status_binary"
] = node_characteristics_valid[
    "node_status_binary"
].astype(int)

# If multiple node-related rows exist for a sample, use a consistent
# value only. Conflicting samples are excluded and audited.
node_status_audit = (
    node_characteristics_valid
    .groupby(
        "geo_accession",
        as_index=False,
    )
    .agg(
        n_node_records=(
            "node_status_binary",
            "size",
        ),
        n_unique_node_values=(
            "node_status_binary",
            "nunique",
        ),
        node_status_binary=(
            "node_status_binary",
            "first",
        ),
        node_keys=(
            "characteristic_key",
            lambda values:
                " | ".join(
                    sorted(
                        set(
                            values.astype(str)
                        )
                    )
                ),
        ),
        node_raw_values=(
            "characteristic_value",
            lambda values:
                " | ".join(
                    sorted(
                        set(
                            values.astype(str)
                        )
                    )
                ),
        ),
    )
)

conflicting_node_samples = (
    node_status_audit[
        node_status_audit[
            "n_unique_node_values"
        ]
        >
        1
    ]
    .copy()
)

node_status = (
    node_status_audit[
        node_status_audit[
            "n_unique_node_values"
        ]
        ==
        1
    ]
    .copy()
)

print("\nUsable lymph-node labels:")
display(
    node_status[
        "node_status_binary"
    ]
    .value_counts()
    .rename_axis(
        "node_status_binary"
    )
    .reset_index(
        name="n_samples"
    )
)

print("\nConflicting node-label samples:")
print(len(conflicting_node_samples))


# %% [CELL 9] ============================================================
# BUILD SAMPLE ALIAS TABLE
# ============================================================

metadata_with_node = (
    geo_metadata.merge(
        node_status[
            [
                "geo_accession",
                "node_status_binary",
                "node_keys",
                "node_raw_values",
            ]
        ],
        on="geo_accession",
        how="inner",
        validate="one_to_one",
    )
)

alias_records = []

alias_columns = [
    "geo_accession",
    "sample_title",
    "source_name",
    "sample_description",
]

for row in metadata_with_node.itertuples(
    index=False,
):

    row_dict = row._asdict()

    for alias_column in alias_columns:

        alias_value = row_dict.get(
            alias_column
        )

        if alias_value is None or pd.isna(alias_value):
            continue

        alias_normalized = normalize_identifier(
            alias_value
        )

        if alias_normalized is None:
            continue

        alias_records.append({
            "geo_accession":
                row_dict[
                    "geo_accession"
                ],
            "node_status_binary":
                row_dict[
                    "node_status_binary"
                ],
            "alias_type":
                alias_column,
            "alias_raw":
                str(
                    alias_value
                ),
            "alias_normalized":
                alias_normalized,
        })

sample_aliases = pd.DataFrame(
    alias_records
)

# Keep only unambiguous aliases.
alias_uniqueness = (
    sample_aliases
    .groupby(
        "alias_normalized"
    )[
        "geo_accession"
    ]
    .nunique()
)

unambiguous_aliases = set(
    alias_uniqueness[
        alias_uniqueness == 1
    ].index
)

sample_aliases = (
    sample_aliases[
        sample_aliases[
            "alias_normalized"
        ]
        .isin(
            unambiguous_aliases
        )
    ]
    .drop_duplicates(
        "alias_normalized"
    )
)

alias_lookup = (
    sample_aliases
    .set_index(
        "alias_normalized"
    )[
        [
            "geo_accession",
            "node_status_binary",
            "alias_type",
            "alias_raw",
        ]
    ]
    .to_dict(
        orient="index"
    )
)

print("\nUnambiguous sample aliases:")
print(len(sample_aliases))


# %% [CELL 10] ============================================================
# READ EXPRESSION HEADER AND MATCH SAMPLE COLUMNS
# ============================================================

expression_header = pd.read_csv(
    EXPRESSION_PATH,
    nrows=0,
)

expression_columns = (
    expression_header.columns.tolist()
)

print("\nExpression columns:")
print(len(expression_columns))

print("\nFirst expression columns:")
print(expression_columns[:20])

gene_column_candidates = [
    "gene",
    "gene_symbol",
    "symbol",
    "hugo_symbol",
    "Gene",
    "Gene Symbol",
    "Hugo_Symbol",
    "Unnamed: 0",
]

gene_column = None

normalized_expression_columns = {
    str(column).strip().lower():
        column
    for column in expression_columns
}

for candidate in gene_column_candidates:

    candidate_lower = candidate.lower()

    if candidate_lower in normalized_expression_columns:
        gene_column = normalized_expression_columns[
            candidate_lower
        ]
        break

if gene_column is None:
    gene_column = expression_columns[0]

matched_expression_records = []

for expression_column in expression_columns:

    if expression_column == gene_column:
        continue

    normalized_column = normalize_identifier(
        expression_column
    )

    if normalized_column in alias_lookup:

        alias_info = alias_lookup[
            normalized_column
        ]

        matched_expression_records.append({
            "expression_column":
                expression_column,
            "expression_column_normalized":
                normalized_column,
            "geo_accession":
                alias_info[
                    "geo_accession"
                ],
            "node_status_binary":
                alias_info[
                    "node_status_binary"
                ],
            "matched_alias_type":
                alias_info[
                    "alias_type"
                ],
            "matched_alias_raw":
                alias_info[
                    "alias_raw"
                ],
        })

matched_expression_samples = pd.DataFrame(
    matched_expression_records
)

# Remove duplicated GEO accessions or duplicated expression columns.
matched_expression_samples = (
    matched_expression_samples
    .drop_duplicates(
        "expression_column"
    )
    .drop_duplicates(
        "geo_accession"
    )
    .reset_index(drop=True)
)

print("\nMatched expression samples:")
print(len(matched_expression_samples))

print("\nMatched node-status counts:")
display(
    matched_expression_samples[
        "node_status_binary"
    ]
    .value_counts()
    .rename_axis(
        "node_status_binary"
    )
    .reset_index(
        name="n_samples"
    )
)

if len(matched_expression_samples) < MIN_MATCHED_SAMPLES:
    raise ValueError(
        f"Only {len(matched_expression_samples)} expression samples "
        f"matched clinical metadata; at least {MIN_MATCHED_SAMPLES} "
        "were required."
    )

class_counts = (
    matched_expression_samples[
        "node_status_binary"
    ]
    .value_counts()
)

if class_counts.min() < MIN_CLASS_COUNT:
    raise ValueError(
        "One lymph-node class contains fewer than "
        f"{MIN_CLASS_COUNT} matched samples."
    )


# %% [CELL 11] ============================================================
# LOAD BRCA REFERENCE GENES BEFORE STREAMING
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

brca_gap_gene_set = set(
    brca_gap_genes[
        "raw_gene_id"
    ]
    .astype(str)
    .str.upper()
)

stable_core_gene_set = set(
    core_manifest[
        "gene_id"
    ]
    .astype(str)
    .str.upper()
)

required_projection_genes = (
    brca_gap_gene_set
    |
    stable_core_gene_set
)

print("\nBRCA gap genes:")
print(len(brca_gap_gene_set))

print("Stable-core genes:")
print(len(stable_core_gene_set))


# %% [CELL 12] ============================================================
# STREAM EXPRESSION CSV
#
# Keeps:
#   A. top MAX_VARIANCE_GENES by variance across matched samples
#   B. all BRCA gap/stable-core genes required for projection
# ============================================================

selected_expression_columns = (
    matched_expression_samples[
        "expression_column"
    ]
    .tolist()
)

use_columns = [
    gene_column,
    *selected_expression_columns,
]

variance_heap = []
required_gene_rows = {}

n_rows_processed = 0
n_rows_numeric = 0

stream_start = time.time()

for chunk_number, chunk in enumerate(
    pd.read_csv(
        EXPRESSION_PATH,
        usecols=use_columns,
        chunksize=CSV_CHUNK_SIZE,
        low_memory=False,
    ),
    start=1,
):
    gene_symbols = (
        chunk[
            gene_column
        ]
        .map(
            normalize_gene_symbol
        )
    )

    numeric_chunk = (
        chunk[
            selected_expression_columns
        ]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
    )

    for row_position in range(
        len(chunk)
    ):
        n_rows_processed += 1

        gene_symbol = gene_symbols.iloc[
            row_position
        ]

        if gene_symbol is None:
            continue

        values = (
            numeric_chunk.iloc[
                row_position
            ]
            .to_numpy(
                dtype=np.float32
            )
        )

        finite_mask = np.isfinite(
            values
        )

        if finite_mask.sum() < max(
            20,
            int(
                0.80
                *
                len(values)
            ),
        ):
            continue

        if not finite_mask.all():
            median_value = np.nanmedian(
                values
            )

            values = np.where(
                finite_mask,
                values,
                median_value,
            ).astype(
                np.float32
            )

        variance = float(
            np.var(
                values,
                ddof=1,
            )
        )

        if not np.isfinite(
            variance
        ):
            continue

        n_rows_numeric += 1

        if gene_symbol in required_projection_genes:
            existing = required_gene_rows.get(
                gene_symbol
            )

            if (
                existing is None
                or
                variance
                >
                existing[
                    "variance"
                ]
            ):
                required_gene_rows[
                    gene_symbol
                ] = {
                    "variance":
                        variance,
                    "values":
                        values.copy(),
                }

        heap_entry = (
            variance,
            gene_symbol,
            values.copy(),
        )

        if len(
            variance_heap
        ) < MAX_VARIANCE_GENES:
            heapq.heappush(
                variance_heap,
                heap_entry,
            )

        elif variance > variance_heap[0][0]:
            heapq.heapreplace(
                variance_heap,
                heap_entry,
            )

    if (
        chunk_number == 1
        or
        chunk_number % 20 == 0
    ):
        elapsed_minutes = (
            time.time()
            -
            stream_start
        ) / 60.0

        print(
            f"Chunk {chunk_number:>4} | "
            f"rows={n_rows_processed:,} | "
            f"numeric={n_rows_numeric:,} | "
            f"{elapsed_minutes:.2f} min"
        )

print("\nStreaming completed.")

print("Rows processed:")
print(n_rows_processed)

print("Numeric gene rows:")
print(n_rows_numeric)

print("High-variance heap:")
print(len(variance_heap))

print("Required projection genes recovered:")
print(len(required_gene_rows))


# %% [CELL 13] ============================================================
# CONSTRUCT MODEL AND PROJECTION MATRICES
# ============================================================

# Deduplicate high-variance genes by retaining the highest-variance row.
variance_gene_lookup = {}

for variance, gene_symbol, values in variance_heap:

    existing = variance_gene_lookup.get(
        gene_symbol
    )

    if (
        existing is None
        or
        variance
        >
        existing[
            "variance"
        ]
    ):
        variance_gene_lookup[
            gene_symbol
        ] = {
            "variance":
                variance,
            "values":
                values,
        }

variance_genes_sorted = sorted(
    variance_gene_lookup,
    key=lambda gene:
        variance_gene_lookup[
            gene
        ][
            "variance"
        ],
    reverse=True,
)

model_gene_ids = np.asarray(
    variance_genes_sorted,
    dtype=object,
)

X = np.vstack([
    variance_gene_lookup[
        gene
    ][
        "values"
    ]
    for gene in variance_genes_sorted
]).T.astype(
    np.float32
)

projection_gene_lookup = dict(
    variance_gene_lookup
)

for gene, record in required_gene_rows.items():
    projection_gene_lookup[
        gene
    ] = record

projection_gene_ids = sorted(
    projection_gene_lookup
)

projection_gene_by_sample = pd.DataFrame(
    np.vstack([
        projection_gene_lookup[
            gene
        ][
            "values"
        ]
        for gene in projection_gene_ids
    ]),
    index=projection_gene_ids,
    columns=selected_expression_columns,
)

y = (
    matched_expression_samples[
        "node_status_binary"
    ]
    .to_numpy(
        dtype=int
    )
)

print("\nModel matrix (samples × genes):")
print(X.shape)

print("\nProjection matrix (genes × samples):")
print(projection_gene_by_sample.shape)


# %% [CELL 14] ============================================================
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
    len(y),
    dtype=float,
)

patient_probability_count = np.zeros(
    len(y),
    dtype=int,
)

selected_gene_counter = Counter()

cv_start = time.time()

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
        model_gene_ids[
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
            cv_start
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


# %% [CELL 15] ============================================================
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

patient_predictions = (
    matched_expression_samples.copy()
)

patient_predictions[
    "true_node_status"
] = np.where(
    y == 1,
    "Node_positive",
    "Node_negative",
)

patient_predictions[
    "mean_oof_probability_node_positive"
] = (
    mean_oof_probability
)

patient_predictions[
    "n_oof_predictions"
] = (
    patient_probability_count
)

patient_predictions[
    "predicted_node_status_at_0_50"
] = np.where(
    oof_predicted_label == 1,
    "Node_positive",
    "Node_negative",
)

performance_summary = pd.DataFrame([
    {
        "metric":
            "n_matched_samples",
        "value":
            len(y),
    },
    {
        "metric":
            "n_node_negative",
        "value":
            int(
                np.sum(
                    y == 0
                )
            ),
    },
    {
        "metric":
            "n_node_positive",
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
display(performance_summary)

print("\nOOF confusion matrix:")
display(
    pd.DataFrame(
        oof_confusion_matrix,
        index=[
            "True_node_negative",
            "True_node_positive",
        ],
        columns=[
            "Pred_node_negative",
            "Pred_node_positive",
        ],
    )
)


# %% [CELL 16] ============================================================
# SELECTED-GENE RECURRENCE AND BRCA OVERLAP
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
    .reset_index(drop=True)
)

gse_selected_gene_set = set(
    selected_gene_frequency[
        "gene_id"
    ]
    .astype(str)
    .str.upper()
)

gap_overlap = sorted(
    gse_selected_gene_set
    &
    brca_gap_gene_set
)

core_overlap = sorted(
    gse_selected_gene_set
    &
    stable_core_gene_set
)

gene_overlap_summary = pd.DataFrame([
    {
        "metric":
            "n_gse96058_selected_genes",
        "value":
            len(
                gse_selected_gene_set
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
            "n_gse96058_brca_gap_overlap",
        "value":
            len(
                gap_overlap
            ),
    },
    {
        "metric":
            "n_gse96058_stable_core_overlap",
        "value":
            len(
                core_overlap
            ),
    },
    {
        "metric":
            "brca_gap_overlap_fraction",
        "value":
            len(
                gap_overlap
            )
            /
            max(
                1,
                len(
                    brca_gap_gene_set
                )
            ),
    },
    {
        "metric":
            "stable_core_overlap_fraction",
        "value":
            len(
                core_overlap
            )
            /
            max(
                1,
                len(
                    stable_core_gene_set
                )
            ),
    },
])

print("\nGene-overlap summary:")
display(gene_overlap_summary)


# %% [CELL 17] ============================================================
# PROJECT STABLE CORES INTO GSE96058
# ============================================================

projection_gene_set = set(
    projection_gene_by_sample.index
    .astype(str)
    .str.upper()
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
        ]
        .astype(str)
        .str.upper()
    )

    matched_core_genes = sorted(
        core_genes
        &
        projection_gene_set
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
            "mean_node_negative":
                np.nan,
            "mean_node_positive":
                np.nan,
            "mean_difference_positive_minus_negative":
                np.nan,
            "cliffs_delta_negative_minus_positive":
                np.nan,
            "mannwhitney_p_value":
                np.nan,
        })

        continue

    core_expression = (
        projection_gene_by_sample.loc[
            matched_core_genes,
            selected_expression_columns,
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

    core_score = core_z.mean(
        axis=0
    ).to_numpy()

    node_negative_values = (
        core_score[
            y == 0
        ]
    )

    node_positive_values = (
        core_score[
            y == 1
        ]
    )

    _, p_value = mannwhitneyu(
        node_negative_values,
        node_positive_values,
        alternative="two-sided",
    )

    cliffs_delta = (
        cliffs_delta_from_mannwhitney(
            node_negative_values,
            node_positive_values,
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
        "mean_node_negative":
            float(
                np.mean(
                    node_negative_values
                )
            ),
        "mean_node_positive":
            float(
                np.mean(
                    node_positive_values
                )
            ),
        "mean_difference_positive_minus_negative":
            float(
                np.mean(
                    node_positive_values
                )
                -
                np.mean(
                    node_negative_values
                )
            ),
        "cliffs_delta_negative_minus_positive":
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
display(core_projection)


# %% [CELL 18] ============================================================
# SAVE ALL OUTPUTS
# ============================================================

geo_metadata.to_csv(
    SUMMARY_DIR
    / "gse96058_geo_metadata.tsv",
    sep="\t",
    index=False,
)

characteristic_key_summary.to_csv(
    SUMMARY_DIR
    / "gse96058_characteristic_key_summary.tsv",
    sep="\t",
    index=False,
)

node_characteristics.to_csv(
    SUMMARY_DIR
    / "gse96058_node_characteristics_raw.tsv",
    sep="\t",
    index=False,
)

node_status_audit.to_csv(
    SUMMARY_DIR
    / "gse96058_node_status_audit.tsv",
    sep="\t",
    index=False,
)

conflicting_node_samples.to_csv(
    SUMMARY_DIR
    / "gse96058_conflicting_node_labels.tsv",
    sep="\t",
    index=False,
)

matched_expression_samples.to_csv(
    SUMMARY_DIR
    / "gse96058_matched_expression_samples.tsv",
    sep="\t",
    index=False,
)

fold_results.to_csv(
    SUMMARY_DIR
    / "gse96058_fold_results.tsv",
    sep="\t",
    index=False,
)

patient_predictions.to_csv(
    SUMMARY_DIR
    / "gse96058_patient_oof_predictions.tsv",
    sep="\t",
    index=False,
)

performance_summary.to_csv(
    SUMMARY_DIR
    / "gse96058_performance_summary.tsv",
    sep="\t",
    index=False,
)

selected_gene_frequency.to_csv(
    SUMMARY_DIR
    / "gse96058_selected_gene_frequency.tsv",
    sep="\t",
    index=False,
)

gene_overlap_summary.to_csv(
    SUMMARY_DIR
    / "gse96058_brca_gene_overlap_summary.tsv",
    sep="\t",
    index=False,
)

pd.DataFrame({
    "gene_id":
        gap_overlap
}).to_csv(
    SUMMARY_DIR
    / "gse96058_brca_gap_gene_overlap.tsv",
    sep="\t",
    index=False,
)

pd.DataFrame({
    "gene_id":
        core_overlap
}).to_csv(
    SUMMARY_DIR
    / "gse96058_stable_core_gene_overlap.tsv",
    sep="\t",
    index=False,
)

core_projection.to_csv(
    SUMMARY_DIR
    / "gse96058_stable_core_projection.tsv",
    sep="\t",
    index=False,
)

final_summary = pd.concat(
    [
        performance_summary,
        gene_overlap_summary,
    ],
    ignore_index=True,
)

final_summary.to_csv(
    SUMMARY_DIR
    / "gse96058_external_stress_summary.tsv",
    sep="\t",
    index=False,
)

manifest = {
    "analysis":
        "AIDO-BBA BRCA GSE96058 external stress test",
    "run_directory":
        str(
            RUN_DIR
        ),
    "expression_path":
        str(
            EXPRESSION_PATH
        ),
    "series_matrix_paths": [
        str(path)
        for path in SERIES_MATRIX_PATHS
    ],
    "endpoint":
        "lymph-node status: node-negative versus node-positive",
    "n_splits":
        N_SPLITS,
    "n_repeats":
        N_REPEATS,
    "top_k_genes":
        TOP_K_GENES,
    "maximum_variance_genes":
        MAX_VARIANCE_GENES,
    "n_trees":
        N_TREES,
    "random_seed":
        RANDOM_SEED,
    "interpretation_boundary":
        (
            "This is an endpoint-swapped dataset-replacement "
            "stress test, not definitive external validation "
            "of the TCGA stage model."
        ),
}

with open(
    STRESS_DIR
    / "gse96058_external_stress_manifest.json",
    "w",
    encoding="utf-8",
) as handle:
    json.dump(
        manifest,
        handle,
        indent=2,
    )


# %% [CELL 19] ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 92)
print("AIDO-BBA PART D GSE96058 EXTERNAL STRESS TEST COMPLETED")
print("=" * 92)

print("\nPerformance summary:")
display(performance_summary)

print("\nGene-overlap summary:")
display(gene_overlap_summary)

print("\nStable-core projection:")
display(core_projection)

print("\nOutput directory:")
print(STRESS_DIR)
