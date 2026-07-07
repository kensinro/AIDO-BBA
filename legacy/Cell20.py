# ============================================================
# AIDO-BBA BRCA 1.0
# CELL 20
# Identifier correction, BP-size bias, and completeness-state audit
# ============================================================

from pathlib import Path
from collections import defaultdict
from scipy.stats import spearmanr, mannwhitneyu, kruskal
import numpy as np
import pandas as pd
import re
import json

# ------------------------------------------------------------
# 0. Paths
# ------------------------------------------------------------

DATA_ROOT = Path(r"D:\AIDO-Data")

NCBI_GENE_INFO = (
    DATA_ROOT
    / "NCBI_Gene"
    / "Homo_sapiens.gene_info"
)

RUN_ROOT = Path(
    r"D:\AIDO-Temp\AIDO_BBA_BRCA_1_0"
)

candidate_runs = []

for run_dir in RUN_ROOT.iterdir():

    if not run_dir.is_dir():
        continue

    required_file = (
        run_dir
        / "06_bp_reconstruction"
        / "summaries"
        / "global_bp_shap_stability.tsv"
    )

    if required_file.exists():

        candidate_runs.append(
            run_dir
        )

if len(candidate_runs) == 0:

    raise FileNotFoundError(
        "No completed BP reconstruction run was found."
    )

RUN_DIR = sorted(
    candidate_runs,
    key=lambda path: path.stat().st_mtime,
    reverse=True
)[0]

BLACKBOX_DIR = (
    RUN_DIR
    / "04_blackbox"
)

BP_SUMMARY_DIR = (
    RUN_DIR
    / "06_bp_reconstruction"
    / "summaries"
)

AUDIT_DIR = (
    RUN_DIR
    / "07_completeness_audit"
)

AUDIT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

print("=" * 78)
print("AIDO-BBA IDENTIFIER + BP-SIZE + COMPLETENESS AUDIT")
print("=" * 78)

print("\nRun:")
print(RUN_DIR)


# ============================================================
# 1. Correctly read NCBI gene_info
# ============================================================

if not NCBI_GENE_INFO.exists():

    raise FileNotFoundError(
        f"NCBI gene-info file not found:\n{NCBI_GENE_INFO}"
    )

# Read header manually because it begins with "#tax_id"
with open(
    NCBI_GENE_INFO,
    "r",
    encoding="utf-8"
) as handle:

    header_line = handle.readline().rstrip("\n")

ncbi_columns = (
    header_line
    .lstrip("#")
    .split("\t")
)

ncbi = pd.read_csv(
    NCBI_GENE_INFO,
    sep="\t",
    skiprows=1,
    names=ncbi_columns,
    dtype=str,
    low_memory=False
)

print("\nNCBI columns:")
print(ncbi.columns.tolist())

required_ncbi_columns = [
    "Symbol",
    "Synonyms",
    "GeneID"
]

missing_columns = [
    column
    for column in required_ncbi_columns
    if column not in ncbi.columns
]

if missing_columns:

    raise ValueError(
        "NCBI parsing failed. Missing columns: "
        + ", ".join(missing_columns)
    )


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


ncbi_candidate_map = defaultdict(set)

for _, row in ncbi.iterrows():

    official = clean_symbol(
        row["Symbol"]
    )

    if official is None:
        continue

    synonyms_raw = row.get(
        "Synonyms",
        ""
    )

    if pd.isna(synonyms_raw):
        continue

    synonyms = str(
        synonyms_raw
    ).split("|")

    for synonym in synonyms:

        synonym = clean_symbol(
            synonym
        )

        if (
            synonym is not None
            and synonym != official
        ):

            ncbi_candidate_map[
                synonym
            ].add(official)


ncbi_unique_alias_map = {
    alias: next(iter(targets))
    for alias, targets in ncbi_candidate_map.items()
    if len(targets) == 1
}

ncbi_ambiguous_alias_map = {
    alias: sorted(targets)
    for alias, targets in ncbi_candidate_map.items()
    if len(targets) > 1
}

print("\nCorrected NCBI mapping:")
print(
    "Unique synonyms:",
    len(ncbi_unique_alias_map)
)

print(
    "Ambiguous synonyms:",
    len(ncbi_ambiguous_alias_map)
)


# ============================================================
# 2. Audit previously unresolved model genes
# ============================================================

gene_harmonization_file = (
    BP_SUMMARY_DIR
    / "model_gene_harmonization_audit.tsv"
)

gene_harmonization = pd.read_csv(
    gene_harmonization_file,
    sep="\t",
    dtype=str
)

previously_unresolved = (
    gene_harmonization[
        gene_harmonization[
            "mapping_status"
        ] == "unresolved_retained"
    ]
    .copy()
)

previously_unresolved[
    "normalized_symbol"
] = (
    previously_unresolved[
        "normalized_symbol"
    ]
    .map(clean_symbol)
)

previously_unresolved[
    "ncbi_rescued_symbol"
] = (
    previously_unresolved[
        "normalized_symbol"
    ]
    .map(ncbi_unique_alias_map)
)

previously_unresolved[
    "ncbi_rescue_status"
] = np.select(
    [
        previously_unresolved[
            "ncbi_rescued_symbol"
        ].notna(),

        previously_unresolved[
            "normalized_symbol"
        ].isin(
            ncbi_ambiguous_alias_map
        )
    ],
    [
        "ncbi_unique_rescue",
        "ncbi_ambiguous"
    ],
    default="still_unresolved"
)

ncbi_rescue_summary = (
    previously_unresolved[
        "ncbi_rescue_status"
    ]
    .value_counts(dropna=False)
    .rename_axis(
        "ncbi_rescue_status"
    )
    .reset_index(name="n")
)

print("\nPreviously unresolved genes:")
display(
    ncbi_rescue_summary
)

previously_unresolved.to_csv(
    AUDIT_DIR
    / "ncbi_rescue_of_previously_unresolved_genes.tsv",
    sep="\t",
    index=False
)

ncbi_rescue_summary.to_csv(
    AUDIT_DIR
    / "ncbi_rescue_summary.tsv",
    sep="\t",
    index=False
)


# ============================================================
# 3. Load BP stability and completeness
# ============================================================

global_bp_stability = pd.read_csv(
    BP_SUMMARY_DIR
    / "global_bp_shap_stability.tsv",
    sep="\t"
)

patient_completeness = pd.read_csv(
    BP_SUMMARY_DIR
    / "patient_completeness_with_bba_states.tsv",
    sep="\t"
)

print("\nGlobal BP table:")
print(global_bp_stability.shape)

print("Patient completeness table:")
print(patient_completeness.shape)


# ============================================================
# 4. BP-size bias audit
# ============================================================

global_bp_stability[
    "log10_matched_gene_count"
] = np.log10(
    global_bp_stability[
        "matched_gene_count"
    ].clip(lower=1)
)

size_rho, size_p = spearmanr(
    global_bp_stability[
        "matched_gene_count"
    ],
    global_bp_stability[
        "mean_absolute_bp_shap"
    ],
    nan_policy="omit"
)

print("\nBP size vs mean absolute SHAP:")
print("Spearman rho:", round(size_rho, 4))
print("p-value:", size_p)


# ------------------------------------------------------------
# Size-normalized scores
# ------------------------------------------------------------

global_bp_stability[
    "shap_per_gene"
] = (
    global_bp_stability[
        "mean_absolute_bp_shap"
    ]
    /
    global_bp_stability[
        "matched_gene_count"
    ]
)

global_bp_stability[
    "shap_per_sqrt_gene"
] = (
    global_bp_stability[
        "mean_absolute_bp_shap"
    ]
    /
    np.sqrt(
        global_bp_stability[
            "matched_gene_count"
        ]
    )
)

# Residualized score:
# remove expected log-linear dependence on BP size
valid_mask = (
    global_bp_stability[
        "mean_absolute_bp_shap"
    ] > 0
)

x_size = (
    global_bp_stability.loc[
        valid_mask,
        "log10_matched_gene_count"
    ]
    .to_numpy(dtype=float)
)

y_shap = np.log10(
    global_bp_stability.loc[
        valid_mask,
        "mean_absolute_bp_shap"
    ].to_numpy(dtype=float)
)

slope, intercept = np.polyfit(
    x_size,
    y_shap,
    deg=1
)

expected_log_shap = (
    intercept
    + slope
    * global_bp_stability[
        "log10_matched_gene_count"
    ]
)

global_bp_stability[
    "size_residualized_log10_shap"
] = (
    np.log10(
        global_bp_stability[
            "mean_absolute_bp_shap"
        ].clip(lower=1e-15)
    )
    -
    expected_log_shap
)

global_bp_stability[
    "bp_size_class"
] = pd.cut(
    global_bp_stability[
        "matched_gene_count"
    ],
    bins=[
        0,
        50,
        200,
        500,
        1000,
        np.inf
    ],
    labels=[
        "10_to_50",
        "51_to_200",
        "201_to_500",
        "501_to_1000",
        "greater_than_1000"
    ],
    include_lowest=True
)

global_bp_stability[
    "stability_weight"
] = (
    1
    /
    (
        1
        +
        global_bp_stability[
            "absolute_bp_shap_cv"
        ].fillna(
            global_bp_stability[
                "absolute_bp_shap_cv"
            ].median()
        )
    )
)

global_bp_stability[
    "size_adjusted_stable_score"
] = (
    global_bp_stability[
        "size_residualized_log10_shap"
    ]
    *
    global_bp_stability[
        "stability_weight"
    ]
)

global_bp_stability[
    "raw_rank"
] = (
    global_bp_stability[
        "mean_absolute_bp_shap"
    ]
    .rank(
        ascending=False,
        method="min"
    )
)

global_bp_stability[
    "size_adjusted_rank"
] = (
    global_bp_stability[
        "size_adjusted_stable_score"
    ]
    .rank(
        ascending=False,
        method="min"
    )
)


print("\nTop raw BP terms:")
display(
    global_bp_stability[
        [
            "term_name",
            "matched_gene_count",
            "mean_absolute_bp_shap",
            "absolute_bp_shap_cv",
            "raw_rank"
        ]
    ]
    .sort_values("raw_rank")
    .head(20)
)

print("\nTop size-adjusted stable BP terms:")
display(
    global_bp_stability[
        [
            "term_name",
            "matched_gene_count",
            "mean_absolute_bp_shap",
            "absolute_bp_shap_cv",
            "size_residualized_log10_shap",
            "size_adjusted_stable_score",
            "size_adjusted_rank"
        ]
    ]
    .sort_values(
        "size_adjusted_rank"
    )
    .head(30)
)


# ============================================================
# 5. Completeness by clinical and BBA state
# ============================================================

state_columns = [
    column
    for column in [
        "true_group",
        "integrated_bba_state",
        "clinical_molecular_rank_state",
        "model_dependence_tier",
        "repeat_instability_tier"
    ]
    if column in patient_completeness.columns
]

metric_columns = [
    column
    for column in [
        "mean_selected_gene_coverage",
        "mean_attribution_mass_coverage",
        "mean_unmapped_absolute_shap",
        "mean_unmapped_signed_residual",
        "top10_bp_mass_fraction_mean",
        "top20_bp_mass_fraction_mean",
        "top50_bp_mass_fraction_mean",
        "top100_bp_mass_fraction_mean"
    ]
    if column in patient_completeness.columns
]

state_summary_tables = []

for state_column in state_columns:

    summary = (
        patient_completeness
        .groupby(
            state_column,
            dropna=False
        )[metric_columns]
        .agg([
            "count",
            "mean",
            "median",
            "std"
        ])
    )

    summary.columns = [
        f"{metric}_{statistic}"
        for metric, statistic
        in summary.columns
    ]

    summary = (
        summary
        .reset_index()
    )

    summary.insert(
        0,
        "state_variable",
        state_column
    )

    summary = summary.rename(
        columns={
            state_column:
                "state_value"
        }
    )

    state_summary_tables.append(
        summary
    )

completeness_by_state = pd.concat(
    state_summary_tables,
    ignore_index=True
)

print("\nCompleteness by integrated BBA state:")

if (
    "integrated_bba_state"
    in patient_completeness.columns
):

    display(
        completeness_by_state[
            completeness_by_state[
                "state_variable"
            ]
            == "integrated_bba_state"
        ]
    )


# ============================================================
# 6. Nonparametric state tests
# ============================================================

state_test_records = []

for state_column in state_columns:

    groups = (
        patient_completeness[
            state_column
        ]
        .dropna()
        .unique()
        .tolist()
    )

    if len(groups) < 2:
        continue

    for metric in metric_columns:

        arrays = []

        valid_group_names = []

        for group_name in groups:

            values = (
                patient_completeness.loc[
                    patient_completeness[
                        state_column
                    ] == group_name,
                    metric
                ]
                .dropna()
                .to_numpy(dtype=float)
            )

            if len(values) > 1:

                arrays.append(values)
                valid_group_names.append(
                    group_name
                )

        if len(arrays) < 2:
            continue

        statistic, p_value = kruskal(
            *arrays
        )

        state_test_records.append({
            "state_variable":
                state_column,

            "metric":
                metric,

            "n_groups":
                len(arrays),

            "groups":
                " | ".join(
                    map(
                        str,
                        valid_group_names
                    )
                ),

            "kruskal_h":
                float(statistic),

            "p_value":
                float(p_value),

            "D_minus_log10_p":
                float(
                    -np.log10(
                        max(
                            p_value,
                            1e-300
                        )
                    )
                )
        })

state_tests = pd.DataFrame(
    state_test_records
)

print("\nStrongest completeness-state associations:")

display(
    state_tests
    .sort_values(
        "D_minus_log10_p",
        ascending=False
    )
    .head(30)
)


# ============================================================
# 7. Residual-priority patient table
# ============================================================

patient_completeness[
    "unmapped_residual_percentile"
] = (
    patient_completeness[
        "mean_unmapped_absolute_shap"
    ]
    .rank(
        pct=True,
        method="average"
    )
)

patient_completeness[
    "low_completeness_percentile"
] = (
    1
    -
    patient_completeness[
        "mean_attribution_mass_coverage"
    ]
    .rank(
        pct=True,
        method="average"
    )
)

patient_completeness[
    "residual_priority_score"
] = (
    0.5
    * patient_completeness[
        "unmapped_residual_percentile"
    ]
    +
    0.5
    * patient_completeness[
        "low_completeness_percentile"
    ]
)

if "n_audit_flags" in patient_completeness.columns:

    max_flags = max(
        1,
        patient_completeness[
            "n_audit_flags"
        ].max()
    )

    patient_completeness[
        "integrated_residual_priority_score"
    ] = (
        0.7
        * patient_completeness[
            "residual_priority_score"
        ]
        +
        0.3
        * (
            patient_completeness[
                "n_audit_flags"
            ]
            / max_flags
        )
    )

else:

    patient_completeness[
        "integrated_residual_priority_score"
    ] = (
        patient_completeness[
            "residual_priority_score"
        ]
    )

priority_residual_patients = (
    patient_completeness
    .sort_values(
        "integrated_residual_priority_score",
        ascending=False
    )
)

priority_columns = [
    column
    for column in [
        "patient_id",
        "true_group",
        "integrated_bba_state",
        "clinical_molecular_rank_state",
        "model_dependence_tier",
        "repeat_instability_tier",
        "mean_probability_advanced",
        "mean_attribution_mass_coverage",
        "mean_unmapped_absolute_shap",
        "mean_unmapped_signed_residual",
        "n_audit_flags",
        "residual_priority_score",
        "integrated_residual_priority_score"
    ]
    if column in priority_residual_patients.columns
]

print("\nTop residual-priority patients:")

display(
    priority_residual_patients[
        priority_columns
    ].head(30)
)


# ============================================================
# 8. Save outputs
# ============================================================

bp_size_audit_summary = pd.DataFrame([
    {
        "metric":
            "spearman_bp_size_vs_mean_absolute_shap",

        "value":
            float(size_rho)
    },
    {
        "metric":
            "spearman_p_value",

        "value":
            float(size_p)
    },
    {
        "metric":
            "log10_size_slope",

        "value":
            float(slope)
    },
    {
        "metric":
            "log10_size_intercept",

        "value":
            float(intercept)
    },
    {
        "metric":
            "corrected_ncbi_unique_synonyms",

        "value":
            int(
                len(
                    ncbi_unique_alias_map
                )
            )
    },
    {
        "metric":
            "corrected_ncbi_ambiguous_synonyms",

        "value":
            int(
                len(
                    ncbi_ambiguous_alias_map
                )
            )
    },
    {
        "metric":
            "previously_unresolved_genes_rescued_by_ncbi",

        "value":
            int(
                (
                    previously_unresolved[
                        "ncbi_rescue_status"
                    ]
                    == "ncbi_unique_rescue"
                ).sum()
            )
    }
])

global_bp_stability.to_csv(
    AUDIT_DIR
    / "global_bp_size_adjusted_audit.tsv",
    sep="\t",
    index=False
)

bp_size_audit_summary.to_csv(
    AUDIT_DIR
    / "bp_size_audit_summary.tsv",
    sep="\t",
    index=False
)

completeness_by_state.to_csv(
    AUDIT_DIR
    / "completeness_by_bba_state.tsv",
    sep="\t",
    index=False
)

state_tests.to_csv(
    AUDIT_DIR
    / "completeness_state_kruskal_tests.tsv",
    sep="\t",
    index=False
)

priority_residual_patients.to_csv(
    AUDIT_DIR
    / "residual_priority_patients.tsv",
    sep="\t",
    index=False
)

priority_residual_patients.head(
    100
).to_csv(
    AUDIT_DIR
    / "residual_priority_patients_top100.tsv",
    sep="\t",
    index=False
)


# ============================================================
# 9. Final report
# ============================================================

print("\n" + "=" * 78)
print("CELL 20 COMPLETED")
print("=" * 78)

display(
    bp_size_audit_summary
)

print("\nOutput directory:")
print(AUDIT_DIR)