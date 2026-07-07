# ============================================================
# AIDO-BBA BRCA 1.0
# CELL 26
# Stable-core representation-gap module refinement
#
# Main goal:
# Extract robust gene cores from the moderate-stability
# modules discovered in CELL 25.
#
# This does NOT define biological pathways or patient subtypes.
# ============================================================

from pathlib import Path
from aido_bba.config import (
    data_root, brca_output_root, brca_run_dir, go_bp_gmt,
    hgnc_file, ncbi_gene_info, metabric_dir, gse96058_dir,
    kirc_dir, kirc_output_root,
)

from itertools import combinations

from scipy.stats import (
    kruskal,
    mannwhitneyu,
    spearmanr
)

import numpy as np
import pandas as pd
import json
import warnings

warnings.filterwarnings("ignore")


# ============================================================
# 0. SETTINGS
# ============================================================

OUTPUT_ROOT = brca_output_root()

MIN_PARENT_MODULE_SIZE = 5

MIN_GENE_WITHIN_CONSENSUS = 0.60
MAX_GENE_BETWEEN_CONSENSUS = 0.25
MIN_GENE_CONSENSUS_MARGIN = 0.35

MIN_FOLD_SELECTION_FREQUENCY = 0.20

MIN_CORE_GENES = 5
MIN_GROUP_N = 10

# Modules satisfying the formal moderate/high stability rule
# will be considered. Micro-modules with <5 genes are excluded
# from the main stable-core analysis.
ALLOWED_STABILITY_TIERS = [
    "high_stability",
    "moderate_stability"
]


# ============================================================
# 1. FIND COMPLETED MODULE RUN
# ============================================================

candidate_runs = []

for run_dir in OUTPUT_ROOT.iterdir():

    if not run_dir.is_dir():
        continue

    required_paths = [
        (
            run_dir
            / "12_gap_gene_modules"
            / "matrices"
            / "selected_module_solution.npz"
        ),

        (
            run_dir
            / "12_gap_gene_modules"
            / "summaries"
            / "gap_gene_module_assignments.tsv"
        ),

        (
            run_dir
            / "12_gap_gene_modules"
            / "summaries"
            / "gap_gene_module_stability.tsv"
        ),

        (
            run_dir
            / "11_representation_gap_genes_corrected"
            / "matrices"
            / "patient_by_gap_gene_pipeline_shap.tsv"
        ),

        (
            run_dir
            / "10_null_corrected_completeness"
            / "patient_null_corrected_completeness.tsv"
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
        "No completed gap-module run was found."
    )

RUN_DIR = sorted(
    candidate_runs,
    key=lambda path: path.stat().st_mtime,
    reverse=True
)[0]


MODULE_DIR = (
    RUN_DIR
    / "12_gap_gene_modules"
)

MODULE_SUMMARY_DIR = (
    MODULE_DIR
    / "summaries"
)

MODULE_MATRIX_DIR = (
    MODULE_DIR
    / "matrices"
)

GAP_MATRIX_DIR = (
    RUN_DIR
    / "11_representation_gap_genes_corrected"
    / "matrices"
)

NULL_CORRECTED_DIR = (
    RUN_DIR
    / "10_null_corrected_completeness"
)

CORE_DIR = (
    RUN_DIR
    / "13_gap_module_cores"
)

CORE_SUMMARY_DIR = (
    CORE_DIR
    / "summaries"
)

CORE_MATRIX_DIR = (
    CORE_DIR
    / "matrices"
)

for directory in [
    CORE_DIR,
    CORE_SUMMARY_DIR,
    CORE_MATRIX_DIR
]:

    directory.mkdir(
        parents=True,
        exist_ok=True
    )


print("=" * 80)
print("AIDO-BBA STABLE-CORE MODULE REFINEMENT")
print("=" * 80)

print("\nRun:")
print(RUN_DIR)


# ============================================================
# 2. UTILITIES
# ============================================================

def benjamini_hochberg(
    p_values
):

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

    ordered_adjusted = (
        np.minimum.accumulate(
            ordered_adjusted[::-1]
        )[::-1]
    )

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


def cliffs_delta_from_u(
    u_statistic,
    n_group_1,
    n_group_2
):

    return float(
        (
            2.0
            * u_statistic
            /
            (
                n_group_1
                * n_group_2
            )
        )
        - 1.0
    )


def effect_size_label(
    delta
):

    absolute_delta = abs(
        delta
    )

    if absolute_delta < 0.147:
        return "negligible"

    if absolute_delta < 0.330:
        return "small"

    if absolute_delta < 0.474:
        return "medium"

    return "large"


# ============================================================
# 3. LOAD MODULE SOLUTION
# ============================================================

module_assignment = pd.read_csv(
    MODULE_SUMMARY_DIR
    / "gap_gene_module_assignments.tsv",
    sep="\t"
)

module_stability = pd.read_csv(
    MODULE_SUMMARY_DIR
    / "gap_gene_module_stability.tsv",
    sep="\t"
)

patient_gene_matrix = pd.read_csv(
    GAP_MATRIX_DIR
    / "patient_by_gap_gene_pipeline_shap.tsv",
    sep="\t",
    index_col=0
)

patient_null_corrected = pd.read_csv(
    NULL_CORRECTED_DIR
    / "patient_null_corrected_completeness.tsv",
    sep="\t"
)

with np.load(
    MODULE_MATRIX_DIR
    / "selected_module_solution.npz",
    allow_pickle=False
) as data:

    solution_genes = (
        data["genes"]
        .astype(str)
    )

    solution_module_labels = (
        data["module_labels"]
        .astype(int)
    )

    consensus_matrix = (
        data["consensus_matrix"]
        .astype(float)
    )


patient_gene_matrix.index = (
    patient_gene_matrix.index.astype(str)
)

patient_gene_matrix.columns = (
    patient_gene_matrix.columns.astype(str)
)


if len(solution_genes) != consensus_matrix.shape[0]:

    raise ValueError(
        "Consensus matrix and gene list dimensions differ."
    )

if consensus_matrix.shape[0] != consensus_matrix.shape[1]:

    raise ValueError(
        "Consensus matrix is not square."
    )


# ============================================================
# 4. MAP CONSENSUS MATRIX TO MODULE ASSIGNMENTS
# ============================================================

solution_gene_to_index = {
    gene: index
    for index, gene
    in enumerate(
        solution_genes
    )
}

module_assignment[
    "consensus_matrix_index"
] = (
    module_assignment[
        "gene_id"
    ]
    .astype(str)
    .map(
        solution_gene_to_index
    )
)

if module_assignment[
    "consensus_matrix_index"
].isna().any():

    missing_genes = (
        module_assignment.loc[
            module_assignment[
                "consensus_matrix_index"
            ].isna(),
            "gene_id"
        ]
        .astype(str)
        .tolist()
    )

    raise ValueError(
        "Module genes missing from consensus matrix: "
        + ", ".join(
            missing_genes[:20]
        )
    )

module_assignment[
    "consensus_matrix_index"
] = (
    module_assignment[
        "consensus_matrix_index"
    ].astype(int)
)


# ============================================================
# 5. CHOOSE PARENT MODULES ELIGIBLE FOR CORE EXTRACTION
# ============================================================

eligible_parent_modules = (
    module_stability[
        (
            module_stability[
                "module_stability_tier"
            ].isin(
                ALLOWED_STABILITY_TIERS
            )
        )
        &
        (
            module_stability[
                "n_genes"
            ]
            >=
            MIN_PARENT_MODULE_SIZE
        )
    ]
    .copy()
)

print("\nEligible parent modules:")

display(
    eligible_parent_modules[
        [
            "module_id",
            "module_name",
            "n_genes",
            "mean_within_consensus",
            "mean_between_consensus",
            "consensus_separation",
            "module_stability_tier"
        ]
    ]
)

if len(
    eligible_parent_modules
) == 0:

    raise ValueError(
        "No eligible moderate/high stability modules "
        "with at least five genes were found."
    )


# ============================================================
# 6. GENE-LEVEL CORE MEMBERSHIP AUDIT
# ============================================================

gene_core_records = []

all_gene_indices = np.arange(
    len(
        solution_genes
    )
)

for parent_row in (
    eligible_parent_modules.itertuples(
        index=False
    )
):

    module_id = int(
        parent_row.module_id
    )

    module_name = str(
        parent_row.module_name
    )

    module_gene_rows = (
        module_assignment[
            module_assignment[
                "module_id"
            ]
            ==
            module_id
        ]
        .copy()
    )

    module_indices = (
        module_gene_rows[
            "consensus_matrix_index"
        ]
        .to_numpy(dtype=int)
    )

    outside_indices = np.setdiff1d(
        all_gene_indices,
        module_indices
    )

    for row in module_gene_rows.itertuples(
        index=False
    ):

        gene_index = int(
            row.consensus_matrix_index
        )

        other_module_indices = (
            module_indices[
                module_indices
                !=
                gene_index
            ]
        )

        if len(
            other_module_indices
        ) > 0:

            within_values = consensus_matrix[
                gene_index,
                other_module_indices
            ]

            mean_within = float(
                np.mean(
                    within_values
                )
            )

            median_within = float(
                np.median(
                    within_values
                )
            )

            minimum_within = float(
                np.min(
                    within_values
                )
            )

        else:

            mean_within = np.nan
            median_within = np.nan
            minimum_within = np.nan

        if len(
            outside_indices
        ) > 0:

            between_values = consensus_matrix[
                gene_index,
                outside_indices
            ]

            mean_between = float(
                np.mean(
                    between_values
                )
            )

            maximum_between = float(
                np.max(
                    between_values
                )
            )

        else:

            mean_between = np.nan
            maximum_between = np.nan

        consensus_margin = (
            mean_within
            -
            mean_between
        )

        fold_selection_frequency = float(
            row.fold_selection_frequency
        )

        core_eligible = (
            np.isfinite(
                mean_within
            )
            and
            np.isfinite(
                mean_between
            )
            and
            mean_within
            >=
            MIN_GENE_WITHIN_CONSENSUS
            and
            mean_between
            <=
            MAX_GENE_BETWEEN_CONSENSUS
            and
            consensus_margin
            >=
            MIN_GENE_CONSENSUS_MARGIN
            and
            fold_selection_frequency
            >=
            MIN_FOLD_SELECTION_FREQUENCY
        )

        gene_core_records.append({
            "parent_module_id":
                module_id,

            "parent_module_name":
                module_name,

            "gene_id":
                str(
                    row.gene_id
                ),

            "harmonized_gene_id":
                str(
                    row.harmonized_gene_id
                ),

            "mean_within_consensus":
                mean_within,

            "median_within_consensus":
                median_within,

            "minimum_within_consensus":
                minimum_within,

            "mean_between_consensus":
                mean_between,

            "maximum_between_consensus":
                maximum_between,

            "consensus_margin":
                consensus_margin,

            "fold_selection_frequency":
                fold_selection_frequency,

            "pipeline_mean_absolute_shap":
                float(
                    row.pipeline_mean_absolute_shap
                ),

            "pipeline_mean_signed_shap":
                float(
                    row.pipeline_mean_signed_shap
                ),

            "representation_gap_support_tier":
                str(
                    row.representation_gap_support_tier
                ),

            "core_eligible":
                bool(
                    core_eligible
                )
        })


gene_core_audit = pd.DataFrame(
    gene_core_records
)


# ============================================================
# 7. CORE MODULE ELIGIBILITY
# ============================================================

core_size_table = (
    gene_core_audit
    .groupby(
        [
            "parent_module_id",
            "parent_module_name"
        ],
        as_index=False
    )
    .agg(
        parent_module_genes=(
            "gene_id",
            "size"
        ),

        n_core_genes=(
            "core_eligible",
            "sum"
        ),

        mean_gene_within_consensus=(
            "mean_within_consensus",
            "mean"
        ),

        mean_gene_between_consensus=(
            "mean_between_consensus",
            "mean"
        ),

        mean_consensus_margin=(
            "consensus_margin",
            "mean"
        )
    )
)

core_size_table[
    "core_fraction"
] = (
    core_size_table[
        "n_core_genes"
    ]
    /
    core_size_table[
        "parent_module_genes"
    ]
)

core_size_table[
    "core_module_retained"
] = (
    core_size_table[
        "n_core_genes"
    ]
    >=
    MIN_CORE_GENES
)


retained_core_modules = (
    core_size_table[
        core_size_table[
            "core_module_retained"
        ]
    ]
    .copy()
)

print("\nCore extraction summary:")

display(
    core_size_table
)

if len(
    retained_core_modules
) == 0:

    raise ValueError(
        "No parent module retained at least "
        f"{MIN_CORE_GENES} core genes."
    )


# ============================================================
# 8. FINAL CORE-GENE MANIFEST
# ============================================================

retained_module_names = set(
    retained_core_modules[
        "parent_module_name"
    ]
    .astype(str)
)

core_gene_manifest = (
    gene_core_audit[
        (
            gene_core_audit[
                "core_eligible"
            ]
        )
        &
        (
            gene_core_audit[
                "parent_module_name"
            ]
            .astype(str)
            .isin(
                retained_module_names
            )
        )
    ]
    .copy()
)

retained_module_order = (
    retained_core_modules[
        [
            "parent_module_id",
            "parent_module_name"
        ]
    ]
    .sort_values(
        "parent_module_id"
    )
)

core_name_map = {
    row.parent_module_name:
        f"GapCore_{position:02d}"

    for position, row
    in enumerate(
        retained_module_order.itertuples(
            index=False
        ),
        start=1
    )
}

core_gene_manifest[
    "core_module_name"
] = (
    core_gene_manifest[
        "parent_module_name"
    ]
    .map(
        core_name_map
    )
)

core_gene_manifest = (
    core_gene_manifest
    .sort_values(
        [
            "core_module_name",
            "consensus_margin",
            "pipeline_mean_absolute_shap"
        ],
        ascending=[
            True,
            False,
            False
        ]
    )
    .reset_index(drop=True)
)


# ============================================================
# 9. PATIENT × CORE-MODULE MATRICES
# ============================================================

patient_core_signed = pd.DataFrame(
    index=patient_gene_matrix.index
)

patient_core_absolute = pd.DataFrame(
    index=patient_gene_matrix.index
)

for core_module_name, core_df in (
    core_gene_manifest.groupby(
        "core_module_name"
    )
):

    core_genes = (
        core_df[
            "gene_id"
        ]
        .astype(str)
        .tolist()
    )

    missing_matrix_genes = [
        gene
        for gene in core_genes
        if gene not in patient_gene_matrix.columns
    ]

    if missing_matrix_genes:

        raise ValueError(
            f"Core genes missing from matrix for "
            f"{core_module_name}: "
            + ", ".join(
                missing_matrix_genes
            )
        )

    patient_core_signed[
        core_module_name
    ] = (
        patient_gene_matrix[
            core_genes
        ]
        .sum(axis=1)
    )

    patient_core_absolute[
        core_module_name
    ] = (
        patient_gene_matrix[
            core_genes
        ]
        .abs()
        .sum(axis=1)
    )


patient_core_signed.index.name = (
    "patient_id"
)

patient_core_absolute.index.name = (
    "patient_id"
)


# ============================================================
# 10. CORE MODULE DESCRIPTIVE SUMMARY
# ============================================================

core_module_summary_records = []

for core_module_name, core_df in (
    core_gene_manifest.groupby(
        "core_module_name"
    )
):

    signed_values = (
        patient_core_signed[
            core_module_name
        ]
        .to_numpy(dtype=float)
    )

    absolute_values = (
        patient_core_absolute[
            core_module_name
        ]
        .to_numpy(dtype=float)
    )

    core_module_summary_records.append({
        "core_module_name":
            core_module_name,

        "parent_module_name":
            core_df[
                "parent_module_name"
            ].iloc[0],

        "n_core_genes":
            len(
                core_df
            ),

        "mean_gene_within_consensus":
            float(
                core_df[
                    "mean_within_consensus"
                ].mean()
            ),

        "mean_gene_between_consensus":
            float(
                core_df[
                    "mean_between_consensus"
                ].mean()
            ),

        "mean_gene_consensus_margin":
            float(
                core_df[
                    "consensus_margin"
                ].mean()
            ),

        "mean_pipeline_absolute_shap_per_gene":
            float(
                core_df[
                    "pipeline_mean_absolute_shap"
                ].mean()
            ),

        "sum_pipeline_absolute_shap":
            float(
                core_df[
                    "pipeline_mean_absolute_shap"
                ].sum()
            ),

        "mean_patient_signed_attribution":
            float(
                np.mean(
                    signed_values
                )
            ),

        "median_patient_signed_attribution":
            float(
                np.median(
                    signed_values
                )
            ),

        "mean_patient_absolute_attribution":
            float(
                np.mean(
                    absolute_values
                )
            ),

        "fraction_patients_nonzero":
            float(
                np.mean(
                    absolute_values > 0
                )
            ),

        "fraction_patients_toward_advanced":
            float(
                np.mean(
                    signed_values > 0
                )
            ),

        "fraction_patients_toward_early":
            float(
                np.mean(
                    signed_values < 0
                )
            )
    })


core_module_summary = pd.DataFrame(
    core_module_summary_records
)


# ============================================================
# 11. MERGE PATIENT STATES
# ============================================================

state_columns = [
    column
    for column in [
        "patient_id",
        "true_group",
        "integrated_bba_state",
        "clinical_molecular_rank_state",
        "model_dependence_tier",
        "repeat_instability_tier",
        "excess_signed_residual_mean",
        "excess_coverage_mean",
        "excess_top100_fraction_mean",
        "null_corrected_priority_score"
    ]
    if column in patient_null_corrected.columns
]

patient_core_with_states = (
    patient_null_corrected[
        state_columns
    ]
    .merge(
        patient_core_signed.reset_index(),
        on="patient_id",
        how="left",
        validate="one_to_one"
    )
)


# ============================================================
# 12. CORE MODULE–STATE OMNIBUS TESTS
# ============================================================

state_variables = [
    column
    for column in [
        "true_group",
        "integrated_bba_state",
        "clinical_molecular_rank_state",
        "model_dependence_tier",
        "repeat_instability_tier"
    ]
    if column in patient_core_with_states.columns
]

core_module_names = (
    patient_core_signed.columns
    .tolist()
)

omnibus_records = []

for state_variable in state_variables:

    state_counts = (
        patient_core_with_states[
            state_variable
        ]
        .value_counts()
    )

    eligible_states = (
        state_counts[
            state_counts >= MIN_GROUP_N
        ]
        .index
        .tolist()
    )

    for core_module_name in core_module_names:

        arrays = []
        valid_states = []

        for state in eligible_states:

            values = (
                patient_core_with_states.loc[
                    patient_core_with_states[
                        state_variable
                    ]
                    ==
                    state,
                    core_module_name
                ]
                .dropna()
                .to_numpy(dtype=float)
            )

            if len(values) >= MIN_GROUP_N:

                arrays.append(
                    values
                )

                valid_states.append(
                    str(state)
                )

        if len(arrays) < 2:
            continue

        h_statistic, p_value = kruskal(
            *arrays
        )

        omnibus_records.append({
            "state_variable":
                state_variable,

            "core_module_name":
                core_module_name,

            "n_groups":
                len(arrays),

            "groups":
                " | ".join(
                    valid_states
                ),

            "kruskal_h":
                float(
                    h_statistic
                ),

            "p_value":
                float(
                    p_value
                )
        })


core_state_omnibus = pd.DataFrame(
    omnibus_records
)

core_state_omnibus[
    "fdr_global"
] = benjamini_hochberg(
    core_state_omnibus[
        "p_value"
    ].to_numpy()
)

core_state_omnibus[
    "D_minus_log10_fdr"
] = (
    -np.log10(
        core_state_omnibus[
            "fdr_global"
        ].clip(
            lower=1e-300
        )
    )
)


# ============================================================
# 13. CORE MODULE–STATE PAIRWISE TESTS
# ============================================================

pairwise_records = []

for state_variable in state_variables:

    state_counts = (
        patient_core_with_states[
            state_variable
        ]
        .value_counts()
    )

    eligible_states = (
        state_counts[
            state_counts >= MIN_GROUP_N
        ]
        .index
        .tolist()
    )

    for group_1, group_2 in combinations(
        eligible_states,
        2
    ):

        for core_module_name in core_module_names:

            values_1 = (
                patient_core_with_states.loc[
                    patient_core_with_states[
                        state_variable
                    ]
                    ==
                    group_1,
                    core_module_name
                ]
                .dropna()
                .to_numpy(dtype=float)
            )

            values_2 = (
                patient_core_with_states.loc[
                    patient_core_with_states[
                        state_variable
                    ]
                    ==
                    group_2,
                    core_module_name
                ]
                .dropna()
                .to_numpy(dtype=float)
            )

            if (
                len(values_1) < MIN_GROUP_N
                or len(values_2) < MIN_GROUP_N
            ):
                continue

            u_statistic, p_value = mannwhitneyu(
                values_1,
                values_2,
                alternative="two-sided"
            )

            delta = cliffs_delta_from_u(
                u_statistic,
                len(values_1),
                len(values_2)
            )

            pairwise_records.append({
                "state_variable":
                    state_variable,

                "core_module_name":
                    core_module_name,

                "group_1":
                    str(
                        group_1
                    ),

                "group_2":
                    str(
                        group_2
                    ),

                "n_group_1":
                    len(values_1),

                "n_group_2":
                    len(values_2),

                "mean_group_1":
                    float(
                        np.mean(
                            values_1
                        )
                    ),

                "mean_group_2":
                    float(
                        np.mean(
                            values_2
                        )
                    ),

                "mean_difference":
                    float(
                        np.mean(
                            values_1
                        )
                        -
                        np.mean(
                            values_2
                        )
                    ),

                "median_group_1":
                    float(
                        np.median(
                            values_1
                        )
                    ),

                "median_group_2":
                    float(
                        np.median(
                            values_2
                        )
                    ),

                "cliffs_delta":
                    delta,

                "absolute_cliffs_delta":
                    abs(
                        delta
                    ),

                "effect_size":
                    effect_size_label(
                        delta
                    ),

                "p_value":
                    float(
                        p_value
                    )
            })


core_state_pairwise = pd.DataFrame(
    pairwise_records
)

core_state_pairwise[
    "fdr_within_state_core"
] = np.nan

for (
    state_variable,
    core_module_name
), row_indices in (
    core_state_pairwise
    .groupby(
        [
            "state_variable",
            "core_module_name"
        ]
    )
    .groups
    .items()
):

    row_indices = list(
        row_indices
    )

    core_state_pairwise.loc[
        row_indices,
        "fdr_within_state_core"
    ] = benjamini_hochberg(
        core_state_pairwise.loc[
            row_indices,
            "p_value"
        ].to_numpy()
    )

core_state_pairwise[
    "fdr_global"
] = benjamini_hochberg(
    core_state_pairwise[
        "p_value"
    ].to_numpy()
)


# ============================================================
# 14. CORRELATIONS WITH NULL-CORRECTED METRICS
# ============================================================

null_metrics = [
    column
    for column in [
        "excess_signed_residual_mean",
        "excess_coverage_mean",
        "excess_top100_fraction_mean",
        "null_corrected_priority_score"
    ]
    if column in patient_core_with_states.columns
]

correlation_records = []

for core_module_name in core_module_names:

    for metric in null_metrics:

        rho, p_value = spearmanr(
            patient_core_with_states[
                core_module_name
            ],
            patient_core_with_states[
                metric
            ],
            nan_policy="omit"
        )

        correlation_records.append({
            "core_module_name":
                core_module_name,

            "metric":
                metric,

            "spearman_rho":
                float(
                    rho
                ),

            "p_value":
                float(
                    p_value
                )
        })


core_metric_correlations = pd.DataFrame(
    correlation_records
)

core_metric_correlations[
    "fdr_global"
] = benjamini_hochberg(
    core_metric_correlations[
        "p_value"
    ].to_numpy()
)


# ============================================================
# 15. PARENT VS CORE CONCORDANCE
# ============================================================

parent_module_matrix = pd.read_csv(
    MODULE_MATRIX_DIR
    / "patient_by_gap_module_signed_attribution.tsv",
    sep="\t",
    index_col=0
)

parent_module_matrix.index = (
    parent_module_matrix.index.astype(str)
)

parent_core_records = []

for core_module_name, core_df in (
    core_gene_manifest.groupby(
        "core_module_name"
    )
):

    parent_module_name = str(
        core_df[
            "parent_module_name"
        ].iloc[0]
    )

    parent_values = (
        parent_module_matrix.loc[
            patient_core_signed.index,
            parent_module_name
        ]
        .to_numpy(dtype=float)
    )

    core_values = (
        patient_core_signed[
            core_module_name
        ]
        .to_numpy(dtype=float)
    )

    rho, p_value = spearmanr(
        parent_values,
        core_values
    )

    parent_core_records.append({
        "core_module_name":
            core_module_name,

        "parent_module_name":
            parent_module_name,

        "spearman_parent_core":
            float(
                rho
            ),

        "p_value":
            float(
                p_value
            ),

        "mean_parent_attribution":
            float(
                np.mean(
                    parent_values
                )
            ),

        "mean_core_attribution":
            float(
                np.mean(
                    core_values
                )
            ),

        "mean_absolute_difference":
            float(
                np.mean(
                    np.abs(
                        parent_values
                        -
                        core_values
                    )
                )
            )
    })


parent_core_concordance = pd.DataFrame(
    parent_core_records
)


# ============================================================
# 16. SAVE OUTPUTS
# ============================================================

gene_core_audit.to_csv(
    CORE_SUMMARY_DIR
    / "gene_core_membership_audit.tsv",
    sep="\t",
    index=False
)

core_size_table.to_csv(
    CORE_SUMMARY_DIR
    / "core_extraction_summary.tsv",
    sep="\t",
    index=False
)

core_gene_manifest.to_csv(
    CORE_SUMMARY_DIR
    / "stable_core_gene_manifest.tsv",
    sep="\t",
    index=False
)

core_module_summary.to_csv(
    CORE_SUMMARY_DIR
    / "stable_core_module_summary.tsv",
    sep="\t",
    index=False
)

core_state_omnibus.to_csv(
    CORE_SUMMARY_DIR
    / "stable_core_state_omnibus_tests.tsv",
    sep="\t",
    index=False
)

core_state_pairwise.to_csv(
    CORE_SUMMARY_DIR
    / "stable_core_state_pairwise_tests.tsv",
    sep="\t",
    index=False
)

core_metric_correlations.to_csv(
    CORE_SUMMARY_DIR
    / "stable_core_null_metric_correlations.tsv",
    sep="\t",
    index=False
)

parent_core_concordance.to_csv(
    CORE_SUMMARY_DIR
    / "parent_core_attribution_concordance.tsv",
    sep="\t",
    index=False
)

patient_core_signed.to_csv(
    CORE_MATRIX_DIR
    / "patient_by_stable_core_signed_attribution.tsv",
    sep="\t",
    index=True
)

patient_core_absolute.to_csv(
    CORE_MATRIX_DIR
    / "patient_by_stable_core_absolute_attribution.tsv",
    sep="\t",
    index=True
)

patient_core_with_states.to_csv(
    CORE_MATRIX_DIR
    / "patient_stable_cores_with_bba_states.tsv",
    sep="\t",
    index=False
)

np.savez_compressed(
    CORE_MATRIX_DIR
    / "stable_core_solution.npz",

    patient_ids=np.asarray(
        patient_core_signed.index,
        dtype=str
    ),

    core_module_names=np.asarray(
        patient_core_signed.columns,
        dtype=str
    ),

    signed_core_matrix=np.asarray(
        patient_core_signed.to_numpy(
            dtype=np.float32
        ),
        dtype=np.float32
    ),

    absolute_core_matrix=np.asarray(
        patient_core_absolute.to_numpy(
            dtype=np.float32
        ),
        dtype=np.float32
    ),

    core_genes=np.asarray(
        core_gene_manifest[
            "gene_id"
        ],
        dtype=str
    ),

    core_gene_modules=np.asarray(
        core_gene_manifest[
            "core_module_name"
        ],
        dtype=str
    )
)


# ============================================================
# 17. SUMMARY
# ============================================================

summary_table = pd.DataFrame([
    {
        "metric":
            "eligible_parent_modules",

        "value":
            len(
                eligible_parent_modules
            )
    },
    {
        "metric":
            "retained_core_modules",

        "value":
            len(
                retained_core_modules
            )
    },
    {
        "metric":
            "total_core_genes",

        "value":
            len(
                core_gene_manifest
            )
    },
    {
        "metric":
            "minimum_core_size",

        "value":
            int(
                core_module_summary[
                    "n_core_genes"
                ].min()
            )
    },
    {
        "metric":
            "maximum_core_size",

        "value":
            int(
                core_module_summary[
                    "n_core_genes"
                ].max()
            )
    },
    {
        "metric":
            "significant_core_state_omnibus_fdr_0_05",

        "value":
            int(
                (
                    core_state_omnibus[
                        "fdr_global"
                    ]
                    <= 0.05
                ).sum()
            )
    },
    {
        "metric":
            "significant_core_state_pairwise_fdr_0_05",

        "value":
            int(
                (
                    core_state_pairwise[
                        "fdr_within_state_core"
                    ]
                    <= 0.05
                ).sum()
            )
    },
    {
        "metric":
            "medium_or_large_core_pairwise_effects",

        "value":
            int(
                core_state_pairwise[
                    "effect_size"
                ]
                .isin([
                    "medium",
                    "large"
                ])
                .sum()
            )
    }
])

summary_table.to_csv(
    CORE_SUMMARY_DIR
    / "stable_core_refinement_summary.tsv",
    sep="\t",
    index=False
)


manifest = {
    "analysis":
        "Stable-core refinement of representation-gap modules",

    "run_directory":
        str(RUN_DIR),

    "eligible_parent_stability_tiers":
        ALLOWED_STABILITY_TIERS,

    "minimum_parent_module_size":
        MIN_PARENT_MODULE_SIZE,

    "minimum_gene_within_consensus":
        MIN_GENE_WITHIN_CONSENSUS,

    "maximum_gene_between_consensus":
        MAX_GENE_BETWEEN_CONSENSUS,

    "minimum_gene_consensus_margin":
        MIN_GENE_CONSENSUS_MARGIN,

    "minimum_fold_selection_frequency":
        MIN_FOLD_SELECTION_FREQUENCY,

    "minimum_core_genes":
        MIN_CORE_GENES,

    "interpretation_boundary":
        (
            "Stable cores are explanatory attribution "
            "structures. They are not claimed as new pathways, "
            "mechanisms, or disease subtypes."
        )
}

with open(
    CORE_DIR
    / "stable_core_refinement_manifest.json",
    "w",
    encoding="utf-8"
) as handle:

    json.dump(
        manifest,
        handle,
        indent=2
    )


# ============================================================
# 18. FINAL REPORT
# ============================================================

print("\n" + "=" * 80)
print("CELL 26 COMPLETED")
print("=" * 80)

display(
    summary_table
)

print("\nCore extraction:")

display(
    core_size_table
)

print("\nStable-core module summary:")

display(
    core_module_summary
)

print("\nTop core genes:")

display(
    core_gene_manifest[
        [
            "core_module_name",
            "parent_module_name",
            "gene_id",
            "harmonized_gene_id",
            "mean_within_consensus",
            "mean_between_consensus",
            "consensus_margin",
            "fold_selection_frequency",
            "pipeline_mean_absolute_shap",
            "representation_gap_support_tier"
        ]
    ]
    .groupby(
        "core_module_name",
        as_index=False
    )
    .head(15)
)

print("\nParent–core attribution concordance:")

display(
    parent_core_concordance
)

print("\nStrongest stable-core state effects:")

core_pairwise_sorted = (
    core_state_pairwise
    .sort_values(
        [
            "fdr_within_state_core",
            "absolute_cliffs_delta"
        ],
        ascending=[
            True,
            False
        ]
    )
)

display(
    core_pairwise_sorted[
        [
            "state_variable",
            "core_module_name",
            "group_1",
            "group_2",
            "mean_group_1",
            "mean_group_2",
            "mean_difference",
            "cliffs_delta",
            "absolute_cliffs_delta",
            "effect_size",
            "p_value",
            "fdr_within_state_core",
            "fdr_global"
        ]
    ].head(40)
)

print("\nStable-core correlations with null-corrected metrics:")

display(
    core_metric_correlations
    .sort_values(
        [
            "fdr_global",
            "spearman_rho"
        ],
        ascending=[
            True,
            False
        ]
    )
)

print("\nOutput directory:")
print(CORE_DIR)