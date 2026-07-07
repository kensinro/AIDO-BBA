# ============================================================
# AIDO-BBA BRCA 1.0
# CELL 25
# Stable representation-gap attribution module discovery
# ============================================================

from pathlib import Path
from aido_bba.config import (
    data_root, brca_output_root, brca_run_dir, go_bp_gmt,
    hgnc_file, ncbi_gene_info, metabric_dir, gse96058_dir,
    kirc_dir, kirc_output_root,
)

from itertools import combinations
from scipy.cluster.hierarchy import (
    linkage,
    fcluster,
    leaves_list
)
from scipy.spatial.distance import squareform
from scipy.stats import (
    kruskal,
    mannwhitneyu,
    spearmanr
)
from sklearn.metrics import silhouette_score

import numpy as np
import pandas as pd
import json
import warnings
import time

warnings.filterwarnings("ignore")


# ============================================================
# 0. SETTINGS
# ============================================================

OUTPUT_ROOT = brca_output_root()

K_VALUES = list(
    range(3, 13)
)

N_BOOTSTRAPS = 100
PATIENT_BOOTSTRAP_FRACTION = 0.80
RANDOM_SEED = 20260701

MIN_MODULE_SIZE = 5
MIN_GROUP_N = 10

CONSENSUS_LOWER_BOUND = 0.10
CONSENSUS_UPPER_BOUND = 0.90

rng = np.random.default_rng(
    RANDOM_SEED
)


# ============================================================
# 1. FIND COMPLETED CORRECTED RUN
# ============================================================

candidate_runs = []

for run_dir in OUTPUT_ROOT.iterdir():

    if not run_dir.is_dir():
        continue

    matrix_file = (
        run_dir
        / "11_representation_gap_genes_corrected"
        / "matrices"
        / "patient_by_gap_gene_pipeline_shap.tsv"
    )

    patient_file = (
        run_dir
        / "10_null_corrected_completeness"
        / "patient_null_corrected_completeness.tsv"
    )

    gene_file = (
        run_dir
        / "11_representation_gap_genes_corrected"
        / "summaries"
        / "global_representation_gap_gene_stability.tsv"
    )

    if all([
        matrix_file.exists(),
        patient_file.exists(),
        gene_file.exists()
    ]):

        candidate_runs.append(
            run_dir
        )

if len(candidate_runs) == 0:

    raise FileNotFoundError(
        "No corrected representation-gap run found."
    )

RUN_DIR = sorted(
    candidate_runs,
    key=lambda path: path.stat().st_mtime,
    reverse=True
)[0]

GAP_DIR = (
    RUN_DIR
    / "11_representation_gap_genes_corrected"
)

GAP_MATRIX_DIR = (
    GAP_DIR
    / "matrices"
)

GAP_SUMMARY_DIR = (
    GAP_DIR
    / "summaries"
)

NULL_CORRECTED_DIR = (
    RUN_DIR
    / "10_null_corrected_completeness"
)

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

MODULE_BOOTSTRAP_DIR = (
    MODULE_DIR
    / "bootstrap"
)

for directory in [
    MODULE_DIR,
    MODULE_SUMMARY_DIR,
    MODULE_MATRIX_DIR,
    MODULE_BOOTSTRAP_DIR
]:

    directory.mkdir(
        parents=True,
        exist_ok=True
    )


print("=" * 80)
print("AIDO-BBA REPRESENTATION-GAP MODULE DISCOVERY")
print("=" * 80)

print("\nRun:")
print(RUN_DIR)


# ============================================================
# 2. UTILITY FUNCTIONS
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
    n_1,
    n_2
):

    return float(
        (
            2.0
            * u_statistic
            /
            (
                n_1
                * n_2
            )
        )
        - 1.0
    )


def safe_gene_correlation(
    matrix
):
    """
    matrix:
        patients × genes

    Returns:
        genes × genes signed Pearson correlation.

    Constant genes are protected and assigned zero
    off-diagonal correlation.
    """

    matrix = np.asarray(
        matrix,
        dtype=float
    )

    gene_sd = np.std(
        matrix,
        axis=0,
        ddof=1
    )

    nonconstant_mask = (
        gene_sd > 0
    )

    standardized = np.zeros_like(
        matrix,
        dtype=float
    )

    standardized[
        :,
        nonconstant_mask
    ] = (
        matrix[
            :,
            nonconstant_mask
        ]
        -
        np.mean(
            matrix[
                :,
                nonconstant_mask
            ],
            axis=0
        )
    ) / gene_sd[
        nonconstant_mask
    ]

    correlation = (
        standardized.T
        @ standardized
    ) / max(
        1,
        matrix.shape[0] - 1
    )

    correlation = np.clip(
        correlation,
        -1,
        1
    )

    np.fill_diagonal(
        correlation,
        1.0
    )

    return correlation


def correlation_to_distance(
    correlation
):

    distance = (
        1.0
        -
        correlation
    ) / 2.0

    distance = np.clip(
        distance,
        0,
        1
    )

    np.fill_diagonal(
        distance,
        0.0
    )

    return distance


def hierarchical_labels(
    distance_matrix,
    n_clusters
):

    condensed_distance = squareform(
        distance_matrix,
        checks=False
    )

    linkage_matrix = linkage(
        condensed_distance,
        method="average"
    )

    labels = fcluster(
        linkage_matrix,
        t=n_clusters,
        criterion="maxclust"
    )

    return (
        labels.astype(int),
        linkage_matrix
    )


def consensus_metrics(
    consensus_matrix,
    labels
):

    within_values = []
    between_values = []

    n_genes = len(
        labels
    )

    for gene_1 in range(
        n_genes
    ):

        for gene_2 in range(
            gene_1 + 1,
            n_genes
        ):

            value = consensus_matrix[
                gene_1,
                gene_2
            ]

            if (
                labels[
                    gene_1
                ]
                ==
                labels[
                    gene_2
                ]
            ):

                within_values.append(
                    value
                )

            else:

                between_values.append(
                    value
                )

    within_values = np.asarray(
        within_values,
        dtype=float
    )

    between_values = np.asarray(
        between_values,
        dtype=float
    )

    all_off_diagonal = np.concatenate([
        within_values,
        between_values
    ])

    pac = float(
        np.mean(
            (
                all_off_diagonal
                >
                CONSENSUS_LOWER_BOUND
            )
            &
            (
                all_off_diagonal
                <
                CONSENSUS_UPPER_BOUND
            )
        )
    )

    return {
        "mean_within_consensus":
            float(
                np.mean(
                    within_values
                )
            )
            if len(
                within_values
            ) > 0
            else np.nan,

        "median_within_consensus":
            float(
                np.median(
                    within_values
                )
            )
            if len(
                within_values
            ) > 0
            else np.nan,

        "mean_between_consensus":
            float(
                np.mean(
                    between_values
                )
            )
            if len(
                between_values
            ) > 0
            else np.nan,

        "consensus_separation":
            float(
                np.mean(
                    within_values
                )
                -
                np.mean(
                    between_values
                )
            )
            if (
                len(
                    within_values
                ) > 0
                and len(
                    between_values
                ) > 0
            )
            else np.nan,

        "pac":
            pac
    }


# ============================================================
# 3. LOAD MATRIX AND MANIFESTS
# ============================================================

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

gene_stability = pd.read_csv(
    GAP_SUMMARY_DIR
    / "global_representation_gap_gene_stability.tsv",
    sep="\t"
)

patient_gene_matrix.index = (
    patient_gene_matrix.index.astype(str)
)

patient_gene_matrix.columns = (
    patient_gene_matrix.columns.astype(str)
)

patient_ids = (
    patient_gene_matrix.index
    .to_numpy(dtype=str)
)

gene_ids = (
    patient_gene_matrix.columns
    .to_numpy(dtype=str)
)

X = patient_gene_matrix.to_numpy(
    dtype=float
)

print("\nPatient × gene matrix:")
print(X.shape)

if X.shape != (1073, 250):

    print(
        "WARNING: expected (1073, 250), found",
        X.shape
    )


# ============================================================
# 4. FULL-DATA GENE CORRELATION
# ============================================================

full_correlation = safe_gene_correlation(
    X
)

full_distance = correlation_to_distance(
    full_correlation
)

np.savez_compressed(
    MODULE_MATRIX_DIR
    / "full_gene_correlation_and_distance.npz",

    genes=np.asarray(
        gene_ids,
        dtype=str
    ),

    correlation=np.asarray(
        full_correlation,
        dtype=np.float32
    ),

    distance=np.asarray(
        full_distance,
        dtype=np.float32
    )
)


# ============================================================
# 5. BOOTSTRAP CONSENSUS FOR EACH K
# ============================================================

k_results = []
consensus_by_k = {}
labels_by_k = {}
linkage_by_k = {}

bootstrap_sample_size = int(
    round(
        PATIENT_BOOTSTRAP_FRACTION
        * X.shape[0]
    )
)

start_time = time.time()

print("\n" + "=" * 80)
print("BOOTSTRAP CONSENSUS MODULE AUDIT")
print("=" * 80)

for k in K_VALUES:

    print(
        f"\nK={k}"
    )

    full_labels, full_linkage = (
        hierarchical_labels(
            full_distance,
            k
        )
    )

    labels_by_k[
        k
    ] = full_labels

    linkage_by_k[
        k
    ] = full_linkage

    consensus_counts = np.zeros(
        (
            X.shape[1],
            X.shape[1]
        ),
        dtype=np.float32
    )

    for bootstrap_id in range(
        1,
        N_BOOTSTRAPS + 1
    ):

        bootstrap_indices = rng.choice(
            X.shape[0],
            size=bootstrap_sample_size,
            replace=True
        )

        X_bootstrap = X[
            bootstrap_indices,
            :
        ]

        bootstrap_correlation = (
            safe_gene_correlation(
                X_bootstrap
            )
        )

        bootstrap_distance = (
            correlation_to_distance(
                bootstrap_correlation
            )
        )

        bootstrap_labels, _ = (
            hierarchical_labels(
                bootstrap_distance,
                k
            )
        )

        same_cluster = (
            bootstrap_labels[:, None]
            ==
            bootstrap_labels[None, :]
        )

        consensus_counts += (
            same_cluster.astype(
                np.float32
            )
        )

        if (
            bootstrap_id == 1
            or bootstrap_id % 20 == 0
            or bootstrap_id == N_BOOTSTRAPS
        ):

            print(
                f"  bootstrap "
                f"{bootstrap_id:>3}/"
                f"{N_BOOTSTRAPS}"
            )

    consensus_matrix = (
        consensus_counts
        /
        N_BOOTSTRAPS
    )

    consensus_by_k[
        k
    ] = consensus_matrix

    module_sizes = (
        pd.Series(
            full_labels
        )
        .value_counts()
        .sort_index()
    )

    minimum_module_size = int(
        module_sizes.min()
    )

    maximum_module_size = int(
        module_sizes.max()
    )

    n_small_modules = int(
        (
            module_sizes
            <
            MIN_MODULE_SIZE
        ).sum()
    )

    silhouette = silhouette_score(
        full_distance,
        full_labels,
        metric="precomputed"
    )

    consensus_summary = (
        consensus_metrics(
            consensus_matrix,
            full_labels
        )
    )

    k_results.append({
        "k":
            k,

        "silhouette":
            float(
                silhouette
            ),

        "mean_within_consensus":
            consensus_summary[
                "mean_within_consensus"
            ],

        "median_within_consensus":
            consensus_summary[
                "median_within_consensus"
            ],

        "mean_between_consensus":
            consensus_summary[
                "mean_between_consensus"
            ],

        "consensus_separation":
            consensus_summary[
                "consensus_separation"
            ],

        "pac":
            consensus_summary[
                "pac"
            ],

        "minimum_module_size":
            minimum_module_size,

        "maximum_module_size":
            maximum_module_size,

        "n_small_modules":
            n_small_modules
    })

    np.savez_compressed(
        MODULE_BOOTSTRAP_DIR
        / f"consensus_k_{k:02d}.npz",

        genes=np.asarray(
            gene_ids,
            dtype=str
        ),

        consensus_matrix=np.asarray(
            consensus_matrix,
            dtype=np.float32
        ),

        full_labels=np.asarray(
            full_labels,
            dtype=np.int16
        )
    )


# ============================================================
# 6. SELECT AUDIT-SUPPORTED K
# ============================================================

k_audit = pd.DataFrame(
    k_results
)

# Rank-based composite avoids scale domination.
k_audit[
    "rank_silhouette"
] = (
    k_audit[
        "silhouette"
    ]
    .rank(
        ascending=False,
        method="min"
    )
)

k_audit[
    "rank_consensus_separation"
] = (
    k_audit[
        "consensus_separation"
    ]
    .rank(
        ascending=False,
        method="min"
    )
)

k_audit[
    "rank_pac"
] = (
    k_audit[
        "pac"
    ]
    .rank(
        ascending=True,
        method="min"
    )
)

k_audit[
    "rank_small_modules"
] = (
    k_audit[
        "n_small_modules"
    ]
    .rank(
        ascending=True,
        method="min"
    )
)

k_audit[
    "selection_score"
] = (
    k_audit[
        "rank_silhouette"
    ]
    +
    k_audit[
        "rank_consensus_separation"
    ]
    +
    k_audit[
        "rank_pac"
    ]
    +
    k_audit[
        "rank_small_modules"
    ]
)

eligible_k_audit = k_audit[
    k_audit[
        "minimum_module_size"
    ]
    >= MIN_MODULE_SIZE
].copy()

if len(
    eligible_k_audit
) == 0:

    eligible_k_audit = (
        k_audit.copy()
    )

SELECTED_K = int(
    eligible_k_audit
    .sort_values(
        [
            "selection_score",
            "silhouette",
            "consensus_separation"
        ],
        ascending=[
            True,
            False,
            False
        ]
    )
    .iloc[0][
        "k"
    ]
)

print("\nSelected K:")
print(SELECTED_K)

display(
    k_audit.sort_values(
        "selection_score"
    )
)


# ============================================================
# 7. BUILD FINAL MODULE ASSIGNMENTS
# ============================================================

selected_labels = labels_by_k[
    SELECTED_K
]

selected_linkage = linkage_by_k[
    SELECTED_K
]

selected_consensus = consensus_by_k[
    SELECTED_K
]

leaf_order = leaves_list(
    selected_linkage
)

module_assignment = pd.DataFrame({
    "gene_id":
        gene_ids,

    "module_id_raw":
        selected_labels
})

# Renumber according to dendrogram order.
ordered_raw_modules = []

for gene_index in leaf_order:

    raw_module = int(
        selected_labels[
            gene_index
        ]
    )

    if raw_module not in ordered_raw_modules:

        ordered_raw_modules.append(
            raw_module
        )

module_renumber = {
    raw_module:
        new_module

    for new_module, raw_module
    in enumerate(
        ordered_raw_modules,
        start=1
    )
}

module_assignment[
    "module_id"
] = (
    module_assignment[
        "module_id_raw"
    ]
    .map(
        module_renumber
    )
)

module_assignment[
    "module_name"
] = (
    "GapModule_"
    +
    module_assignment[
        "module_id"
    ]
    .astype(int)
    .astype(str)
    .str.zfill(2)
)

module_assignment = (
    module_assignment
    .merge(
        gene_stability,
        left_on="gene_id",
        right_on="raw_gene_id",
        how="left",
        validate="one_to_one"
    )
)


# ============================================================
# 8. MODULE STABILITY SUMMARY
# ============================================================

module_stability_records = []

for module_id, module_df in (
    module_assignment.groupby(
        "module_id"
    )
):

    module_gene_indices = np.where(
        module_assignment[
            "module_id"
        ].to_numpy()
        ==
        module_id
    )[0]

    outside_gene_indices = np.where(
        module_assignment[
            "module_id"
        ].to_numpy()
        !=
        module_id
    )[0]

    within_consensus_values = []

    if len(
        module_gene_indices
    ) >= 2:

        for gene_1, gene_2 in combinations(
            module_gene_indices,
            2
        ):

            within_consensus_values.append(
                selected_consensus[
                    gene_1,
                    gene_2
                ]
            )

    between_consensus_values = []

    for gene_index in module_gene_indices:

        between_consensus_values.extend(
            selected_consensus[
                gene_index,
                outside_gene_indices
            ].tolist()
        )

    module_stability_records.append({
        "module_id":
            int(
                module_id
            ),

        "module_name":
            f"GapModule_{int(module_id):02d}",

        "n_genes":
            len(
                module_gene_indices
            ),

        "mean_within_consensus":
            float(
                np.mean(
                    within_consensus_values
                )
            )
            if len(
                within_consensus_values
            ) > 0
            else np.nan,

        "minimum_within_consensus":
            float(
                np.min(
                    within_consensus_values
                )
            )
            if len(
                within_consensus_values
            ) > 0
            else np.nan,

        "mean_between_consensus":
            float(
                np.mean(
                    between_consensus_values
                )
            )
            if len(
                between_consensus_values
            ) > 0
            else np.nan,

        "consensus_separation":
            (
                float(
                    np.mean(
                        within_consensus_values
                    )
                )
                -
                float(
                    np.mean(
                        between_consensus_values
                    )
                )
            )
            if (
                len(
                    within_consensus_values
                ) > 0
                and len(
                    between_consensus_values
                ) > 0
            )
            else np.nan,

        "mean_pipeline_absolute_shap":
            float(
                module_df[
                    "pipeline_mean_absolute_shap"
                ].mean()
            ),

        "sum_pipeline_absolute_shap":
            float(
                module_df[
                    "pipeline_mean_absolute_shap"
                ].sum()
            ),

        "fraction_high_support_genes":
            float(
                np.mean(
                    module_df[
                        "representation_gap_support_tier"
                    ]
                    ==
                    "high_support_gap_gene"
                )
            ),

        "fraction_moderate_or_high_support":
            float(
                np.mean(
                    module_df[
                        "representation_gap_support_tier"
                    ]
                    .isin([
                        "high_support_gap_gene",
                        "moderate_support_gap_gene"
                    ])
                )
            )
    })

module_stability = pd.DataFrame(
    module_stability_records
)

module_stability[
    "module_stability_tier"
] = np.select(
    [
        (
            module_stability[
                "mean_within_consensus"
            ] >= 0.80
        )
        &
        (
            module_stability[
                "consensus_separation"
            ] >= 0.50
        ),

        (
            module_stability[
                "mean_within_consensus"
            ] >= 0.60
        )
        &
        (
            module_stability[
                "consensus_separation"
            ] >= 0.30
        )
    ],
    [
        "high_stability",
        "moderate_stability"
    ],
    default="limited_stability"
)


# ============================================================
# 9. PATIENT × MODULE ATTRIBUTION MATRIX
# ============================================================

patient_module_matrix = pd.DataFrame(
    index=patient_gene_matrix.index
)

patient_module_absolute_matrix = pd.DataFrame(
    index=patient_gene_matrix.index
)

for module_id in sorted(
    module_assignment[
        "module_id"
    ].unique()
):

    module_genes = (
        module_assignment.loc[
            module_assignment[
                "module_id"
            ]
            ==
            module_id,
            "gene_id"
        ]
        .astype(str)
        .tolist()
    )

    module_name = (
        f"GapModule_{int(module_id):02d}"
    )

    # Signed attribution retained as a sum, because module
    # contribution is additive over member genes.
    patient_module_matrix[
        module_name
    ] = (
        patient_gene_matrix[
            module_genes
        ]
        .sum(axis=1)
    )

    patient_module_absolute_matrix[
        module_name
    ] = (
        patient_gene_matrix[
            module_genes
        ]
        .abs()
        .sum(axis=1)
    )

patient_module_matrix.index.name = (
    "patient_id"
)

patient_module_absolute_matrix.index.name = (
    "patient_id"
)


# ============================================================
# 10. MODULE DIRECTION AND PATIENT RECURRENCE
# ============================================================

module_patient_summary_records = []

for module_name in (
    patient_module_matrix.columns
):

    signed_values = (
        patient_module_matrix[
            module_name
        ]
        .to_numpy(dtype=float)
    )

    absolute_values = (
        patient_module_absolute_matrix[
            module_name
        ]
        .to_numpy(dtype=float)
    )

    module_patient_summary_records.append({
        "module_name":
            module_name,

        "mean_signed_module_attribution":
            float(
                np.mean(
                    signed_values
                )
            ),

        "median_signed_module_attribution":
            float(
                np.median(
                    signed_values
                )
            ),

        "mean_absolute_module_attribution":
            float(
                np.mean(
                    absolute_values
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
            ),

        "fraction_patients_nonzero":
            float(
                np.mean(
                    absolute_values > 0
                )
            )
    })

module_patient_summary = pd.DataFrame(
    module_patient_summary_records
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

patient_module_with_states = (
    patient_null_corrected[
        state_columns
    ]
    .merge(
        patient_module_matrix.reset_index(),
        on="patient_id",
        how="left",
        validate="one_to_one"
    )
)


# ============================================================
# 12. MODULE–STATE OMNIBUS TESTS
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
    if column in patient_module_with_states.columns
]

module_names = (
    patient_module_matrix.columns
    .tolist()
)

omnibus_records = []

for state_variable in state_variables:

    state_counts = (
        patient_module_with_states[
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

    for module_name in module_names:

        arrays = []
        valid_states = []

        for state in eligible_states:

            values = (
                patient_module_with_states.loc[
                    patient_module_with_states[
                        state_variable
                    ]
                    ==
                    state,
                    module_name
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

            "module_name":
                module_name,

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

module_state_omnibus = pd.DataFrame(
    omnibus_records
)

module_state_omnibus[
    "fdr_global"
] = benjamini_hochberg(
    module_state_omnibus[
        "p_value"
    ].to_numpy()
)

module_state_omnibus[
    "D_minus_log10_fdr"
] = (
    -np.log10(
        module_state_omnibus[
            "fdr_global"
        ].clip(
            lower=1e-300
        )
    )
)


# ============================================================
# 13. MODULE–STATE PAIRWISE TESTS
# ============================================================

pairwise_records = []

for state_variable in state_variables:

    state_counts = (
        patient_module_with_states[
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

        for module_name in module_names:

            values_1 = (
                patient_module_with_states.loc[
                    patient_module_with_states[
                        state_variable
                    ]
                    ==
                    group_1,
                    module_name
                ]
                .dropna()
                .to_numpy(dtype=float)
            )

            values_2 = (
                patient_module_with_states.loc[
                    patient_module_with_states[
                        state_variable
                    ]
                    ==
                    group_2,
                    module_name
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

            cliffs_delta = (
                cliffs_delta_from_u(
                    u_statistic,
                    len(values_1),
                    len(values_2)
                )
            )

            pairwise_records.append({
                "state_variable":
                    state_variable,

                "module_name":
                    module_name,

                "group_1":
                    str(group_1),

                "group_2":
                    str(group_2),

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
                    cliffs_delta,

                "absolute_cliffs_delta":
                    abs(
                        cliffs_delta
                    ),

                "p_value":
                    float(
                        p_value
                    )
            })

module_state_pairwise = pd.DataFrame(
    pairwise_records
)

module_state_pairwise[
    "fdr_within_state_module"
] = np.nan

for (
    state_variable,
    module_name
), row_indices in (
    module_state_pairwise
    .groupby(
        [
            "state_variable",
            "module_name"
        ]
    )
    .groups
    .items()
):

    row_indices = list(
        row_indices
    )

    module_state_pairwise.loc[
        row_indices,
        "fdr_within_state_module"
    ] = benjamini_hochberg(
        module_state_pairwise.loc[
            row_indices,
            "p_value"
        ].to_numpy()
    )

module_state_pairwise[
    "fdr_global"
] = benjamini_hochberg(
    module_state_pairwise[
        "p_value"
    ].to_numpy()
)


# ============================================================
# 14. MODULE–NULL-CORRECTED METRIC CORRELATIONS
# ============================================================

correlation_records = []

null_metrics = [
    metric
    for metric in [
        "excess_signed_residual_mean",
        "excess_coverage_mean",
        "excess_top100_fraction_mean",
        "null_corrected_priority_score"
    ]
    if metric in patient_module_with_states.columns
]

for module_name in module_names:

    for metric in null_metrics:

        rho, p_value = spearmanr(
            patient_module_with_states[
                module_name
            ],
            patient_module_with_states[
                metric
            ],
            nan_policy="omit"
        )

        correlation_records.append({
            "module_name":
                module_name,

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

module_metric_correlations = pd.DataFrame(
    correlation_records
)

module_metric_correlations[
    "fdr_global"
] = benjamini_hochberg(
    module_metric_correlations[
        "p_value"
    ].to_numpy()
)


# ============================================================
# 15. SAVE OUTPUTS
# ============================================================

k_audit.to_csv(
    MODULE_SUMMARY_DIR
    / "candidate_k_consensus_audit.tsv",
    sep="\t",
    index=False
)

module_assignment.to_csv(
    MODULE_SUMMARY_DIR
    / "gap_gene_module_assignments.tsv",
    sep="\t",
    index=False
)

module_stability.to_csv(
    MODULE_SUMMARY_DIR
    / "gap_gene_module_stability.tsv",
    sep="\t",
    index=False
)

module_patient_summary.to_csv(
    MODULE_SUMMARY_DIR
    / "gap_gene_module_patient_summary.tsv",
    sep="\t",
    index=False
)

module_state_omnibus.to_csv(
    MODULE_SUMMARY_DIR
    / "gap_module_state_omnibus_tests.tsv",
    sep="\t",
    index=False
)

module_state_pairwise.to_csv(
    MODULE_SUMMARY_DIR
    / "gap_module_state_pairwise_tests.tsv",
    sep="\t",
    index=False
)

module_metric_correlations.to_csv(
    MODULE_SUMMARY_DIR
    / "gap_module_null_metric_correlations.tsv",
    sep="\t",
    index=False
)

patient_module_matrix.to_csv(
    MODULE_MATRIX_DIR
    / "patient_by_gap_module_signed_attribution.tsv",
    sep="\t",
    index=True
)

patient_module_absolute_matrix.to_csv(
    MODULE_MATRIX_DIR
    / "patient_by_gap_module_absolute_attribution.tsv",
    sep="\t",
    index=True
)

patient_module_with_states.to_csv(
    MODULE_MATRIX_DIR
    / "patient_gap_modules_with_bba_states.tsv",
    sep="\t",
    index=False
)

np.savez_compressed(
    MODULE_MATRIX_DIR
    / "selected_module_solution.npz",

    genes=np.asarray(
        gene_ids,
        dtype=str
    ),

    module_labels=np.asarray(
        module_assignment[
            "module_id"
        ],
        dtype=np.int16
    ),

    consensus_matrix=np.asarray(
        selected_consensus,
        dtype=np.float32
    ),

    patient_ids=np.asarray(
        patient_module_matrix.index,
        dtype=str
    ),

    signed_module_matrix=np.asarray(
        patient_module_matrix.to_numpy(
            dtype=np.float32
        ),
        dtype=np.float32
    ),

    absolute_module_matrix=np.asarray(
        patient_module_absolute_matrix.to_numpy(
            dtype=np.float32
        ),
        dtype=np.float32
    )
)


# ============================================================
# 16. SUMMARY AND MANIFEST
# ============================================================

summary_table = pd.DataFrame([
    {
        "metric":
            "selected_k",

        "value":
            SELECTED_K
    },
    {
        "metric":
            "n_patients",

        "value":
            patient_module_matrix.shape[0]
    },
    {
        "metric":
            "n_gap_genes",

        "value":
            patient_gene_matrix.shape[1]
    },
    {
        "metric":
            "n_modules",

        "value":
            patient_module_matrix.shape[1]
    },
    {
        "metric":
            "minimum_module_size",

        "value":
            int(
                module_stability[
                    "n_genes"
                ].min()
            )
    },
    {
        "metric":
            "maximum_module_size",

        "value":
            int(
                module_stability[
                    "n_genes"
                ].max()
            )
    },
    {
        "metric":
            "high_stability_modules",

        "value":
            int(
                (
                    module_stability[
                        "module_stability_tier"
                    ]
                    ==
                    "high_stability"
                ).sum()
            )
    },
    {
        "metric":
            "moderate_stability_modules",

        "value":
            int(
                (
                    module_stability[
                        "module_stability_tier"
                    ]
                    ==
                    "moderate_stability"
                ).sum()
            )
    },
    {
        "metric":
            "significant_module_state_omnibus_fdr_0_05",

        "value":
            int(
                (
                    module_state_omnibus[
                        "fdr_global"
                    ]
                    <= 0.05
                ).sum()
            )
    },
    {
        "metric":
            "significant_module_state_pairwise_fdr_0_05",

        "value":
            int(
                (
                    module_state_pairwise[
                        "fdr_within_state_module"
                    ]
                    <= 0.05
                ).sum()
            )
    }
])

summary_table.to_csv(
    MODULE_SUMMARY_DIR
    / "gap_module_discovery_summary.tsv",
    sep="\t",
    index=False
)

manifest = {
    "analysis":
        "Representation-gap attribution module discovery",

    "run_directory":
        str(RUN_DIR),

    "input_matrix":
        (
            "Patient-by-gap-gene held-out "
            "pipeline-level signed SHAP matrix"
        ),

    "n_bootstraps":
        N_BOOTSTRAPS,

    "patient_bootstrap_fraction":
        PATIENT_BOOTSTRAP_FRACTION,

    "candidate_k_values":
        K_VALUES,

    "selected_k":
        SELECTED_K,

    "distance":
        (
            "Signed Pearson attribution distance: "
            "(1-correlation)/2"
        ),

    "clustering":
        "Average-linkage hierarchical clustering",

    "interpretation_boundary":
        (
            "Modules are stable attribution-pattern modules, "
            "not newly defined pathways, mechanisms, or "
            "patient subtypes."
        )
}

with open(
    MODULE_DIR
    / "gap_module_discovery_manifest.json",
    "w",
    encoding="utf-8"
) as handle:

    json.dump(
        manifest,
        handle,
        indent=2
    )


# ============================================================
# 17. FINAL REPORT
# ============================================================

duration_minutes = (
    time.time()
    - start_time
) / 60

print("\n" + "=" * 80)
print("CELL 25 COMPLETED")
print("=" * 80)

print(
    "Duration:",
    round(
        duration_minutes,
        2
    ),
    "minutes"
)

display(
    summary_table
)

print("\nCandidate K audit:")

display(
    k_audit
    .sort_values(
        "selection_score"
    )
)

print("\nSelected module stability:")

display(
    module_stability
    .sort_values(
        [
            "module_stability_tier",
            "consensus_separation"
        ],
        ascending=[
            True,
            False
        ]
    )
)

print("\nTop genes in each module:")

top_module_genes = (
    module_assignment
    .sort_values(
        [
            "module_id",
            "pipeline_mean_absolute_shap"
        ],
        ascending=[
            True,
            False
        ]
    )
    .groupby(
        "module_id",
        as_index=False
    )
    .head(10)
)

display(
    top_module_genes[
        [
            "module_name",
            "gene_id",
            "harmonized_gene_id",
            "pipeline_mean_absolute_shap",
            "pipeline_mean_signed_shap",
            "fold_selection_frequency",
            "representation_gap_support_tier"
        ]
    ]
)

print("\nStrongest module–state effects:")

module_pairwise_sorted = (
    module_state_pairwise
    .sort_values(
        [
            "fdr_within_state_module",
            "absolute_cliffs_delta"
        ],
        ascending=[
            True,
            False
        ]
    )
)

display(
    module_pairwise_sorted[
        [
            "state_variable",
            "module_name",
            "group_1",
            "group_2",
            "mean_group_1",
            "mean_group_2",
            "mean_difference",
            "cliffs_delta",
            "absolute_cliffs_delta",
            "p_value",
            "fdr_within_state_module",
            "fdr_global"
        ]
    ].head(40)
)

print("\nModule correlations with null-corrected metrics:")

display(
    module_metric_correlations
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
    .head(30)
)

print("\nOutput directory:")
print(MODULE_DIR)