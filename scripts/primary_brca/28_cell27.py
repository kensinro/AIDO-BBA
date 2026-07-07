# ============================================================
# AIDO-BBA BRCA 1.0
# CELL 27
# Repeated fuzzy explanatory-state reconstruction
#
# Input:
# - corrected held-out patient-repeat gap-gene SHAP
# - stable-core gene manifest
# - null-corrected patient/BBA states
#
# Output:
# - patient × repeat × core attribution
# - repeat-normalized core percentiles
# - fuzzy memberships
# - dominant/blended explanatory-state taxonomy
# - repeat consistency audit
# - state association tests
#
# Important:
# These are candidate explanatory states, not biological
# subtypes or ontology-level disease classes.
# ============================================================

from pathlib import Path
from aido_bba.config import (
    data_root, brca_output_root, brca_run_dir, go_bp_gmt,
    hgnc_file, ncbi_gene_info, metabric_dir, gse96058_dir,
    kirc_dir, kirc_output_root,
)

from itertools import product, combinations

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

N_EXPECTED_REPEATS = 5

MIN_GROUP_N = 10

# Descriptive fuzzy-state thresholds.
# These do not define biological subtypes.
HIGH_MEMBERSHIP_THRESHOLD = 0.75
LOW_MEMBERSHIP_THRESHOLD = 0.25
DOMINANCE_MARGIN = 0.15

# A repeat-level state is considered stable when the same
# state occurs in at least this fraction of repeats.
MIN_STATE_REPEAT_CONSISTENCY = 0.80


# ============================================================
# 1. FIND COMPLETED CORE RUN
# ============================================================

candidate_runs = []

for run_dir in OUTPUT_ROOT.iterdir():

    if not run_dir.is_dir():
        continue

    required_paths = [
        (
            run_dir
            / "13_gap_module_cores"
            / "summaries"
            / "stable_core_gene_manifest.tsv"
        ),

        (
            run_dir
            / "11_representation_gap_genes_corrected"
            / "summaries"
            / "patient_repeat_gap_gene_attributions.tsv"
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
        "No completed stable-core run was found."
    )

RUN_DIR = sorted(
    candidate_runs,
    key=lambda path: path.stat().st_mtime,
    reverse=True
)[0]


CORE_DIR = (
    RUN_DIR
    / "13_gap_module_cores"
)

CORE_SUMMARY_DIR = (
    CORE_DIR
    / "summaries"
)

GAP_SUMMARY_DIR = (
    RUN_DIR
    / "11_representation_gap_genes_corrected"
    / "summaries"
)

NULL_CORRECTED_DIR = (
    RUN_DIR
    / "10_null_corrected_completeness"
)

FUZZY_DIR = (
    RUN_DIR
    / "14_fuzzy_explanatory_states"
)

FUZZY_SUMMARY_DIR = (
    FUZZY_DIR
    / "summaries"
)

FUZZY_MATRIX_DIR = (
    FUZZY_DIR
    / "matrices"
)

for directory in [
    FUZZY_DIR,
    FUZZY_SUMMARY_DIR,
    FUZZY_MATRIX_DIR
]:

    directory.mkdir(
        parents=True,
        exist_ok=True
    )


print("=" * 80)
print("AIDO-BBA FUZZY EXPLANATORY-STATE RECONSTRUCTION")
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


def classify_fuzzy_state(
    memberships,
    core_names
):
    """
    memberships:
        fuzzy high-membership values between 0 and 1.

    Rules:
    - >=2 high cores: shared_high
    - exactly 1 high core and sufficient dominance margin:
      dominant core
    - all cores low: shared_low
    - otherwise: blended_intermediate
    """

    memberships = np.asarray(
        memberships,
        dtype=float
    )

    high_mask = (
        memberships
        >=
        HIGH_MEMBERSHIP_THRESHOLD
    )

    low_mask = (
        memberships
        <=
        LOW_MEMBERSHIP_THRESHOLD
    )

    n_high = int(
        high_mask.sum()
    )

    if n_high >= 2:

        high_names = [
            core_names[index]
            for index in np.where(
                high_mask
            )[0]
        ]

        return (
            "shared_high__"
            +
            "__".join(
                high_names
            )
        )

    if n_high == 1:

        dominant_index = int(
            np.argmax(
                memberships
            )
        )

        sorted_memberships = np.sort(
            memberships
        )[::-1]

        margin = (
            sorted_memberships[0]
            -
            sorted_memberships[1]
        )

        if margin >= DOMINANCE_MARGIN:

            return (
                "dominant__"
                +
                core_names[
                    dominant_index
                ]
            )

    if bool(
        np.all(
            low_mask
        )
    ):

        return "shared_low"

    return "blended_intermediate"


# ============================================================
# 3. LOAD INPUTS
# ============================================================

core_gene_manifest = pd.read_csv(
    CORE_SUMMARY_DIR
    / "stable_core_gene_manifest.tsv",
    sep="\t",
    dtype=str
)

patient_repeat_gene = pd.read_csv(
    GAP_SUMMARY_DIR
    / "patient_repeat_gap_gene_attributions.tsv",
    sep="\t"
)

patient_null_corrected = pd.read_csv(
    NULL_CORRECTED_DIR
    / "patient_null_corrected_completeness.tsv",
    sep="\t"
)

core_names = sorted(
    core_gene_manifest[
        "core_module_name"
    ]
    .astype(str)
    .unique()
    .tolist()
)

patient_ids = (
    patient_null_corrected[
        "patient_id"
    ]
    .astype(str)
    .tolist()
)

repeat_ids = list(
    range(
        1,
        N_EXPECTED_REPEATS + 1
    )
)

print("\nStable cores:")
print(core_names)

print("Patients:", len(patient_ids))
print("Patient-repeat gap-gene rows:", len(patient_repeat_gene))


# ============================================================
# 4. MAP GENES TO STABLE CORES
# ============================================================

gene_to_core = dict(
    zip(
        core_gene_manifest[
            "gene_id"
        ].astype(str),

        core_gene_manifest[
            "core_module_name"
        ].astype(str)
    )
)

core_gene_set = set(
    gene_to_core
)

patient_repeat_core_long = (
    patient_repeat_gene[
        patient_repeat_gene[
            "raw_gene_id"
        ]
        .astype(str)
        .isin(
            core_gene_set
        )
    ][
        [
            "patient_id",
            "repeat_id",
            "raw_gene_id",
            "shap_value",
            "absolute_shap_value"
        ]
    ]
    .copy()
)

patient_repeat_core_long[
    "core_module_name"
] = (
    patient_repeat_core_long[
        "raw_gene_id"
    ]
    .astype(str)
    .map(
        gene_to_core
    )
)


# ============================================================
# 5. SUM GENE ATTRIBUTIONS WITHIN EACH CORE
# ============================================================

observed_core_sums = (
    patient_repeat_core_long
    .groupby(
        [
            "patient_id",
            "repeat_id",
            "core_module_name"
        ],
        as_index=False
    )
    .agg(
        signed_core_attribution=(
            "shap_value",
            "sum"
        ),

        absolute_core_attribution=(
            "absolute_shap_value",
            "sum"
        ),

        n_selected_core_genes=(
            "raw_gene_id",
            "nunique"
        )
    )
)


# ============================================================
# 6. CREATE COMPLETE PATIENT × REPEAT × CORE GRID
# ============================================================

complete_grid = pd.DataFrame(
    product(
        patient_ids,
        repeat_ids,
        core_names
    ),
    columns=[
        "patient_id",
        "repeat_id",
        "core_module_name"
    ]
)

patient_repeat_core = (
    complete_grid.merge(
        observed_core_sums,
        on=[
            "patient_id",
            "repeat_id",
            "core_module_name"
        ],
        how="left",
        validate="one_to_one"
    )
)

for column in [
    "signed_core_attribution",
    "absolute_core_attribution",
    "n_selected_core_genes"
]:

    patient_repeat_core[
        column
    ] = (
        patient_repeat_core[
            column
        ]
        .fillna(0)
    )


# ============================================================
# 7. REPEAT-SPECIFIC EMPIRICAL FUZZY MEMBERSHIP
# ============================================================

# Attribution direction is largely negative globally.
# For fuzzy activation, use empirical magnitude of deviation
# from the repeat-specific cohort centre rather than raw sign.

patient_repeat_core[
    "repeat_core_median"
] = (
    patient_repeat_core
    .groupby(
        [
            "repeat_id",
            "core_module_name"
        ]
    )[
        "signed_core_attribution"
    ]
    .transform(
        "median"
    )
)

patient_repeat_core[
    "centered_core_attribution"
] = (
    patient_repeat_core[
        "signed_core_attribution"
    ]
    -
    patient_repeat_core[
        "repeat_core_median"
    ]
)

# Positive-direction membership:
# percentile of centred signed attribution.
patient_repeat_core[
    "fuzzy_positive_membership"
] = (
    patient_repeat_core
    .groupby(
        [
            "repeat_id",
            "core_module_name"
        ]
    )[
        "centered_core_attribution"
    ]
    .rank(
        method="average",
        pct=True
    )
)

# Negative-direction membership:
# high when attribution is more negative than other patients.
patient_repeat_core[
    "fuzzy_negative_membership"
] = (
    1.0
    -
    patient_repeat_core[
        "fuzzy_positive_membership"
    ]
)

# Magnitude membership:
# high when absolute deviation from cohort centre is large.
patient_repeat_core[
    "absolute_centered_attribution"
] = (
    patient_repeat_core[
        "centered_core_attribution"
    ].abs()
)

patient_repeat_core[
    "fuzzy_magnitude_membership"
] = (
    patient_repeat_core
    .groupby(
        [
            "repeat_id",
            "core_module_name"
        ]
    )[
        "absolute_centered_attribution"
    ]
    .rank(
        method="average",
        pct=True
    )
)


# ============================================================
# 8. REPEAT-LEVEL FUZZY STATE TAXONOMY
# ============================================================

repeat_membership_wide = (
    patient_repeat_core
    .pivot_table(
        index=[
            "patient_id",
            "repeat_id"
        ],
        columns="core_module_name",
        values="fuzzy_magnitude_membership",
        aggfunc="first"
    )
    .reindex(
        columns=core_names
    )
    .reset_index()
)

repeat_state_values = []

for row in repeat_membership_wide.itertuples(
    index=False
):

    memberships = np.asarray(
        [
            getattr(
                row,
                core_name
            )
            for core_name in core_names
        ],
        dtype=float
    )

    repeat_state_values.append(
        classify_fuzzy_state(
            memberships,
            core_names
        )
    )

repeat_membership_wide[
    "repeat_fuzzy_state"
] = repeat_state_values


# ============================================================
# 9. PATIENT-LEVEL CORE MEMBERSHIP SUMMARY
# ============================================================

patient_core_summary = (
    patient_repeat_core
    .groupby(
        [
            "patient_id",
            "core_module_name"
        ],
        as_index=False
    )
    .agg(
        mean_signed_core_attribution=(
            "signed_core_attribution",
            "mean"
        ),

        sd_signed_core_attribution=(
            "signed_core_attribution",
            "std"
        ),

        median_signed_core_attribution=(
            "signed_core_attribution",
            "median"
        ),

        mean_absolute_core_attribution=(
            "absolute_core_attribution",
            "mean"
        ),

        mean_centered_core_attribution=(
            "centered_core_attribution",
            "mean"
        ),

        sd_centered_core_attribution=(
            "centered_core_attribution",
            "std"
        ),

        mean_fuzzy_positive_membership=(
            "fuzzy_positive_membership",
            "mean"
        ),

        sd_fuzzy_positive_membership=(
            "fuzzy_positive_membership",
            "std"
        ),

        mean_fuzzy_negative_membership=(
            "fuzzy_negative_membership",
            "mean"
        ),

        mean_fuzzy_magnitude_membership=(
            "fuzzy_magnitude_membership",
            "mean"
        ),

        sd_fuzzy_magnitude_membership=(
            "fuzzy_magnitude_membership",
            "std"
        ),

        mean_selected_core_genes=(
            "n_selected_core_genes",
            "mean"
        )
    )
)


# ============================================================
# 10. PATIENT-LEVEL FUZZY STATE
# ============================================================

patient_magnitude_membership_wide = (
    patient_core_summary
    .pivot_table(
        index="patient_id",
        columns="core_module_name",
        values="mean_fuzzy_magnitude_membership",
        aggfunc="first"
    )
    .reindex(
        columns=core_names
    )
    .reset_index()
)

patient_state_values = []

for row in patient_magnitude_membership_wide.itertuples(
    index=False
):

    memberships = np.asarray(
        [
            getattr(
                row,
                core_name
            )
            for core_name in core_names
        ],
        dtype=float
    )

    patient_state_values.append(
        classify_fuzzy_state(
            memberships,
            core_names
        )
    )

patient_magnitude_membership_wide[
    "mean_fuzzy_state"
] = patient_state_values


# ============================================================
# 11. REPEAT-LEVEL STATE CONSISTENCY
# ============================================================

repeat_state_consistency = (
    repeat_membership_wide
    .groupby(
        "patient_id",
        as_index=False
    )
    .agg(
        n_repeats=(
            "repeat_id",
            "nunique"
        ),

        n_unique_repeat_states=(
            "repeat_fuzzy_state",
            "nunique"
        ),

        modal_repeat_state=(
            "repeat_fuzzy_state",
            lambda values:
                values.value_counts().index[0]
        ),

        modal_repeat_state_count=(
            "repeat_fuzzy_state",
            lambda values:
                int(
                    values.value_counts().iloc[0]
                )
        )
    )
)

repeat_state_consistency[
    "repeat_state_consistency"
] = (
    repeat_state_consistency[
        "modal_repeat_state_count"
    ]
    /
    repeat_state_consistency[
        "n_repeats"
    ]
)

repeat_state_consistency[
    "fuzzy_state_stability_tier"
] = np.select(
    [
        repeat_state_consistency[
            "repeat_state_consistency"
        ]
        >=
        MIN_STATE_REPEAT_CONSISTENCY,

        repeat_state_consistency[
            "repeat_state_consistency"
        ]
        >=
        0.60
    ],
    [
        "stable",
        "moderately_stable"
    ],
    default="unstable"
)


# ============================================================
# 12. REPEAT-TO-REPEAT CORE RELIABILITY
# ============================================================

repeat_reliability_records = []

for core_name in core_names:

    core_repeat_matrix = (
        patient_repeat_core[
            patient_repeat_core[
                "core_module_name"
            ]
            ==
            core_name
        ]
        .pivot_table(
            index="patient_id",
            columns="repeat_id",
            values="centered_core_attribution",
            aggfunc="first"
        )
        .reindex(
            index=patient_ids,
            columns=repeat_ids
        )
    )

    pairwise_rhos = []

    for repeat_1, repeat_2 in combinations(
        repeat_ids,
        2
    ):

        rho, p_value = spearmanr(
            core_repeat_matrix[
                repeat_1
            ],
            core_repeat_matrix[
                repeat_2
            ],
            nan_policy="omit"
        )

        pairwise_rhos.append(
            rho
        )

        repeat_reliability_records.append({
            "core_module_name":
                core_name,

            "repeat_1":
                repeat_1,

            "repeat_2":
                repeat_2,

            "spearman_rho":
                float(
                    rho
                ),

            "p_value":
                float(
                    p_value
                )
        })


repeat_pairwise_reliability = pd.DataFrame(
    repeat_reliability_records
)

core_reliability_summary = (
    repeat_pairwise_reliability
    .groupby(
        "core_module_name",
        as_index=False
    )
    .agg(
        mean_pairwise_repeat_rho=(
            "spearman_rho",
            "mean"
        ),

        median_pairwise_repeat_rho=(
            "spearman_rho",
            "median"
        ),

        minimum_pairwise_repeat_rho=(
            "spearman_rho",
            "min"
        ),

        maximum_pairwise_repeat_rho=(
            "spearman_rho",
            "max"
        )
    )
)


# ============================================================
# 13. BUILD FINAL PATIENT FUZZY TABLE
# ============================================================

patient_fuzzy_states = (
    patient_magnitude_membership_wide
    .merge(
        repeat_state_consistency,
        on="patient_id",
        how="left",
        validate="one_to_one"
    )
)

# Add positive/negative membership columns for each core.
for core_name in core_names:

    core_patient_summary = (
        patient_core_summary[
            patient_core_summary[
                "core_module_name"
            ]
            ==
            core_name
        ][
            [
                "patient_id",
                "mean_signed_core_attribution",
                "sd_signed_core_attribution",
                "mean_centered_core_attribution",
                "sd_centered_core_attribution",
                "mean_fuzzy_positive_membership",
                "mean_fuzzy_negative_membership",
                "mean_fuzzy_magnitude_membership",
                "sd_fuzzy_magnitude_membership",
                "mean_selected_core_genes"
            ]
        ]
        .copy()
    )

    rename_map = {
        column:
            f"{core_name}__{column}"

        for column in core_patient_summary.columns
        if column != "patient_id"
    }

    core_patient_summary = (
        core_patient_summary.rename(
            columns=rename_map
        )
    )

    patient_fuzzy_states = (
        patient_fuzzy_states.merge(
            core_patient_summary,
            on="patient_id",
            how="left",
            validate="one_to_one"
        )
    )


# ============================================================
# 14. MERGE BBA AND CLINICAL–MOLECULAR STATES
# ============================================================

state_columns = [
    column
    for column in [
        "patient_id",
        "true_label",
        "true_group",
        "integrated_bba_state",
        "clinical_molecular_rank_state",
        "model_dependence_tier",
        "repeat_instability_tier",
        "n_audit_flags",
        "absolute_probability_difference",
        "mean_repeat_instability",
        "excess_signed_residual_mean",
        "excess_coverage_mean",
        "excess_top100_fraction_mean",
        "null_corrected_priority_score"
    ]
    if column in patient_null_corrected.columns
]

patient_fuzzy_states = (
    patient_null_corrected[
        state_columns
    ]
    .merge(
        patient_fuzzy_states,
        on="patient_id",
        how="left",
        validate="one_to_one"
    )
)


# ============================================================
# 15. FUZZY STATE COUNTS
# ============================================================

fuzzy_state_counts = (
    patient_fuzzy_states
    .groupby(
        [
            "mean_fuzzy_state",
            "fuzzy_state_stability_tier"
        ],
        dropna=False,
        as_index=False
    )
    .size()
    .rename(
        columns={
            "size": "n"
        }
    )
)

fuzzy_state_by_clinical_molecular = (
    pd.crosstab(
        patient_fuzzy_states[
            "clinical_molecular_rank_state"
        ],
        patient_fuzzy_states[
            "mean_fuzzy_state"
        ],
        dropna=False
    )
    .reset_index()
)


# ============================================================
# 16. FUZZY MEMBERSHIP–STATE OMNIBUS TESTS
# ============================================================

membership_metrics = []

for core_name in core_names:

    membership_metrics.extend([
        core_name,
        (
            f"{core_name}__"
            "mean_fuzzy_positive_membership"
        ),
        (
            f"{core_name}__"
            "mean_fuzzy_negative_membership"
        ),
        (
            f"{core_name}__"
            "mean_fuzzy_magnitude_membership"
        ),
        (
            f"{core_name}__"
            "sd_fuzzy_magnitude_membership"
        )
    ])

membership_metrics = [
    metric
    for metric in membership_metrics
    if metric in patient_fuzzy_states.columns
]

state_variables = [
    column
    for column in [
        "true_group",
        "integrated_bba_state",
        "clinical_molecular_rank_state",
        "model_dependence_tier",
        "repeat_instability_tier",
        "mean_fuzzy_state",
        "fuzzy_state_stability_tier"
    ]
    if column in patient_fuzzy_states.columns
]

omnibus_records = []

for state_variable in state_variables:

    state_counts = (
        patient_fuzzy_states[
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

    for metric in membership_metrics:

        arrays = []
        valid_states = []

        for state in eligible_states:

            values = (
                patient_fuzzy_states.loc[
                    patient_fuzzy_states[
                        state_variable
                    ]
                    ==
                    state,
                    metric
                ]
                .dropna()
                .to_numpy(dtype=float)
            )

            if len(values) >= MIN_GROUP_N:

                arrays.append(
                    values
                )

                valid_states.append(
                    str(
                        state
                    )
                )

        if len(arrays) < 2:
            continue

        h_statistic, p_value = kruskal(
            *arrays
        )

        omnibus_records.append({
            "state_variable":
                state_variable,

            "metric":
                metric,

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


fuzzy_state_omnibus = pd.DataFrame(
    omnibus_records
)

fuzzy_state_omnibus[
    "fdr_global"
] = benjamini_hochberg(
    fuzzy_state_omnibus[
        "p_value"
    ].to_numpy()
)

fuzzy_state_omnibus[
    "D_minus_log10_fdr"
] = (
    -np.log10(
        fuzzy_state_omnibus[
            "fdr_global"
        ].clip(
            lower=1e-300
        )
    )
)


# ============================================================
# 17. FUZZY STATE PAIRWISE TESTS
# ============================================================

pairwise_records = []

for state_variable in state_variables:

    state_counts = (
        patient_fuzzy_states[
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

        for metric in membership_metrics:

            values_1 = (
                patient_fuzzy_states.loc[
                    patient_fuzzy_states[
                        state_variable
                    ]
                    ==
                    group_1,
                    metric
                ]
                .dropna()
                .to_numpy(dtype=float)
            )

            values_2 = (
                patient_fuzzy_states.loc[
                    patient_fuzzy_states[
                        state_variable
                    ]
                    ==
                    group_2,
                    metric
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

                "metric":
                    metric,

                "group_1":
                    str(
                        group_1
                    ),

                "group_2":
                    str(
                        group_2
                    ),

                "n_group_1":
                    len(
                        values_1
                    ),

                "n_group_2":
                    len(
                        values_2
                    ),

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


fuzzy_state_pairwise = pd.DataFrame(
    pairwise_records
)

fuzzy_state_pairwise[
    "fdr_within_state_metric"
] = np.nan

for (
    state_variable,
    metric
), row_indices in (
    fuzzy_state_pairwise
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

    fuzzy_state_pairwise.loc[
        row_indices,
        "fdr_within_state_metric"
    ] = benjamini_hochberg(
        fuzzy_state_pairwise.loc[
            row_indices,
            "p_value"
        ].to_numpy()
    )

fuzzy_state_pairwise[
    "fdr_global"
] = benjamini_hochberg(
    fuzzy_state_pairwise[
        "p_value"
    ].to_numpy()
)


# ============================================================
# 18. FUZZY MEMBERSHIP CORRELATIONS WITH BBA METRICS
# ============================================================

audit_metrics = [
    column
    for column in [
        "excess_signed_residual_mean",
        "excess_coverage_mean",
        "excess_top100_fraction_mean",
        "absolute_probability_difference",
        "mean_repeat_instability",
        "null_corrected_priority_score"
    ]
    if column in patient_fuzzy_states.columns
]

correlation_records = []

for membership_metric in membership_metrics:

    for audit_metric in audit_metrics:

        rho, p_value = spearmanr(
            patient_fuzzy_states[
                membership_metric
            ],
            patient_fuzzy_states[
                audit_metric
            ],
            nan_policy="omit"
        )

        correlation_records.append({
            "membership_metric":
                membership_metric,

            "audit_metric":
                audit_metric,

            "spearman_rho":
                float(
                    rho
                ),

            "p_value":
                float(
                    p_value
                )
        })


fuzzy_metric_correlations = pd.DataFrame(
    correlation_records
)

fuzzy_metric_correlations[
    "fdr_global"
] = benjamini_hochberg(
    fuzzy_metric_correlations[
        "p_value"
    ].to_numpy()
)


# ============================================================
# 19. PRIORITY FUZZY PATIENTS
# ============================================================

magnitude_columns = [
    (
        f"{core_name}__"
        "mean_fuzzy_magnitude_membership"
    )
    for core_name in core_names
]

uncertainty_columns = [
    (
        f"{core_name}__"
        "sd_fuzzy_magnitude_membership"
    )
    for core_name in core_names
]

patient_fuzzy_states[
    "maximum_core_membership"
] = (
    patient_fuzzy_states[
        magnitude_columns
    ]
    .max(axis=1)
)

patient_fuzzy_states[
    "mean_core_membership_uncertainty"
] = (
    patient_fuzzy_states[
        uncertainty_columns
    ]
    .mean(axis=1)
)

patient_fuzzy_states[
    "fuzzy_explanatory_priority_score"
] = (
    0.40
    * patient_fuzzy_states[
        "maximum_core_membership"
    ]
    +
    0.25
    * (
        1.0
        -
        patient_fuzzy_states[
            "repeat_state_consistency"
        ]
    )
    +
    0.20
    * patient_fuzzy_states[
        "null_corrected_priority_score"
    ].rank(
        method="average",
        pct=True
    )
    +
    0.15
    * patient_fuzzy_states[
        "mean_core_membership_uncertainty"
    ].rank(
        method="average",
        pct=True
    )
)

patient_fuzzy_states = (
    patient_fuzzy_states
    .sort_values(
        [
            "fuzzy_explanatory_priority_score",
            "maximum_core_membership"
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


# ============================================================
# 20. SAVE OUTPUTS
# ============================================================

patient_repeat_core.to_csv(
    FUZZY_MATRIX_DIR
    / "patient_repeat_core_attributions.tsv",
    sep="\t",
    index=False
)

repeat_membership_wide.to_csv(
    FUZZY_MATRIX_DIR
    / "patient_repeat_fuzzy_memberships.tsv",
    sep="\t",
    index=False
)

patient_core_summary.to_csv(
    FUZZY_SUMMARY_DIR
    / "patient_core_membership_summary.tsv",
    sep="\t",
    index=False
)

repeat_pairwise_reliability.to_csv(
    FUZZY_SUMMARY_DIR
    / "core_repeat_pairwise_reliability.tsv",
    sep="\t",
    index=False
)

core_reliability_summary.to_csv(
    FUZZY_SUMMARY_DIR
    / "core_repeat_reliability_summary.tsv",
    sep="\t",
    index=False
)

repeat_state_consistency.to_csv(
    FUZZY_SUMMARY_DIR
    / "patient_fuzzy_state_repeat_consistency.tsv",
    sep="\t",
    index=False
)

fuzzy_state_counts.to_csv(
    FUZZY_SUMMARY_DIR
    / "fuzzy_explanatory_state_counts.tsv",
    sep="\t",
    index=False
)

fuzzy_state_by_clinical_molecular.to_csv(
    FUZZY_SUMMARY_DIR
    / "fuzzy_state_by_clinical_molecular_state.tsv",
    sep="\t",
    index=False
)

fuzzy_state_omnibus.to_csv(
    FUZZY_SUMMARY_DIR
    / "fuzzy_state_omnibus_tests.tsv",
    sep="\t",
    index=False
)

fuzzy_state_pairwise.to_csv(
    FUZZY_SUMMARY_DIR
    / "fuzzy_state_pairwise_tests.tsv",
    sep="\t",
    index=False
)

fuzzy_metric_correlations.to_csv(
    FUZZY_SUMMARY_DIR
    / "fuzzy_membership_audit_metric_correlations.tsv",
    sep="\t",
    index=False
)

patient_fuzzy_states.to_csv(
    FUZZY_SUMMARY_DIR
    / "patient_fuzzy_explanatory_states.tsv",
    sep="\t",
    index=False
)

patient_fuzzy_states.head(
    100
).to_csv(
    FUZZY_SUMMARY_DIR
    / "patient_fuzzy_explanatory_states_top100.tsv",
    sep="\t",
    index=False
)


# ============================================================
# 21. SUMMARY
# ============================================================

summary_table = pd.DataFrame([
    {
        "metric":
            "n_patients",

        "value":
            len(
                patient_fuzzy_states
            )
    },
    {
        "metric":
            "n_stable_cores",

        "value":
            len(
                core_names
            )
    },
    {
        "metric":
            "stable_fuzzy_state_patients",

        "value":
            int(
                (
                    patient_fuzzy_states[
                        "fuzzy_state_stability_tier"
                    ]
                    ==
                    "stable"
                ).sum()
            )
    },
    {
        "metric":
            "moderately_stable_fuzzy_state_patients",

        "value":
            int(
                (
                    patient_fuzzy_states[
                        "fuzzy_state_stability_tier"
                    ]
                    ==
                    "moderately_stable"
                ).sum()
            )
    },
    {
        "metric":
            "unstable_fuzzy_state_patients",

        "value":
            int(
                (
                    patient_fuzzy_states[
                        "fuzzy_state_stability_tier"
                    ]
                    ==
                    "unstable"
                ).sum()
            )
    },
    {
        "metric":
            "significant_fuzzy_omnibus_tests_fdr_0_05",

        "value":
            int(
                (
                    fuzzy_state_omnibus[
                        "fdr_global"
                    ]
                    <= 0.05
                ).sum()
            )
    },
    {
        "metric":
            "significant_fuzzy_pairwise_tests_fdr_0_05",

        "value":
            int(
                (
                    fuzzy_state_pairwise[
                        "fdr_within_state_metric"
                    ]
                    <= 0.05
                ).sum()
            )
    },
    {
        "metric":
            "medium_or_large_fuzzy_pairwise_effects",

        "value":
            int(
                fuzzy_state_pairwise[
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
    FUZZY_SUMMARY_DIR
    / "fuzzy_explanatory_state_summary.tsv",
    sep="\t",
    index=False
)


manifest = {
    "analysis":
        "Repeated fuzzy explanatory-state reconstruction",

    "run_directory":
        str(
            RUN_DIR
        ),

    "stable_cores":
        core_names,

    "n_repeats":
        N_EXPECTED_REPEATS,

    "membership_definition":
        (
            "Repeat-specific empirical percentile of "
            "absolute centred stable-core attribution."
        ),

    "high_membership_threshold":
        HIGH_MEMBERSHIP_THRESHOLD,

    "low_membership_threshold":
        LOW_MEMBERSHIP_THRESHOLD,

    "dominance_margin":
        DOMINANCE_MARGIN,

    "stable_repeat_consistency_threshold":
        MIN_STATE_REPEAT_CONSISTENCY,

    "interpretation_boundary":
        (
            "Fuzzy states describe explanatory-attribution "
            "geometry and uncertainty. They are not claimed "
            "as biological subtypes or disease ontologies."
        )
}

with open(
    FUZZY_DIR
    / "fuzzy_explanatory_state_manifest.json",
    "w",
    encoding="utf-8"
) as handle:

    json.dump(
        manifest,
        handle,
        indent=2
    )


# ============================================================
# 22. FINAL REPORT
# ============================================================

print("\n" + "=" * 80)
print("CELL 27 COMPLETED")
print("=" * 80)

display(
    summary_table
)

print("\nCore repeat reliability:")

display(
    core_reliability_summary
)

print("\nFuzzy explanatory-state counts:")

display(
    fuzzy_state_counts
)

print("\nFuzzy states by clinical–molecular state:")

display(
    fuzzy_state_by_clinical_molecular
)

print("\nStrongest fuzzy state associations:")

fuzzy_pairwise_sorted = (
    fuzzy_state_pairwise
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
    fuzzy_pairwise_sorted[
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
            "effect_size",
            "p_value",
            "fdr_within_state_metric",
            "fdr_global"
        ]
    ].head(40)
)

print("\nStrongest fuzzy-membership audit correlations:")

display(
    fuzzy_metric_correlations
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
    .head(40)
)

print("\nTop fuzzy explanatory-priority patients:")

priority_columns = [
    column
    for column in [
        "patient_id",
        "true_group",
        "integrated_bba_state",
        "clinical_molecular_rank_state",
        "model_dependence_tier",
        "repeat_instability_tier",
        "mean_fuzzy_state",
        "modal_repeat_state",
        "repeat_state_consistency",
        "fuzzy_state_stability_tier",
        "GapCore_01",
        "GapCore_02",
        "GapCore_03",
        "maximum_core_membership",
        "mean_core_membership_uncertainty",
        "null_corrected_priority_score",
        "fuzzy_explanatory_priority_score"
    ]
    if column in patient_fuzzy_states.columns
]

display(
    patient_fuzzy_states[
        priority_columns
    ].head(40)
)

print("\nOutput directory:")
print(FUZZY_DIR)