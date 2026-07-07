# ============================================================
# AIDO-BBA BRCA 1.0
# CELL 21
# Residual-direction decomposition, pairwise effect size,
# post-hoc FDR, and candidate residual-state prioritization
# ============================================================

from pathlib import Path
from itertools import combinations
from scipy.stats import (
    kruskal,
    mannwhitneyu,
    spearmanr
)

import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")


# ============================================================
# 0. SETTINGS
# ============================================================

OUTPUT_ROOT = Path(
    r"D:\AIDO-Temp\AIDO_BBA_BRCA_1_0"
)

NEAR_ZERO_RESIDUAL_THRESHOLD = 0.01

# A residual direction is considered stable when at least
# this fraction of repeats shares the same sign.
DIRECTION_STABILITY_THRESHOLD = 0.80

# Minimum group size for formal pairwise testing.
MIN_GROUP_N = 10

# Main metrics to test.
CANDIDATE_METRICS = [
    "mean_attribution_mass_coverage",
    "mean_unmapped_absolute_shap",
    "mean_unmapped_signed_residual",
    "absolute_mean_unmapped_signed_residual",
    "residual_direction_consistency",
    "residual_repeat_sd",
    "top10_bp_mass_fraction_mean",
    "top20_bp_mass_fraction_mean",
    "top50_bp_mass_fraction_mean",
    "top100_bp_mass_fraction_mean"
]

STATE_VARIABLES = [
    "true_group",
    "integrated_bba_state",
    "clinical_molecular_rank_state",
    "model_dependence_tier",
    "repeat_instability_tier"
]


# ============================================================
# 1. FIND COMPLETED RUN
# ============================================================

candidate_runs = []

if not OUTPUT_ROOT.exists():

    raise FileNotFoundError(
        f"Output root not found:\n{OUTPUT_ROOT}"
    )

for run_dir in OUTPUT_ROOT.iterdir():

    if not run_dir.is_dir():
        continue

    completeness_file = (
        run_dir
        / "06_bp_reconstruction"
        / "summaries"
        / "patient_completeness_all_repeats.tsv"
    )

    state_file = (
        run_dir
        / "04_blackbox"
        / "bba_patient_state_taxonomy.tsv"
    )

    if (
        completeness_file.exists()
        and state_file.exists()
    ):

        candidate_runs.append(
            run_dir
        )

if len(candidate_runs) == 0:

    raise FileNotFoundError(
        "No completed AIDO-BBA run with completeness "
        "and patient-state files was found."
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

COMPLETENESS_AUDIT_DIR = (
    RUN_DIR
    / "07_completeness_audit"
)

RESIDUAL_AUDIT_DIR = (
    RUN_DIR
    / "08_residual_direction_audit"
)

RESIDUAL_AUDIT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

print("=" * 80)
print("AIDO-BBA RESIDUAL-DIRECTION AND POST-HOC AUDIT")
print("=" * 80)

print("\nRun directory:")
print(RUN_DIR)


# ============================================================
# 2. LOAD DATA
# ============================================================

patient_repeat = pd.read_csv(
    BP_SUMMARY_DIR
    / "patient_completeness_all_repeats.tsv",
    sep="\t"
)

patient_states = pd.read_csv(
    BLACKBOX_DIR
    / "bba_patient_state_taxonomy.tsv",
    sep="\t"
)

print("\nPatient-repeat rows:")
print(patient_repeat.shape)

print("Patient-state rows:")
print(patient_states.shape)

if len(patient_repeat) != 5365:

    print(
        "WARNING: expected 5,365 patient-repeat rows, found",
        len(patient_repeat)
    )

if patient_repeat["patient_id"].nunique() != 1073:

    raise ValueError(
        "Expected 1,073 unique patients in repeat-level data."
    )


# ============================================================
# 3. PATIENT-LEVEL RESIDUAL DIRECTION DECOMPOSITION
# ============================================================

def sign_fraction_positive(values):

    values = np.asarray(
        values,
        dtype=float
    )

    valid = values[
        np.isfinite(values)
    ]

    if len(valid) == 0:
        return np.nan

    return float(
        np.mean(
            valid > 0
        )
    )


def sign_fraction_negative(values):

    values = np.asarray(
        values,
        dtype=float
    )

    valid = values[
        np.isfinite(values)
    ]

    if len(valid) == 0:
        return np.nan

    return float(
        np.mean(
            valid < 0
        )
    )


def sign_change_count(values):

    values = np.asarray(
        values,
        dtype=float
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) <= 1:
        return 0

    signs = np.sign(
        values
    )

    signs = signs[
        signs != 0
    ]

    if len(signs) <= 1:
        return 0

    return int(
        np.sum(
            signs[1:] != signs[:-1]
        )
    )


patient_residual_summary = (
    patient_repeat
    .sort_values(
        [
            "patient_id",
            "repeat_id"
        ]
    )
    .groupby(
        "patient_id",
        as_index=False
    )
    .agg(
        true_label=(
            "true_label",
            "first"
        ),

        true_group=(
            "true_group",
            "first"
        ),

        n_repeats=(
            "repeat_id",
            "nunique"
        ),

        mean_probability_advanced=(
            "predicted_probability_advanced",
            "mean"
        ),

        sd_probability_advanced=(
            "predicted_probability_advanced",
            "std"
        ),

        mean_attribution_mass_coverage=(
            "attribution_mass_coverage",
            "mean"
        ),

        sd_attribution_mass_coverage=(
            "attribution_mass_coverage",
            "std"
        ),

        mean_unmapped_absolute_shap=(
            "unmapped_absolute_gene_shap",
            "mean"
        ),

        sd_unmapped_absolute_shap=(
            "unmapped_absolute_gene_shap",
            "std"
        ),

        mean_unmapped_signed_residual=(
            "unmapped_signed_residual",
            "mean"
        ),

        median_unmapped_signed_residual=(
            "unmapped_signed_residual",
            "median"
        ),

        residual_repeat_sd=(
            "unmapped_signed_residual",
            "std"
        ),

        residual_repeat_minimum=(
            "unmapped_signed_residual",
            "min"
        ),

        residual_repeat_maximum=(
            "unmapped_signed_residual",
            "max"
        ),

        fraction_positive_residual=(
            "unmapped_signed_residual",
            sign_fraction_positive
        ),

        fraction_negative_residual=(
            "unmapped_signed_residual",
            sign_fraction_negative
        ),

        residual_sign_changes=(
            "unmapped_signed_residual",
            sign_change_count
        ),

        top10_bp_mass_fraction_mean=(
            "top10_bp_mass_fraction",
            "mean"
        ),

        top20_bp_mass_fraction_mean=(
            "top20_bp_mass_fraction",
            "mean"
        ),

        top50_bp_mass_fraction_mean=(
            "top50_bp_mass_fraction",
            "mean"
        ),

        top100_bp_mass_fraction_mean=(
            "top100_bp_mass_fraction",
            "mean"
        )
    )
)

patient_residual_summary[
    "absolute_mean_unmapped_signed_residual"
] = (
    patient_residual_summary[
        "mean_unmapped_signed_residual"
    ].abs()
)

patient_residual_summary[
    "residual_direction_consistency"
] = (
    patient_residual_summary[
        [
            "fraction_positive_residual",
            "fraction_negative_residual"
        ]
    ]
    .max(axis=1)
)

patient_residual_summary[
    "dominant_residual_direction"
] = np.select(
    [
        patient_residual_summary[
            "fraction_positive_residual"
        ]
        >
        patient_residual_summary[
            "fraction_negative_residual"
        ],

        patient_residual_summary[
            "fraction_negative_residual"
        ]
        >
        patient_residual_summary[
            "fraction_positive_residual"
        ]
    ],
    [
        "toward_advanced",
        "toward_early"
    ],
    default="mixed_or_zero"
)


# ============================================================
# 4. RESIDUAL DIRECTION TAXONOMY
# ============================================================

patient_residual_summary[
    "residual_direction_state"
] = np.select(
    [
        patient_residual_summary[
            "absolute_mean_unmapped_signed_residual"
        ]
        <=
        NEAR_ZERO_RESIDUAL_THRESHOLD,

        (
            patient_residual_summary[
                "fraction_positive_residual"
            ]
            >=
            DIRECTION_STABILITY_THRESHOLD
        ),

        (
            patient_residual_summary[
                "fraction_negative_residual"
            ]
            >=
            DIRECTION_STABILITY_THRESHOLD
        )
    ],
    [
        "near_zero_residual",
        "stable_toward_advanced",
        "stable_toward_early"
    ],
    default="directionally_mixed"
)

patient_residual_summary[
    "residual_direction_matches_clinical_stage"
] = np.select(
    [
        (
            patient_residual_summary[
                "true_label"
            ] == 1
        )
        &
        (
            patient_residual_summary[
                "residual_direction_state"
            ] == "stable_toward_advanced"
        ),

        (
            patient_residual_summary[
                "true_label"
            ] == 0
        )
        &
        (
            patient_residual_summary[
                "residual_direction_state"
            ] == "stable_toward_early"
        ),

        patient_residual_summary[
            "residual_direction_state"
        ].isin([
            "near_zero_residual",
            "directionally_mixed"
        ])
    ],
    [
        "direction_concordant",
        "direction_concordant",
        "direction_unresolved"
    ],
    default="direction_discordant"
)


# ============================================================
# 5. MERGE EXISTING BBA STATES
# ============================================================

state_columns_to_merge = [
    column
    for column in [
        "patient_id",
        "integrated_bba_state",
        "clinical_molecular_rank_state",
        "model_dependence_tier",
        "repeat_instability_tier",
        "n_audit_flags",
        "absolute_probability_difference",
        "mean_repeat_instability",
        "extratrees_probability_advanced",
        "elasticnet_probability_advanced"
    ]
    if column in patient_states.columns
]

patient_residual_states = (
    patient_residual_summary.merge(
        patient_states[
            state_columns_to_merge
        ],
        on="patient_id",
        how="left",
        validate="one_to_one"
    )
)

print("\nResidual-direction states:")

residual_direction_counts = (
    patient_residual_states
    .groupby(
        [
            "true_group",
            "residual_direction_state"
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

display(
    residual_direction_counts
)

print("\nClinical-stage direction agreement:")

clinical_direction_counts = (
    patient_residual_states
    .groupby(
        [
            "true_group",
            "residual_direction_matches_clinical_stage"
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

display(
    clinical_direction_counts
)


# ============================================================
# 6. EFFECT-SIZE FUNCTIONS
# ============================================================

def cliffs_delta_from_u(
    u_statistic,
    n_group_1,
    n_group_2
):
    """
    Cliff's delta based on the Mann-Whitney U statistic.

    Positive:
        Group 1 tends to have higher values.

    Negative:
        Group 2 tends to have higher values.
    """

    if (
        n_group_1 == 0
        or n_group_2 == 0
    ):
        return np.nan

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


def cliffs_delta_magnitude(delta):

    if pd.isna(delta):
        return np.nan

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


def benjamini_hochberg(p_values):
    """
    Benjamini-Hochberg FDR correction.
    """

    p_values = np.asarray(
        p_values,
        dtype=float
    )

    n_tests = len(
        p_values
    )

    adjusted = np.full(
        n_tests,
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

    valid_adjusted = np.empty(
        len(valid_p),
        dtype=float
    )

    valid_adjusted[
        order
    ] = ordered_adjusted

    adjusted[
        valid_indices
    ] = valid_adjusted

    return adjusted


# ============================================================
# 7. SELECT AVAILABLE METRICS AND STATES
# ============================================================

available_metrics = [
    metric
    for metric in CANDIDATE_METRICS
    if metric in patient_residual_states.columns
]

available_state_variables = [
    state_variable
    for state_variable in STATE_VARIABLES
    if state_variable in patient_residual_states.columns
]

print("\nMetrics available for testing:")
print(available_metrics)

print("\nState variables available for testing:")
print(available_state_variables)


# ============================================================
# 8. OMNIBUS KRUSKAL-WALLIS TESTS
# ============================================================

omnibus_records = []

for state_variable in available_state_variables:

    state_values = (
        patient_residual_states[
            state_variable
        ]
        .dropna()
        .unique()
        .tolist()
    )

    for metric in available_metrics:

        arrays = []
        valid_states = []

        for state_value in state_values:

            values = (
                patient_residual_states.loc[
                    patient_residual_states[
                        state_variable
                    ] == state_value,
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
                    str(state_value)
                )

        if len(arrays) < 2:
            continue

        statistic, p_value = kruskal(
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

omnibus_tests = pd.DataFrame(
    omnibus_records
)

if len(omnibus_tests) > 0:

    omnibus_tests[
        "fdr_bh"
    ] = benjamini_hochberg(
        omnibus_tests[
            "p_value"
        ].to_numpy()
    )

    omnibus_tests[
        "D_minus_log10_fdr"
    ] = (
        -np.log10(
            omnibus_tests[
                "fdr_bh"
            ].clip(
                lower=1e-300
            )
        )
    )


# ============================================================
# 9. ALL PAIRWISE POST-HOC TESTS
# ============================================================

pairwise_records = []

for state_variable in available_state_variables:

    state_counts = (
        patient_residual_states[
            state_variable
        ]
        .value_counts(
            dropna=True
        )
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

        for metric in available_metrics:

            values_1 = (
                patient_residual_states.loc[
                    patient_residual_states[
                        state_variable
                    ] == group_1,
                    metric
                ]
                .dropna()
                .to_numpy(dtype=float)
            )

            values_2 = (
                patient_residual_states.loc[
                    patient_residual_states[
                        state_variable
                    ] == group_2,
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
                alternative="two-sided",
                method="auto"
            )

            delta = cliffs_delta_from_u(
                u_statistic=u_statistic,
                n_group_1=len(values_1),
                n_group_2=len(values_2)
            )

            pairwise_records.append({
                "state_variable":
                    state_variable,

                "metric":
                    metric,

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

                "mean_difference_group1_minus_group2":
                    float(
                        np.mean(
                            values_1
                        )
                        -
                        np.mean(
                            values_2
                        )
                    ),

                "median_difference_group1_minus_group2":
                    float(
                        np.median(
                            values_1
                        )
                        -
                        np.median(
                            values_2
                        )
                    ),

                "mann_whitney_u":
                    float(
                        u_statistic
                    ),

                "p_value":
                    float(
                        p_value
                    ),

                "cliffs_delta":
                    delta,

                "cliffs_delta_absolute":
                    abs(
                        delta
                    ),

                "cliffs_delta_magnitude":
                    cliffs_delta_magnitude(
                        delta
                    )
            })

pairwise_tests = pd.DataFrame(
    pairwise_records
)

if len(pairwise_tests) > 0:

    # FDR within each state-variable × metric family.
    pairwise_tests[
        "fdr_within_metric"
    ] = np.nan

    for (
        state_variable,
        metric
    ), index_values in (
        pairwise_tests
        .groupby(
            [
                "state_variable",
                "metric"
            ]
        )
        .groups
        .items()
    ):

        index_values = list(
            index_values
        )

        pairwise_tests.loc[
            index_values,
            "fdr_within_metric"
        ] = benjamini_hochberg(
            pairwise_tests.loc[
                index_values,
                "p_value"
            ].to_numpy()
        )

    # Global FDR across all pairwise tests.
    pairwise_tests[
        "fdr_global"
    ] = benjamini_hochberg(
        pairwise_tests[
            "p_value"
        ].to_numpy()
    )

    pairwise_tests[
        "D_minus_log10_p"
    ] = (
        -np.log10(
            pairwise_tests[
                "p_value"
            ].clip(
                lower=1e-300
            )
        )
    )

    pairwise_tests[
        "D_minus_log10_fdr_within_metric"
    ] = (
        -np.log10(
            pairwise_tests[
                "fdr_within_metric"
            ].clip(
                lower=1e-300
            )
        )
    )

    pairwise_tests[
        "direction"
    ] = np.where(
        pairwise_tests[
            "cliffs_delta"
        ] > 0,
        "group_1_higher",
        np.where(
            pairwise_tests[
                "cliffs_delta"
            ] < 0,
            "group_2_higher",
            "no_direction"
        )
    )


# ============================================================
# 10. STATE DESCRIPTIVE SUMMARIES
# ============================================================

state_summary_records = []

for state_variable in available_state_variables:

    for state_value, state_df in (
        patient_residual_states
        .groupby(
            state_variable,
            dropna=False
        )
    ):

        for metric in available_metrics:

            values = (
                state_df[
                    metric
                ]
                .dropna()
                .to_numpy(dtype=float)
            )

            if len(values) == 0:
                continue

            state_summary_records.append({
                "state_variable":
                    state_variable,

                "state_value":
                    str(state_value),

                "metric":
                    metric,

                "n":
                    len(values),

                "mean":
                    float(
                        np.mean(
                            values
                        )
                    ),

                "sd":
                    float(
                        np.std(
                            values,
                            ddof=1
                        )
                    )
                    if len(values) > 1
                    else np.nan,

                "median":
                    float(
                        np.median(
                            values
                        )
                    ),

                "q25":
                    float(
                        np.quantile(
                            values,
                            0.25
                        )
                    ),

                "q75":
                    float(
                        np.quantile(
                            values,
                            0.75
                        )
                    ),

                "minimum":
                    float(
                        np.min(
                            values
                        )
                    ),

                "maximum":
                    float(
                        np.max(
                            values
                        )
                    )
            })

state_descriptive_summary = pd.DataFrame(
    state_summary_records
)


# ============================================================
# 11. SPECIFIC HIGH-VALUE CONTRASTS
# ============================================================

priority_contrasts = [
    (
        "integrated_bba_state",
        "model_dependent",
        "clinical_molecular_concordant"
    ),
    (
        "integrated_bba_state",
        "resampling_unstable",
        "clinical_molecular_concordant"
    ),
    (
        "integrated_bba_state",
        "clinical_early_molecular_advanced_like",
        "clinical_molecular_concordant"
    ),
    (
        "integrated_bba_state",
        "clinical_advanced_molecular_early_like",
        "clinical_molecular_concordant"
    ),
    (
        "model_dependence_tier",
        "high_model_dependence",
        "low_model_dependence"
    ),
    (
        "repeat_instability_tier",
        "high_repeat_instability",
        "low_repeat_instability"
    ),
    (
        "clinical_molecular_rank_state",
        "clinical_early_molecular_advanced_like",
        "early_rank_concordant"
    ),
    (
        "clinical_molecular_rank_state",
        "clinical_advanced_molecular_early_like",
        "advanced_rank_concordant"
    )
]

priority_pairwise_parts = []

for (
    state_variable,
    group_1,
    group_2
) in priority_contrasts:

    if state_variable not in pairwise_tests[
        "state_variable"
    ].unique():
        continue

    direct_match = pairwise_tests[
        (
            pairwise_tests[
                "state_variable"
            ] == state_variable
        )
        &
        (
            pairwise_tests[
                "group_1"
            ] == group_1
        )
        &
        (
            pairwise_tests[
                "group_2"
            ] == group_2
        )
    ].copy()

    reverse_match = pairwise_tests[
        (
            pairwise_tests[
                "state_variable"
            ] == state_variable
        )
        &
        (
            pairwise_tests[
                "group_1"
            ] == group_2
        )
        &
        (
            pairwise_tests[
                "group_2"
            ] == group_1
        )
    ].copy()

    if len(reverse_match) > 0:

        reverse_match[
            "group_1"
        ] = group_1

        reverse_match[
            "group_2"
        ] = group_2

        reverse_match[
            "mean_difference_group1_minus_group2"
        ] *= -1

        reverse_match[
            "median_difference_group1_minus_group2"
        ] *= -1

        reverse_match[
            "cliffs_delta"
        ] *= -1

        reverse_match[
            "direction"
        ] = np.where(
            reverse_match[
                "cliffs_delta"
            ] > 0,
            "group_1_higher",
            np.where(
                reverse_match[
                    "cliffs_delta"
                ] < 0,
                "group_2_higher",
                "no_direction"
            )
        )

        priority_pairwise_parts.append(
            reverse_match
        )

    if len(direct_match) > 0:

        priority_pairwise_parts.append(
            direct_match
        )

if len(priority_pairwise_parts) > 0:

    priority_pairwise_tests = pd.concat(
        priority_pairwise_parts,
        ignore_index=True
    )

    priority_pairwise_tests = (
        priority_pairwise_tests
        .sort_values(
            [
                "fdr_within_metric",
                "cliffs_delta_absolute"
            ],
            ascending=[
                True,
                False
            ]
        )
    )

else:

    priority_pairwise_tests = pd.DataFrame()


# ============================================================
# 12. CANDIDATE RESIDUAL-STATE PRIORITY SCORE
# ============================================================

def percentile_high(values):

    return values.rank(
        method="average",
        pct=True
    )


def percentile_low(values):

    return (
        1.0
        -
        values.rank(
            method="average",
            pct=True
        )
    )


patient_residual_states[
    "unmapped_absolute_percentile"
] = percentile_high(
    patient_residual_states[
        "mean_unmapped_absolute_shap"
    ]
)

patient_residual_states[
    "low_coverage_percentile"
] = percentile_low(
    patient_residual_states[
        "mean_attribution_mass_coverage"
    ]
)

patient_residual_states[
    "direction_consistency_percentile"
] = percentile_high(
    patient_residual_states[
        "residual_direction_consistency"
    ]
)

patient_residual_states[
    "residual_instability_percentile"
] = percentile_high(
    patient_residual_states[
        "residual_repeat_sd"
    ]
)

if (
    "absolute_probability_difference"
    in patient_residual_states.columns
):

    patient_residual_states[
        "model_dependence_percentile"
    ] = percentile_high(
        patient_residual_states[
            "absolute_probability_difference"
        ]
    )

else:

    patient_residual_states[
        "model_dependence_percentile"
    ] = 0.0


patient_residual_states[
    "candidate_residual_state_score"
] = (
    0.25
    * patient_residual_states[
        "unmapped_absolute_percentile"
    ]
    +
    0.20
    * patient_residual_states[
        "low_coverage_percentile"
    ]
    +
    0.20
    * patient_residual_states[
        "direction_consistency_percentile"
    ]
    +
    0.15
    * patient_residual_states[
        "residual_instability_percentile"
    ]
    +
    0.20
    * patient_residual_states[
        "model_dependence_percentile"
    ]
)

patient_residual_states[
    "candidate_residual_state_tier"
] = pd.cut(
    patient_residual_states[
        "candidate_residual_state_score"
    ],
    bins=[
        -np.inf,
        patient_residual_states[
            "candidate_residual_state_score"
        ].quantile(0.75),
        patient_residual_states[
            "candidate_residual_state_score"
        ].quantile(0.90),
        np.inf
    ],
    labels=[
        "background",
        "candidate",
        "high_priority_candidate"
    ],
    include_lowest=True
)

candidate_residual_patients = (
    patient_residual_states
    .sort_values(
        [
            "candidate_residual_state_score",
            "mean_unmapped_absolute_shap",
            "residual_direction_consistency"
        ],
        ascending=[
            False,
            False,
            False
        ]
    )
    .reset_index(drop=True)
)


# ============================================================
# 13. RESIDUAL DIRECTION × BBA STATE CROSS-TABLES
# ============================================================

cross_tab_records = []

for state_variable in available_state_variables:

    cross_table = pd.crosstab(
        patient_residual_states[
            state_variable
        ],
        patient_residual_states[
            "residual_direction_state"
        ],
        dropna=False
    )

    cross_table = (
        cross_table
        .reset_index()
        .melt(
            id_vars=[
                state_variable
            ],
            var_name=(
                "residual_direction_state"
            ),
            value_name="n"
        )
        .rename(
            columns={
                state_variable:
                    "state_value"
            }
        )
    )

    cross_table.insert(
        0,
        "state_variable",
        state_variable
    )

    cross_tab_records.append(
        cross_table
    )

residual_direction_by_state = pd.concat(
    cross_tab_records,
    ignore_index=True
)


# ============================================================
# 14. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 80)
print("OMNIBUS TESTS")
print("=" * 80)

display(
    omnibus_tests
    .sort_values(
        [
            "fdr_bh",
            "D_minus_log10_p"
        ],
        ascending=[
            True,
            False
        ]
    )
    .head(30)
)

print("\n" + "=" * 80)
print("STRONGEST PAIRWISE EFFECTS")
print("=" * 80)

display(
    pairwise_tests[
        [
            "state_variable",
            "metric",
            "group_1",
            "group_2",
            "n_group_1",
            "n_group_2",
            "mean_group_1",
            "mean_group_2",
            "mean_difference_group1_minus_group2",
            "cliffs_delta",
            "cliffs_delta_magnitude",
            "p_value",
            "fdr_within_metric",
            "fdr_global"
        ]
    ]
    .sort_values(
        [
            "fdr_within_metric",
            "cliffs_delta_absolute"
        ],
        ascending=[
            True,
            False
        ]
    )
    .head(40)
)

print("\n" + "=" * 80)
print("PRIORITY CONTRASTS")
print("=" * 80)

if len(priority_pairwise_tests) > 0:

    display(
        priority_pairwise_tests[
            [
                "state_variable",
                "metric",
                "group_1",
                "group_2",
                "mean_group_1",
                "mean_group_2",
                "mean_difference_group1_minus_group2",
                "cliffs_delta",
                "cliffs_delta_magnitude",
                "p_value",
                "fdr_within_metric"
            ]
        ].head(50)
    )

else:

    print("No priority contrasts were available.")


print("\n" + "=" * 80)
print("TOP CANDIDATE RESIDUAL-STATE PATIENTS")
print("=" * 80)

candidate_display_columns = [
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
        "residual_repeat_sd",
        "fraction_positive_residual",
        "fraction_negative_residual",
        "residual_direction_consistency",
        "residual_direction_state",
        "residual_direction_matches_clinical_stage",
        "candidate_residual_state_score",
        "candidate_residual_state_tier"
    ]
    if column in candidate_residual_patients.columns
]

display(
    candidate_residual_patients[
        candidate_display_columns
    ].head(40)
)


# ============================================================
# 15. SAVE OUTPUTS
# ============================================================

patient_residual_states.to_csv(
    RESIDUAL_AUDIT_DIR
    / "patient_residual_direction_states.tsv",
    sep="\t",
    index=False
)

residual_direction_counts.to_csv(
    RESIDUAL_AUDIT_DIR
    / "residual_direction_state_counts.tsv",
    sep="\t",
    index=False
)

clinical_direction_counts.to_csv(
    RESIDUAL_AUDIT_DIR
    / "clinical_residual_direction_agreement_counts.tsv",
    sep="\t",
    index=False
)

residual_direction_by_state.to_csv(
    RESIDUAL_AUDIT_DIR
    / "residual_direction_by_bba_state.tsv",
    sep="\t",
    index=False
)

state_descriptive_summary.to_csv(
    RESIDUAL_AUDIT_DIR
    / "state_metric_descriptive_summary.tsv",
    sep="\t",
    index=False
)

omnibus_tests.to_csv(
    RESIDUAL_AUDIT_DIR
    / "residual_state_omnibus_kruskal_tests.tsv",
    sep="\t",
    index=False
)

pairwise_tests.to_csv(
    RESIDUAL_AUDIT_DIR
    / "residual_state_pairwise_mannwhitney_cliffs_delta.tsv",
    sep="\t",
    index=False
)

priority_pairwise_tests.to_csv(
    RESIDUAL_AUDIT_DIR
    / "priority_state_contrasts.tsv",
    sep="\t",
    index=False
)

candidate_residual_patients.to_csv(
    RESIDUAL_AUDIT_DIR
    / "candidate_residual_state_patients.tsv",
    sep="\t",
    index=False
)

candidate_residual_patients.head(
    100
).to_csv(
    RESIDUAL_AUDIT_DIR
    / "candidate_residual_state_patients_top100.tsv",
    sep="\t",
    index=False
)


# ============================================================
# 16. RUN SUMMARY
# ============================================================

run_summary = pd.DataFrame([
    {
        "metric":
            "n_patients",

        "value":
            len(
                patient_residual_states
            )
    },
    {
        "metric":
            "near_zero_residual_patients",

        "value":
            int(
                (
                    patient_residual_states[
                        "residual_direction_state"
                    ]
                    == "near_zero_residual"
                ).sum()
            )
    },
    {
        "metric":
            "stable_toward_advanced_patients",

        "value":
            int(
                (
                    patient_residual_states[
                        "residual_direction_state"
                    ]
                    == "stable_toward_advanced"
                ).sum()
            )
    },
    {
        "metric":
            "stable_toward_early_patients",

        "value":
            int(
                (
                    patient_residual_states[
                        "residual_direction_state"
                    ]
                    == "stable_toward_early"
                ).sum()
            )
    },
    {
        "metric":
            "directionally_mixed_patients",

        "value":
            int(
                (
                    patient_residual_states[
                        "residual_direction_state"
                    ]
                    == "directionally_mixed"
                ).sum()
            )
    },
    {
        "metric":
            "direction_discordant_patients",

        "value":
            int(
                (
                    patient_residual_states[
                        "residual_direction_matches_clinical_stage"
                    ]
                    == "direction_discordant"
                ).sum()
            )
    },
    {
        "metric":
            "high_priority_candidate_patients",

        "value":
            int(
                (
                    patient_residual_states[
                        "candidate_residual_state_tier"
                    ]
                    == "high_priority_candidate"
                ).sum()
            )
    },
    {
        "metric":
            "significant_omnibus_tests_fdr_0_05",

        "value":
            int(
                (
                    omnibus_tests[
                        "fdr_bh"
                    ] <= 0.05
                ).sum()
            )
            if len(omnibus_tests) > 0
            else 0
    },
    {
        "metric":
            "significant_pairwise_tests_fdr_0_05",

        "value":
            int(
                (
                    pairwise_tests[
                        "fdr_within_metric"
                    ] <= 0.05
                ).sum()
            )
            if len(pairwise_tests) > 0
            else 0
    },
    {
        "metric":
            "medium_or_large_pairwise_effects",

        "value":
            int(
                pairwise_tests[
                    "cliffs_delta_magnitude"
                ].isin([
                    "medium",
                    "large"
                ]).sum()
            )
            if len(pairwise_tests) > 0
            else 0
    }
])

run_summary.to_csv(
    RESIDUAL_AUDIT_DIR
    / "residual_direction_audit_summary.tsv",
    sep="\t",
    index=False
)

print("\n" + "=" * 80)
print("CELL 21 COMPLETED")
print("=" * 80)

display(
    run_summary
)

print("\nOutput directory:")
print(RESIDUAL_AUDIT_DIR)