# ============================================================
# AIDO-BBA BRCA 1.0
# CELL 28
# Stable archetype anchors, boundary audit cases,
# and non-circular fuzzy-state validation
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

MIN_GROUP_N = 10

STABLE_CONSISTENCY_THRESHOLD = 0.80
MODERATE_CONSISTENCY_THRESHOLD = 0.60

HIGH_CORE_MEMBERSHIP = 0.75
LOW_MEMBERSHIP_UNCERTAINTY = 0.15

TOP_N_ANCHORS_PER_CORE = 25
TOP_N_BOUNDARY_PATIENTS = 100


# ============================================================
# 1. FIND COMPLETED FUZZY RUN
# ============================================================

candidate_runs = []

for run_dir in OUTPUT_ROOT.iterdir():

    if not run_dir.is_dir():
        continue

    required_files = [
        (
            run_dir
            / "14_fuzzy_explanatory_states"
            / "summaries"
            / "patient_fuzzy_explanatory_states.tsv"
        ),
        (
            run_dir
            / "14_fuzzy_explanatory_states"
            / "summaries"
            / "core_repeat_reliability_summary.tsv"
        ),
        (
            run_dir
            / "13_gap_module_cores"
            / "summaries"
            / "stable_core_gene_manifest.tsv"
        )
    ]

    if all(
        path.exists()
        for path in required_files
    ):

        candidate_runs.append(
            run_dir
        )

if len(candidate_runs) == 0:

    raise FileNotFoundError(
        "No completed fuzzy explanatory-state run found."
    )

RUN_DIR = sorted(
    candidate_runs,
    key=lambda path: path.stat().st_mtime,
    reverse=True
)[0]

FUZZY_DIR = (
    RUN_DIR
    / "14_fuzzy_explanatory_states"
)

FUZZY_SUMMARY_DIR = (
    FUZZY_DIR
    / "summaries"
)

CORE_SUMMARY_DIR = (
    RUN_DIR
    / "13_gap_module_cores"
    / "summaries"
)

ANCHOR_DIR = (
    RUN_DIR
    / "15_fuzzy_anchor_boundary_audit"
)

ANCHOR_SUMMARY_DIR = (
    ANCHOR_DIR
    / "summaries"
)

ANCHOR_SUMMARY_DIR.mkdir(
    parents=True,
    exist_ok=True
)

print("=" * 80)
print("AIDO-BBA STABLE ANCHOR AND BOUNDARY-PATIENT AUDIT")
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
# 3. LOAD INPUTS
# ============================================================

patients = pd.read_csv(
    FUZZY_SUMMARY_DIR
    / "patient_fuzzy_explanatory_states.tsv",
    sep="\t"
)

core_reliability = pd.read_csv(
    FUZZY_SUMMARY_DIR
    / "core_repeat_reliability_summary.tsv",
    sep="\t"
)

core_manifest = pd.read_csv(
    CORE_SUMMARY_DIR
    / "stable_core_gene_manifest.tsv",
    sep="\t"
)

core_names = sorted(
    core_reliability[
        "core_module_name"
    ]
    .astype(str)
    .tolist()
)

print("\nPatients:", len(patients))
print("Stable cores:", core_names)


# ============================================================
# 4. CORE RELIABILITY WEIGHTS
# ============================================================

reliability_map = dict(
    zip(
        core_reliability[
            "core_module_name"
        ].astype(str),

        core_reliability[
            "mean_pairwise_repeat_rho"
        ].astype(float)
    )
)

maximum_reliability = max(
    reliability_map.values()
)

reliability_weight = {
    core_name:
        reliability_map[
            core_name
        ]
        /
        maximum_reliability

    for core_name in core_names
}


# ============================================================
# 5. EXTRACT MEMBERSHIP COLUMNS
# ============================================================

magnitude_columns = {
    core_name:
        core_name

    for core_name in core_names
}

positive_columns = {
    core_name:
        (
            f"{core_name}__"
            "mean_fuzzy_positive_membership"
        )

    for core_name in core_names
}

negative_columns = {
    core_name:
        (
            f"{core_name}__"
            "mean_fuzzy_negative_membership"
        )

    for core_name in core_names
}

uncertainty_columns = {
    core_name:
        (
            f"{core_name}__"
            "sd_fuzzy_magnitude_membership"
        )

    for core_name in core_names
}

for column_map in [
    magnitude_columns,
    positive_columns,
    negative_columns,
    uncertainty_columns
]:

    missing_columns = [
        column
        for column in column_map.values()
        if column not in patients.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing fuzzy columns: "
            + ", ".join(
                missing_columns
            )
        )


# ============================================================
# 6. STABLE ARCHETYPE-ANCHOR SCORES
# ============================================================

for core_name in core_names:

    membership_column = (
        magnitude_columns[
            core_name
        ]
    )

    uncertainty_column = (
        uncertainty_columns[
            core_name
        ]
    )

    other_core_names = [
        other_core
        for other_core in core_names
        if other_core != core_name
    ]

    other_membership_columns = [
        magnitude_columns[
            other_core
        ]
        for other_core in other_core_names
    ]

    patients[
        f"{core_name}__next_highest_membership"
    ] = (
        patients[
            other_membership_columns
        ]
        .max(axis=1)
    )

    patients[
        f"{core_name}__dominance_margin"
    ] = (
        patients[
            membership_column
        ]
        -
        patients[
            f"{core_name}__next_highest_membership"
        ]
    )

    patients[
        f"{core_name}__stable_anchor_score"
    ] = (
        0.35
        * patients[
            membership_column
        ]
        +
        0.25
        * patients[
            "repeat_state_consistency"
        ]
        +
        0.20
        * (
            1.0
            -
            patients[
                uncertainty_column
            ]
        )
        +
        0.15
        * patients[
            f"{core_name}__dominance_margin"
        ].clip(
            lower=0
        )
        +
        0.05
        * reliability_weight[
            core_name
        ]
    )

    patients[
        f"{core_name}__anchor_eligible"
    ] = (
        (
            patients[
                membership_column
            ]
            >=
            HIGH_CORE_MEMBERSHIP
        )
        &
        (
            patients[
                "repeat_state_consistency"
            ]
            >=
            MODERATE_CONSISTENCY_THRESHOLD
        )
        &
        (
            patients[
                uncertainty_column
            ]
            <=
            LOW_MEMBERSHIP_UNCERTAINTY
        )
        &
        (
            patients[
                f"{core_name}__dominance_margin"
            ]
            > 0
        )
    )


# ============================================================
# 7. BUILD STABLE ANCHOR TABLE
# ============================================================

anchor_tables = []

for core_name in core_names:

    anchor_table = (
        patients[
            patients[
                f"{core_name}__anchor_eligible"
            ]
        ]
        .copy()
        .sort_values(
            [
                f"{core_name}__stable_anchor_score",
                magnitude_columns[
                    core_name
                ],
                "repeat_state_consistency"
            ],
            ascending=[
                False,
                False,
                False
            ]
        )
        .head(
            TOP_N_ANCHORS_PER_CORE
        )
    )

    anchor_table.insert(
        0,
        "anchor_core",
        core_name
    )

    anchor_table[
        "anchor_score"
    ] = (
        anchor_table[
            f"{core_name}__stable_anchor_score"
        ]
    )

    anchor_table[
        "anchor_membership"
    ] = (
        anchor_table[
            magnitude_columns[
                core_name
            ]
        ]
    )

    anchor_table[
        "anchor_positive_membership"
    ] = (
        anchor_table[
            positive_columns[
                core_name
            ]
        ]
    )

    anchor_table[
        "anchor_negative_membership"
    ] = (
        anchor_table[
            negative_columns[
                core_name
            ]
        ]
    )

    anchor_table[
        "anchor_uncertainty"
    ] = (
        anchor_table[
            uncertainty_columns[
                core_name
            ]
        ]
    )

    anchor_table[
        "anchor_dominance_margin"
    ] = (
        anchor_table[
            f"{core_name}__dominance_margin"
        ]
    )

    anchor_tables.append(
        anchor_table
    )


stable_anchor_patients = pd.concat(
    anchor_tables,
    ignore_index=True
)


# ============================================================
# 8. AUDIT-PRIORITY BOUNDARY SCORE
# ============================================================

discordant_states = {
    "clinical_early_molecular_advanced_like",
    "clinical_advanced_molecular_early_like"
}

patients[
    "clinical_molecular_discordance_flag"
] = (
    patients[
        "clinical_molecular_rank_state"
    ]
    .isin(
        discordant_states
    )
    .astype(float)
)

patients[
    "model_dependence_flag"
] = (
    patients[
        "model_dependence_tier"
    ]
    .astype(str)
    .str.contains(
        "high",
        case=False,
        na=False
    )
    .astype(float)
)

patients[
    "repeat_instability_flag"
] = (
    patients[
        "repeat_instability_tier"
    ]
    .astype(str)
    .str.contains(
        "high",
        case=False,
        na=False
    )
    .astype(float)
)

patients[
    "fuzzy_uncertainty_percentile"
] = (
    patients[
        "mean_core_membership_uncertainty"
    ]
    .rank(
        method="average",
        pct=True
    )
)

patients[
    "low_repeat_consistency"
] = (
    1.0
    -
    patients[
        "repeat_state_consistency"
    ]
)

patients[
    "null_gap_percentile"
] = (
    patients[
        "null_corrected_priority_score"
    ]
    .rank(
        method="average",
        pct=True
    )
)

patients[
    "boundary_audit_priority_score"
] = (
    0.25
    * patients[
        "clinical_molecular_discordance_flag"
    ]
    +
    0.15
    * patients[
        "model_dependence_flag"
    ]
    +
    0.15
    * patients[
        "repeat_instability_flag"
    ]
    +
    0.20
    * patients[
        "fuzzy_uncertainty_percentile"
    ]
    +
    0.15
    * patients[
        "low_repeat_consistency"
    ]
    +
    0.10
    * patients[
        "null_gap_percentile"
    ]
)

boundary_patients = (
    patients
    .sort_values(
        [
            "boundary_audit_priority_score",
            "null_corrected_priority_score",
            "mean_core_membership_uncertainty"
        ],
        ascending=[
            False,
            False,
            False
        ]
    )
    .head(
        TOP_N_BOUNDARY_PATIENTS
    )
    .copy()
)


# ============================================================
# 9. NON-CIRCULAR STATE TESTS
# ============================================================

# Do not test membership metrics against mean_fuzzy_state,
# because the state is constructed from those same metrics.

external_state_variables = [
    column
    for column in [
        "true_group",
        "integrated_bba_state",
        "clinical_molecular_rank_state",
        "model_dependence_tier",
        "repeat_instability_tier"
    ]
    if column in patients.columns
]

membership_metrics = []

for core_name in core_names:

    membership_metrics.extend([
        positive_columns[
            core_name
        ],

        negative_columns[
            core_name
        ],

        magnitude_columns[
            core_name
        ],

        uncertainty_columns[
            core_name
        ]
    ])

omnibus_records = []
pairwise_records = []

for state_variable in external_state_variables:

    state_counts = (
        patients[
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
                patients.loc[
                    patients[
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

        if len(arrays) >= 2:

            statistic, p_value = kruskal(
                *arrays
            )

            omnibus_records.append({
                "state_variable":
                    state_variable,

                "metric":
                    metric,

                "n_groups":
                    len(
                        arrays
                    ),

                "groups":
                    " | ".join(
                        valid_states
                    ),

                "kruskal_h":
                    float(
                        statistic
                    ),

                "p_value":
                    float(
                        p_value
                    )
            })

        for group_1, group_2 in combinations(
            eligible_states,
            2
        ):

            values_1 = (
                patients.loc[
                    patients[
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
                patients.loc[
                    patients[
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


noncircular_omnibus = pd.DataFrame(
    omnibus_records
)

noncircular_pairwise = pd.DataFrame(
    pairwise_records
)

noncircular_omnibus[
    "fdr_global"
] = benjamini_hochberg(
    noncircular_omnibus[
        "p_value"
    ].to_numpy()
)

noncircular_pairwise[
    "fdr_within_state_metric"
] = np.nan

for (
    state_variable,
    metric
), row_indices in (
    noncircular_pairwise
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

    noncircular_pairwise.loc[
        row_indices,
        "fdr_within_state_metric"
    ] = benjamini_hochberg(
        noncircular_pairwise.loc[
            row_indices,
            "p_value"
        ].to_numpy()
    )

noncircular_pairwise[
    "fdr_global"
] = benjamini_hochberg(
    noncircular_pairwise[
        "p_value"
    ].to_numpy()
)


# ============================================================
# 10. FUZZY STATE STABILITY BY BBA STATE
# ============================================================

stability_by_state_records = []

for state_variable in external_state_variables:

    for state_value, state_df in (
        patients.groupby(
            state_variable,
            dropna=False
        )
    ):

        stability_by_state_records.append({
            "state_variable":
                state_variable,

            "state_value":
                str(
                    state_value
                ),

            "n":
                len(
                    state_df
                ),

            "mean_repeat_state_consistency":
                float(
                    state_df[
                        "repeat_state_consistency"
                    ].mean()
                ),

            "median_repeat_state_consistency":
                float(
                    state_df[
                        "repeat_state_consistency"
                    ].median()
                ),

            "fraction_stable":
                float(
                    np.mean(
                        state_df[
                            "fuzzy_state_stability_tier"
                        ]
                        ==
                        "stable"
                    )
                ),

            "fraction_unstable":
                float(
                    np.mean(
                        state_df[
                            "fuzzy_state_stability_tier"
                        ]
                        ==
                        "unstable"
                    )
                )
        })


stability_by_state = pd.DataFrame(
    stability_by_state_records
)


# ============================================================
# 11. CORE DIRECTION PROFILE
# ============================================================

direction_profile_records = []

for core_name in core_names:

    positive_column = (
        positive_columns[
            core_name
        ]
    )

    negative_column = (
        negative_columns[
            core_name
        ]
    )

    direction_profile_records.append({
        "core_name":
            core_name,

        "mean_positive_membership":
            float(
                patients[
                    positive_column
                ].mean()
            ),

        "mean_negative_membership":
            float(
                patients[
                    negative_column
                ].mean()
            ),

        "fraction_positive_dominant":
            float(
                np.mean(
                    patients[
                        positive_column
                    ]
                    >
                    patients[
                        negative_column
                    ]
                )
            ),

        "fraction_negative_dominant":
            float(
                np.mean(
                    patients[
                        negative_column
                    ]
                    >
                    patients[
                        positive_column
                    ]
                )
            ),

        "mean_magnitude_membership":
            float(
                patients[
                    magnitude_columns[
                        core_name
                    ]
                ].mean()
            ),

        "mean_membership_uncertainty":
            float(
                patients[
                    uncertainty_columns[
                        core_name
                    ]
                ].mean()
            ),

        "mean_repeat_reliability":
            float(
                reliability_map[
                    core_name
                ]
            )
    })


core_direction_profile = pd.DataFrame(
    direction_profile_records
)


# ============================================================
# 12. SAVE OUTPUTS
# ============================================================

patients.to_csv(
    ANCHOR_SUMMARY_DIR
    / "patient_anchor_and_boundary_scores.tsv",
    sep="\t",
    index=False
)

stable_anchor_patients.to_csv(
    ANCHOR_SUMMARY_DIR
    / "stable_archetype_anchor_patients.tsv",
    sep="\t",
    index=False
)

boundary_patients.to_csv(
    ANCHOR_SUMMARY_DIR
    / "boundary_audit_priority_patients_top100.tsv",
    sep="\t",
    index=False
)

noncircular_omnibus.to_csv(
    ANCHOR_SUMMARY_DIR
    / "noncircular_fuzzy_omnibus_tests.tsv",
    sep="\t",
    index=False
)

noncircular_pairwise.to_csv(
    ANCHOR_SUMMARY_DIR
    / "noncircular_fuzzy_pairwise_tests.tsv",
    sep="\t",
    index=False
)

stability_by_state.to_csv(
    ANCHOR_SUMMARY_DIR
    / "fuzzy_stability_by_bba_state.tsv",
    sep="\t",
    index=False
)

core_direction_profile.to_csv(
    ANCHOR_SUMMARY_DIR
    / "core_direction_profile.tsv",
    sep="\t",
    index=False
)


# ============================================================
# 13. SUMMARY
# ============================================================

summary_table = pd.DataFrame([
    {
        "metric":
            "n_patients",

        "value":
            len(
                patients
            )
    },
    {
        "metric":
            "stable_anchor_rows",

        "value":
            len(
                stable_anchor_patients
            )
    },
    {
        "metric":
            "unique_stable_anchor_patients",

        "value":
            stable_anchor_patients[
                "patient_id"
            ].nunique()
    },
    {
        "metric":
            "core01_anchor_patients",

        "value":
            int(
                (
                    stable_anchor_patients[
                        "anchor_core"
                    ]
                    ==
                    "GapCore_01"
                ).sum()
            )
    },
    {
        "metric":
            "core02_anchor_patients",

        "value":
            int(
                (
                    stable_anchor_patients[
                        "anchor_core"
                    ]
                    ==
                    "GapCore_02"
                ).sum()
            )
    },
    {
        "metric":
            "core03_anchor_patients",

        "value":
            int(
                (
                    stable_anchor_patients[
                        "anchor_core"
                    ]
                    ==
                    "GapCore_03"
                ).sum()
            )
    },
    {
        "metric":
            "significant_noncircular_omnibus_fdr_0_05",

        "value":
            int(
                (
                    noncircular_omnibus[
                        "fdr_global"
                    ]
                    <= 0.05
                ).sum()
            )
    },
    {
        "metric":
            "significant_noncircular_pairwise_fdr_0_05",

        "value":
            int(
                (
                    noncircular_pairwise[
                        "fdr_within_state_metric"
                    ]
                    <= 0.05
                ).sum()
            )
    },
    {
        "metric":
            "medium_or_large_noncircular_pairwise_effects",

        "value":
            int(
                noncircular_pairwise[
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
    ANCHOR_SUMMARY_DIR
    / "anchor_boundary_audit_summary.tsv",
    sep="\t",
    index=False
)


manifest = {
    "analysis":
        (
            "Stable archetype anchors and "
            "boundary audit-priority patients"
        ),

    "run_directory":
        str(
            RUN_DIR
        ),

    "anchor_definition":
        (
            "High fuzzy membership, moderate-or-better "
            "repeat consistency, low membership uncertainty, "
            "and positive dominance margin."
        ),

    "boundary_definition":
        (
            "Clinical-molecular discordance, model dependence, "
            "repeat instability, fuzzy uncertainty, and "
            "null-corrected representation-gap burden."
        ),

    "noncircular_validation":
        (
            "Fuzzy memberships were tested against clinical and "
            "audit state variables, excluding mean_fuzzy_state "
            "because it was constructed from those memberships."
        ),

    "interpretation_boundary":
        (
            "Anchors are representative explanatory-state "
            "examples; boundary patients are measurement and "
            "audit priorities. Neither is a biological subtype."
        )
}

with open(
    ANCHOR_DIR
    / "anchor_boundary_audit_manifest.json",
    "w",
    encoding="utf-8"
) as handle:

    json.dump(
        manifest,
        handle,
        indent=2
    )


# ============================================================
# 14. FINAL REPORT
# ============================================================

print("\n" + "=" * 80)
print("CELL 28 COMPLETED")
print("=" * 80)

display(
    summary_table
)

print("\nCore direction profiles:")

display(
    core_direction_profile
)

print("\nStable archetype anchors:")

anchor_display_columns = [
    column
    for column in [
        "anchor_core",
        "patient_id",
        "true_group",
        "integrated_bba_state",
        "clinical_molecular_rank_state",
        "model_dependence_tier",
        "repeat_instability_tier",
        "mean_fuzzy_state",
        "repeat_state_consistency",
        "anchor_membership",
        "anchor_positive_membership",
        "anchor_negative_membership",
        "anchor_uncertainty",
        "anchor_dominance_margin",
        "anchor_score"
    ]
    if column in stable_anchor_patients.columns
]

display(
    stable_anchor_patients[
        anchor_display_columns
    ]
    .groupby(
        "anchor_core",
        as_index=False
    )
    .head(10)
)

print("\nBoundary audit-priority patients:")

boundary_display_columns = [
    column
    for column in [
        "patient_id",
        "true_group",
        "integrated_bba_state",
        "clinical_molecular_rank_state",
        "model_dependence_tier",
        "repeat_instability_tier",
        "mean_fuzzy_state",
        "repeat_state_consistency",
        "mean_core_membership_uncertainty",
        "null_corrected_priority_score",
        "clinical_molecular_discordance_flag",
        "model_dependence_flag",
        "repeat_instability_flag",
        "boundary_audit_priority_score"
    ]
    if column in boundary_patients.columns
]

display(
    boundary_patients[
        boundary_display_columns
    ].head(40)
)

print("\nStrongest non-circular fuzzy associations:")

noncircular_sorted = (
    noncircular_pairwise
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
    noncircular_sorted[
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

print("\nFuzzy stability by clinical/audit state:")

display(
    stability_by_state
)

print("\nOutput directory:")
print(ANCHOR_DIR)