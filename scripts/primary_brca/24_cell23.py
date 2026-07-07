# ============================================================
# AIDO-BBA BRCA 1.0
# CELL 23
# Patient-specific mapping-null correction
#
# Outputs:
# 1. residual excess beyond random gene-to-GO mapping
# 2. coverage excess beyond mapping null
# 3. top-100 BP concentration excess
# 4. patient-level empirical null percentiles
# 5. BBA-state comparisons using null-corrected metrics
# ============================================================

from pathlib import Path
from aido_bba.config import (
    data_root, brca_output_root, brca_run_dir, go_bp_gmt,
    hgnc_file, ncbi_gene_info, metabric_dir, gse96058_dir,
    kirc_dir, kirc_output_root,
)

from collections import defaultdict
from scipy.sparse import csr_matrix
from scipy.stats import kruskal, mannwhitneyu
from itertools import combinations

import numpy as np
import pandas as pd
import re
import time
import json
import warnings

warnings.filterwarnings("ignore")


# ============================================================
# 0. SETTINGS
# ============================================================

OUTPUT_ROOT = brca_output_root()

GO_BP_GMT = go_bp_gmt()

N_PERMUTATIONS = 200
RANDOM_SEED = 20260701
TOP_K_BP = 100
MIN_GROUP_N = 10

rng = np.random.default_rng(
    RANDOM_SEED
)


# ============================================================
# 1. FIND RUN
# ============================================================

candidate_runs = []

for run_dir in OUTPUT_ROOT.iterdir():

    if not run_dir.is_dir():
        continue

    required = [
        run_dir / "05_attribution" / "shap_matrices",
        (
            run_dir
            / "06_bp_reconstruction"
            / "summaries"
            / "patient_completeness_all_repeats.tsv"
        ),
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
            / "04_blackbox"
            / "bba_patient_state_taxonomy.tsv"
        )
    ]

    if all(path.exists() for path in required):
        candidate_runs.append(run_dir)

if len(candidate_runs) == 0:

    raise FileNotFoundError(
        "No completed AIDO-BBA run found."
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

BLACKBOX_DIR = (
    RUN_DIR
    / "04_blackbox"
)

NULL_CORRECTED_DIR = (
    RUN_DIR
    / "10_null_corrected_completeness"
)

NULL_CORRECTED_DIR.mkdir(
    parents=True,
    exist_ok=True
)

print("=" * 80)
print("AIDO-BBA PATIENT-SPECIFIC NULL-CORRECTED COMPLETENESS")
print("=" * 80)

print("\nRun:")
print(RUN_DIR)


# ============================================================
# 2. LOAD DATA
# ============================================================

patient_repeat_observed = pd.read_csv(
    BP_SUMMARY_DIR
    / "patient_completeness_all_repeats.tsv",
    sep="\t"
)

patient_states = pd.read_csv(
    BLACKBOX_DIR
    / "bba_patient_state_taxonomy.tsv",
    sep="\t"
)

gene_harmonization = pd.read_csv(
    BP_SUMMARY_DIR
    / "model_gene_harmonization_audit.tsv",
    sep="\t",
    dtype=str
)

eligible_bp = pd.read_csv(
    BP_SUMMARY_DIR
    / "eligible_go_bp_universe.tsv",
    sep="\t"
)

shap_files = sorted(
    SHAP_MATRIX_DIR.glob(
        "extratrees_shap_repeat_*_fold_*.npz"
    )
)

if len(shap_files) != 25:

    raise ValueError(
        f"Expected 25 SHAP files, found {len(shap_files)}."
    )


# ============================================================
# 3. SYMBOL AND BP MAPPING
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


raw_to_harmonized = {}

for _, row in gene_harmonization.iterrows():

    raw_symbol = clean_symbol(
        row.get("normalized_symbol")
    )

    harmonized_symbol = clean_symbol(
        row.get("harmonized_symbol")
    )

    if raw_symbol is not None:

        raw_to_harmonized[
            raw_symbol
        ] = harmonized_symbol


eligible_bp = eligible_bp.sort_values(
    "bp_index"
).reset_index(drop=True)

bp_names = eligible_bp[
    "term_name"
].astype(str).to_numpy()

bp_name_to_index = {
    name: index
    for index, name in enumerate(bp_names)
}

eligible_bp_name_set = set(
    bp_names
)

gene_to_bp_indices = defaultdict(list)

with open(
    GO_BP_GMT,
    "r",
    encoding="utf-8"
) as handle:

    for line in handle:

        fields = line.rstrip("\n").split("\t")

        if len(fields) < 3:
            continue

        term_name = fields[0].strip()

        if term_name not in eligible_bp_name_set:
            continue

        bp_index = bp_name_to_index[
            term_name
        ]

        genes = {
            clean_symbol(gene)
            for gene in fields[2:]
        }

        genes.discard(None)

        for gene in genes:

            gene_to_bp_indices[
                gene
            ].append(bp_index)


def build_allocation_matrix(
    genes,
    n_bp
):

    rows = []
    columns = []
    values = []

    mapped_mask = np.zeros(
        len(genes),
        dtype=bool
    )

    for feature_index, gene in enumerate(
        genes
    ):

        if gene is None:
            continue

        memberships = gene_to_bp_indices.get(
            str(gene),
            []
        )

        degree = len(memberships)

        if degree == 0:
            continue

        mapped_mask[
            feature_index
        ] = True

        weight = 1.0 / degree

        for bp_index in memberships:

            rows.append(feature_index)
            columns.append(bp_index)
            values.append(weight)

    matrix = csr_matrix(
        (
            values,
            (
                rows,
                columns
            )
        ),
        shape=(
            len(genes),
            n_bp
        ),
        dtype=np.float64
    )

    return matrix, mapped_mask


# ============================================================
# 4. PATIENT-REPEAT OBSERVED LOOKUP
# ============================================================

observed_lookup = (
    patient_repeat_observed
    .set_index(
        [
            "patient_id",
            "repeat_id"
        ]
    )
)

required_observed_columns = [
    "unmapped_signed_residual",
    "attribution_mass_coverage",
    "top100_bp_mass_fraction"
]

for column in required_observed_columns:

    if column not in observed_lookup.columns:

        raise ValueError(
            f"Missing observed column: {column}"
        )


# ============================================================
# 5. ONLINE NULL ACCUMULATORS
# ============================================================

patient_repeat_keys = (
    patient_repeat_observed[
        [
            "patient_id",
            "repeat_id",
            "fold_id"
        ]
    ]
    .drop_duplicates()
    .sort_values(
        [
            "repeat_id",
            "fold_id",
            "patient_id"
        ]
    )
    .reset_index(drop=True)
)

key_to_position = {
    (
        row.patient_id,
        int(row.repeat_id)
    ): position

    for position, row
    in enumerate(
        patient_repeat_keys.itertuples(
            index=False
        )
    )
}

n_patient_repeats = len(
    patient_repeat_keys
)

metric_names = [
    "signed_residual",
    "coverage",
    "top100_fraction"
]

null_sum = {
    metric: np.zeros(
        n_patient_repeats,
        dtype=np.float64
    )
    for metric in metric_names
}

null_sum_sq = {
    metric: np.zeros(
        n_patient_repeats,
        dtype=np.float64
    )
    for metric in metric_names
}

null_less_equal_observed = {
    metric: np.zeros(
        n_patient_repeats,
        dtype=np.int32
    )
    for metric in metric_names
}

null_greater_equal_observed = {
    metric: np.zeros(
        n_patient_repeats,
        dtype=np.int32
    )
    for metric in metric_names
}

observed_vectors = {}

for metric, source_column in [
    (
        "signed_residual",
        "unmapped_signed_residual"
    ),
    (
        "coverage",
        "attribution_mass_coverage"
    ),
    (
        "top100_fraction",
        "top100_bp_mass_fraction"
    )
]:

    values = np.zeros(
        n_patient_repeats,
        dtype=np.float64
    )

    for row in patient_repeat_keys.itertuples(
        index=False
    ):

        position = key_to_position[
            (
                row.patient_id,
                int(row.repeat_id)
            )
        ]

        values[position] = float(
            observed_lookup.loc[
                (
                    row.patient_id,
                    int(row.repeat_id)
                ),
                source_column
            ]
        )

    observed_vectors[
        metric
    ] = values


# ============================================================
# 6. LOAD FOLD SHAP DATA
# ============================================================

fold_data = []

for shap_file in shap_files:

    match = re.search(
        r"repeat_(\d+)_fold_(\d+)",
        shap_file.name
    )

    if match is None:

        raise ValueError(
            f"Cannot parse fold file: {shap_file}"
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

        patient_ids = (
            data["patient_ids"]
            .astype(str)
        )

        selected_genes_raw = (
            data["selected_genes"]
            .astype(str)
        )

        shap_values = (
            data["shap_values_advanced"]
            .astype(np.float64)
        )

    harmonized_genes = np.asarray(
        [
            raw_to_harmonized.get(
                clean_symbol(gene),
                clean_symbol(gene)
            )
            for gene in selected_genes_raw
        ],
        dtype=object
    )

    patient_positions = np.asarray(
        [
            key_to_position[
                (
                    patient_id,
                    repeat_id
                )
            ]
            for patient_id in patient_ids
        ],
        dtype=int
    )

    total_absolute_shap = np.abs(
        shap_values
    ).sum(axis=1)

    fold_data.append({
        "repeat_id":
            repeat_id,

        "fold_id":
            fold_id,

        "patient_ids":
            patient_ids,

        "patient_positions":
            patient_positions,

        "harmonized_genes":
            harmonized_genes,

        "shap_values":
            shap_values,

        "total_absolute_shap":
            total_absolute_shap
    })


# ============================================================
# 7. RUN PATIENT-SPECIFIC MAPPING NULL
# ============================================================

start_time = time.time()

print("\n" + "=" * 80)
print("RUNNING PATIENT-SPECIFIC NULL")
print("=" * 80)

for permutation_id in range(
    1,
    N_PERMUTATIONS + 1
):

    for fold in fold_data:

        shap_values = fold[
            "shap_values"
        ]

        permuted_genes = rng.permutation(
            fold[
                "harmonized_genes"
            ]
        )

        (
            allocation,
            mapped_mask
        ) = build_allocation_matrix(
            permuted_genes,
            len(bp_names)
        )

        total_absolute_shap = fold[
            "total_absolute_shap"
        ]

        mapped_absolute_shap = np.abs(
            shap_values[
                :,
                mapped_mask
            ]
        ).sum(axis=1)

        null_coverage = np.divide(
            mapped_absolute_shap,
            total_absolute_shap,
            out=np.zeros_like(
                mapped_absolute_shap
            ),
            where=(
                total_absolute_shap > 0
            )
        )

        null_signed_residual = (
            shap_values[
                :,
                ~mapped_mask
            ]
            .sum(axis=1)
        )

        bp_shap = (
            allocation.T
            .dot(
                shap_values.T
            )
            .T
        )

        bp_shap = np.asarray(
            bp_shap,
            dtype=np.float64
        )

        absolute_bp_shap = np.abs(
            bp_shap
        )

        n_top = min(
            TOP_K_BP,
            absolute_bp_shap.shape[1]
        )

        top_indices = np.argpartition(
            absolute_bp_shap,
            kth=(
                absolute_bp_shap.shape[1]
                - n_top
            ),
            axis=1
        )[:, -n_top:]

        row_indices = np.arange(
            absolute_bp_shap.shape[0]
        )[:, None]

        top_mass = absolute_bp_shap[
            row_indices,
            top_indices
        ].sum(axis=1)

        total_bp_mass = absolute_bp_shap.sum(
            axis=1
        )

        null_top100 = np.divide(
            top_mass,
            total_bp_mass,
            out=np.zeros_like(
                top_mass
            ),
            where=(
                total_bp_mass > 0
            )
        )

        positions = fold[
            "patient_positions"
        ]

        fold_null_metrics = {
            "signed_residual":
                null_signed_residual,

            "coverage":
                null_coverage,

            "top100_fraction":
                null_top100
        }

        for metric, null_values in (
            fold_null_metrics.items()
        ):

            null_sum[
                metric
            ][positions] += null_values

            null_sum_sq[
                metric
            ][positions] += (
                null_values ** 2
            )

            observed_values = (
                observed_vectors[
                    metric
                ][positions]
            )

            null_less_equal_observed[
                metric
            ][positions] += (
                null_values
                <=
                observed_values
            )

            null_greater_equal_observed[
                metric
            ][positions] += (
                null_values
                >=
                observed_values
            )

    if (
        permutation_id == 1
        or permutation_id % 20 == 0
        or permutation_id == N_PERMUTATIONS
    ):

        elapsed = (
            time.time()
            - start_time
        ) / 60

        print(
            f"Permutation "
            f"{permutation_id:>3}/"
            f"{N_PERMUTATIONS} | "
            f"{elapsed:.2f} min"
        )


# ============================================================
# 8. BUILD PATIENT-REPEAT NULL-CORRECTED TABLE
# ============================================================

null_corrected_repeat = (
    patient_repeat_keys.copy()
)

for metric in metric_names:

    null_mean = (
        null_sum[metric]
        / N_PERMUTATIONS
    )

    null_variance = (
        null_sum_sq[metric]
        / N_PERMUTATIONS
        -
        null_mean ** 2
    )

    null_variance = np.maximum(
        null_variance,
        0
    )

    null_sd = np.sqrt(
        null_variance
    )

    observed = observed_vectors[
        metric
    ]

    excess = (
        observed
        - null_mean
    )

    z_score = np.divide(
        excess,
        null_sd,
        out=np.full_like(
            excess,
            np.nan
        ),
        where=(
            null_sd > 0
        )
    )

    lower_p = (
        1
        +
        null_less_equal_observed[
            metric
        ]
    ) / (
        N_PERMUTATIONS
        + 1
    )

    upper_p = (
        1
        +
        null_greater_equal_observed[
            metric
        ]
    ) / (
        N_PERMUTATIONS
        + 1
    )

    two_sided_p = np.minimum(
        1.0,
        2.0
        * np.minimum(
            lower_p,
            upper_p
        )
    )

    null_corrected_repeat[
        f"observed_{metric}"
    ] = observed

    null_corrected_repeat[
        f"null_mean_{metric}"
    ] = null_mean

    null_corrected_repeat[
        f"null_sd_{metric}"
    ] = null_sd

    null_corrected_repeat[
        f"excess_{metric}"
    ] = excess

    null_corrected_repeat[
        f"z_{metric}"
    ] = z_score

    null_corrected_repeat[
        f"empirical_lower_p_{metric}"
    ] = lower_p

    null_corrected_repeat[
        f"empirical_upper_p_{metric}"
    ] = upper_p

    null_corrected_repeat[
        f"empirical_two_sided_p_{metric}"
    ] = two_sided_p


# ============================================================
# 9. AGGREGATE ACROSS FIVE REPEATS
# ============================================================

aggregation_columns = []

for metric in metric_names:

    aggregation_columns.extend([
        f"observed_{metric}",
        f"null_mean_{metric}",
        f"excess_{metric}",
        f"z_{metric}"
    ])

patient_null_corrected = (
    null_corrected_repeat
    .groupby(
        "patient_id",
        as_index=False
    )[aggregation_columns]
    .agg([
        "mean",
        "std",
        "median"
    ])
)

patient_null_corrected.columns = [
    (
        column[0]
        if column[1] == ""
        else f"{column[0]}_{column[1]}"
    )
    for column in patient_null_corrected.columns
]

patient_null_corrected = (
    patient_null_corrected
    .reset_index(drop=True)
)


# ============================================================
# 10. MERGE BBA STATES
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
        "mean_repeat_instability"
    ]
    if column in patient_states.columns
]

patient_null_corrected = (
    patient_states[
        state_columns
    ]
    .merge(
        patient_null_corrected,
        on="patient_id",
        how="left",
        validate="one_to_one"
    )
)


# ============================================================
# 11. NULL-CORRECTED STATE TESTS
# ============================================================

corrected_metrics = [
    "excess_signed_residual_mean",
    "excess_coverage_mean",
    "excess_top100_fraction_mean",
    "z_signed_residual_mean",
    "z_coverage_mean",
    "z_top100_fraction_mean"
]

available_state_variables = [
    column
    for column in [
        "true_group",
        "integrated_bba_state",
        "clinical_molecular_rank_state",
        "model_dependence_tier",
        "repeat_instability_tier"
    ]
    if column in patient_null_corrected.columns
]

omnibus_records = []
pairwise_records = []

for state_variable in (
    available_state_variables
):

    state_counts = (
        patient_null_corrected[
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

    for metric in corrected_metrics:

        arrays = []

        valid_states = []

        for state in eligible_states:

            values = (
                patient_null_corrected.loc[
                    patient_null_corrected[
                        state_variable
                    ] == state,
                    metric
                ]
                .dropna()
                .to_numpy(dtype=float)
            )

            if len(values) >= MIN_GROUP_N:

                arrays.append(values)
                valid_states.append(
                    str(state)
                )

        if len(arrays) >= 2:

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
                    float(h_statistic),

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

        for group_1, group_2 in combinations(
            eligible_states,
            2
        ):

            values_1 = (
                patient_null_corrected.loc[
                    patient_null_corrected[
                        state_variable
                    ] == group_1,
                    metric
                ]
                .dropna()
                .to_numpy(dtype=float)
            )

            values_2 = (
                patient_null_corrected.loc[
                    patient_null_corrected[
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

            u_statistic, p_value = (
                mannwhitneyu(
                    values_1,
                    values_2,
                    alternative="two-sided"
                )
            )

            cliffs_delta = (
                2.0
                * u_statistic
                /
                (
                    len(values_1)
                    * len(values_2)
                )
                - 1.0
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
                        np.mean(values_1)
                    ),

                "mean_group_2":
                    float(
                        np.mean(values_2)
                    ),

                "mean_difference":
                    float(
                        np.mean(values_1)
                        -
                        np.mean(values_2)
                    ),

                "cliffs_delta":
                    float(cliffs_delta),

                "absolute_cliffs_delta":
                    float(
                        abs(cliffs_delta)
                    ),

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

pairwise_tests = pd.DataFrame(
    pairwise_records
)


# ============================================================
# 12. PATIENT PRIORITY SCORE
# ============================================================

patient_null_corrected[
    "absolute_residual_excess"
] = (
    patient_null_corrected[
        "excess_signed_residual_mean"
    ].abs()
)

patient_null_corrected[
    "absolute_residual_excess_percentile"
] = (
    patient_null_corrected[
        "absolute_residual_excess"
    ]
    .rank(
        pct=True
    )
)

patient_null_corrected[
    "negative_coverage_excess_percentile"
] = (
    (
        -patient_null_corrected[
            "excess_coverage_mean"
        ]
    )
    .rank(
        pct=True
    )
)

patient_null_corrected[
    "positive_concentration_excess_percentile"
] = (
    patient_null_corrected[
        "excess_top100_fraction_mean"
    ]
    .rank(
        pct=True
    )
)

patient_null_corrected[
    "null_corrected_priority_score"
] = (
    0.40
    * patient_null_corrected[
        "absolute_residual_excess_percentile"
    ]
    +
    0.30
    * patient_null_corrected[
        "negative_coverage_excess_percentile"
    ]
    +
    0.30
    * patient_null_corrected[
        "positive_concentration_excess_percentile"
    ]
)

patient_null_corrected = (
    patient_null_corrected
    .sort_values(
        "null_corrected_priority_score",
        ascending=False
    )
    .reset_index(drop=True)
)


# ============================================================
# 13. SAVE OUTPUTS
# ============================================================

null_corrected_repeat.to_csv(
    NULL_CORRECTED_DIR
    / "patient_repeat_null_corrected_metrics.tsv",
    sep="\t",
    index=False
)

patient_null_corrected.to_csv(
    NULL_CORRECTED_DIR
    / "patient_null_corrected_completeness.tsv",
    sep="\t",
    index=False
)

patient_null_corrected.head(
    100
).to_csv(
    NULL_CORRECTED_DIR
    / "patient_null_corrected_priority_top100.tsv",
    sep="\t",
    index=False
)

omnibus_tests.to_csv(
    NULL_CORRECTED_DIR
    / "null_corrected_state_omnibus_tests.tsv",
    sep="\t",
    index=False
)

pairwise_tests.to_csv(
    NULL_CORRECTED_DIR
    / "null_corrected_state_pairwise_tests.tsv",
    sep="\t",
    index=False
)


manifest = {
    "analysis":
        "Patient-specific mapping-null-corrected completeness",

    "run_directory":
        str(RUN_DIR),

    "n_permutations":
        N_PERMUTATIONS,

    "random_seed":
        RANDOM_SEED,

    "metrics": {
        "residual_excess":
            (
                "Observed unmapped signed residual minus "
                "patient-specific mapping-null expectation."
            ),

        "coverage_excess":
            (
                "Observed attribution-mass coverage minus "
                "patient-specific mapping-null expectation."
            ),

        "top100_concentration_excess":
            (
                "Observed top-100 BP attribution fraction "
                "minus mapping-null expectation."
            )
    }
}

with open(
    NULL_CORRECTED_DIR
    / "null_corrected_manifest.json",
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

duration_minutes = (
    time.time()
    - start_time
) / 60

print("\n" + "=" * 80)
print("CELL 23 COMPLETED")
print("=" * 80)

print(
    "Permutations:",
    N_PERMUTATIONS
)

print(
    "Duration:",
    round(
        duration_minutes,
        2
    ),
    "minutes"
)

print("\nOverall null-corrected metrics:")

overall_summary = pd.DataFrame([
    {
        "metric":
            "mean_residual_excess",

        "value":
            patient_null_corrected[
                "excess_signed_residual_mean"
            ].mean()
    },
    {
        "metric":
            "mean_absolute_residual_excess",

        "value":
            patient_null_corrected[
                "absolute_residual_excess"
            ].mean()
    },
    {
        "metric":
            "mean_coverage_excess",

        "value":
            patient_null_corrected[
                "excess_coverage_mean"
            ].mean()
    },
    {
        "metric":
            "mean_top100_concentration_excess",

        "value":
            patient_null_corrected[
                "excess_top100_fraction_mean"
            ].mean()
    }
])

display(
    overall_summary
)

print("\nStrongest corrected state associations:")

display(
    omnibus_tests
    .sort_values(
        "D_minus_log10_p",
        ascending=False
    )
    .head(30)
)

print("\nStrongest corrected pairwise effects:")

display(
    pairwise_tests
    .sort_values(
        [
            "D_minus_log10_p",
            "absolute_cliffs_delta"
        ],
        ascending=[
            False,
            False
        ]
    )
    .head(40)
)

print("\nTop null-corrected priority patients:")

display_columns = [
    column
    for column in [
        "patient_id",
        "true_group",
        "integrated_bba_state",
        "clinical_molecular_rank_state",
        "model_dependence_tier",
        "repeat_instability_tier",
        "excess_signed_residual_mean",
        "z_signed_residual_mean",
        "excess_coverage_mean",
        "z_coverage_mean",
        "excess_top100_fraction_mean",
        "z_top100_fraction_mean",
        "null_corrected_priority_score"
    ]
    if column in patient_null_corrected.columns
]

display(
    patient_null_corrected[
        display_columns
    ].head(40)
)

print("\nOutput directory:")
print(NULL_CORRECTED_DIR)