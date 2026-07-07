# ============================================================
# AIDO-BBA BRCA 1.0
# CELL 22
# Residual mapping-permutation and sign-flip null audit
# ============================================================

from pathlib import Path
from collections import defaultdict
from scipy.sparse import csr_matrix
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

OUTPUT_ROOT = Path(
    r"D:\AIDO-Temp\AIDO_BBA_BRCA_1_0"
)

GO_BP_GMT = Path(
    r"D:\AIDO-Data\GSEA\c5.go.bp.v2026.1.Hs.symbols.gmt"
)

N_PERMUTATIONS = 100
RANDOM_SEED = 20260701
TOP_K_BP = 100

rng = np.random.default_rng(
    RANDOM_SEED
)


# ============================================================
# 1. FIND COMPLETED RUN
# ============================================================

candidate_runs = []

for run_dir in OUTPUT_ROOT.iterdir():

    if not run_dir.is_dir():
        continue

    required_files = [
        (
            run_dir
            / "05_attribution"
            / "shap_matrices"
        ),
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
        "No completed BP reconstruction run found."
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

NULL_DIR = (
    RUN_DIR
    / "09_residual_null_audit"
)

NULL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

print("=" * 80)
print("AIDO-BBA RESIDUAL NULL-MODEL AUDIT")
print("=" * 80)

print("\nRun:")
print(RUN_DIR)


# ============================================================
# 2. LOAD EXISTING OUTPUTS
# ============================================================

patient_completeness = pd.read_csv(
    BP_SUMMARY_DIR
    / "patient_completeness_all_repeats.tsv",
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
        f"Expected 25 SHAP files; found {len(shap_files)}."
    )

eligible_bp_names = set(
    eligible_bp[
        "term_name"
    ].astype(str)
)

bp_names = (
    eligible_bp
    .sort_values("bp_index")
    ["term_name"]
    .astype(str)
    .to_numpy()
)

bp_name_to_index = {
    bp_name: index
    for index, bp_name in enumerate(bp_names)
}

print("\nSHAP fold files:", len(shap_files))
print("Eligible BP terms:", len(bp_names))


# ============================================================
# 3. SYMBOL UTILITIES
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


# ============================================================
# 4. REBUILD GENE-TO-ELIGIBLE-BP MAPPING
# ============================================================

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

        if term_name not in eligible_bp_names:
            continue

        bp_index = bp_name_to_index[
            term_name
        ]

        term_genes = {
            clean_symbol(gene)
            for gene in fields[2:]
        }

        term_genes.discard(None)

        for gene in term_genes:

            gene_to_bp_indices[
                gene
            ].append(bp_index)

print(
    "Genes with eligible GO-BP membership:",
    len(gene_to_bp_indices)
)


# ============================================================
# 5. ALLOCATION-MATRIX BUILDER
# ============================================================

def build_allocation_matrix(
    harmonized_genes,
    n_bp
):

    row_indices = []
    column_indices = []
    values = []

    mapped_mask = np.zeros(
        len(harmonized_genes),
        dtype=bool
    )

    membership_degree = np.zeros(
        len(harmonized_genes),
        dtype=int
    )

    for feature_index, gene in enumerate(
        harmonized_genes
    ):

        if gene is None:
            continue

        memberships = gene_to_bp_indices.get(
            str(gene),
            []
        )

        degree = len(
            memberships
        )

        membership_degree[
            feature_index
        ] = degree

        if degree == 0:
            continue

        mapped_mask[
            feature_index
        ] = True

        weight = 1.0 / degree

        for bp_index in memberships:

            row_indices.append(
                feature_index
            )

            column_indices.append(
                bp_index
            )

            values.append(
                weight
            )

    matrix = csr_matrix(
        (
            values,
            (
                row_indices,
                column_indices
            )
        ),
        shape=(
            len(harmonized_genes),
            n_bp
        ),
        dtype=np.float64
    )

    return (
        matrix,
        mapped_mask,
        membership_degree
    )


# ============================================================
# 6. LOAD FOLD DATA ONCE
# ============================================================

fold_data = []

for shap_file in shap_files:

    match = re.search(
        r"repeat_(\d+)_fold_(\d+)",
        shap_file.name
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

        selected_genes_raw = (
            data["selected_genes"]
            .astype(str)
        )

        shap_values = (
            data["shap_values_advanced"]
            .astype(np.float64)
        )

        patient_ids = (
            data["patient_ids"]
            .astype(str)
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

    (
        observed_allocation,
        observed_mapped_mask,
        observed_degree
    ) = build_allocation_matrix(
        harmonized_genes,
        len(bp_names)
    )

    fold_data.append({
        "repeat_id":
            repeat_id,

        "fold_id":
            fold_id,

        "patient_ids":
            patient_ids,

        "selected_genes_raw":
            selected_genes_raw,

        "harmonized_genes":
            harmonized_genes,

        "shap_values":
            shap_values,

        "observed_allocation":
            observed_allocation,

        "observed_mapped_mask":
            observed_mapped_mask,

        "observed_degree":
            observed_degree
    })


# ============================================================
# 7. OBSERVED METRICS
# ============================================================

observed_metrics = {
    "mean_attribution_mass_coverage":
        float(
            patient_completeness[
                "attribution_mass_coverage"
            ].mean()
        ),

    "median_attribution_mass_coverage":
        float(
            patient_completeness[
                "attribution_mass_coverage"
            ].median()
        ),

    "mean_signed_residual":
        float(
            patient_completeness[
                "unmapped_signed_residual"
            ].mean()
        ),

    "mean_absolute_signed_residual":
        float(
            np.mean(
                np.abs(
                    patient_completeness[
                        "unmapped_signed_residual"
                    ]
                )
            )
        ),

    "fraction_negative_residual":
        float(
            np.mean(
                patient_completeness[
                    "unmapped_signed_residual"
                ] < 0
            )
        ),

    "mean_top100_bp_mass_fraction":
        float(
            patient_completeness[
                "top100_bp_mass_fraction"
            ].mean()
        )
}

print("\nObserved metrics:")

display(
    pd.DataFrame([
        {
            "metric": metric,
            "value": value
        }
        for metric, value
        in observed_metrics.items()
    ])
)


# ============================================================
# 8. NULL ITERATIONS
# ============================================================

null_records = []

start_time = time.time()

print("\n" + "=" * 80)
print("RUNNING NULL PERMUTATIONS")
print("=" * 80)

for permutation_id in range(
    1,
    N_PERMUTATIONS + 1
):

    mapping_coverages = []
    mapping_signed_residuals = []
    mapping_top100_fractions = []

    signflip_signed_residuals = []

    for fold in fold_data:

        shap_values = fold[
            "shap_values"
        ]

        harmonized_genes = fold[
            "harmonized_genes"
        ]

        # ----------------------------------------------------
        # Null A:
        # Permute gene identities across SHAP columns
        # ----------------------------------------------------

        permuted_genes = rng.permutation(
            harmonized_genes
        )

        (
            permuted_allocation,
            permuted_mapped_mask,
            _
        ) = build_allocation_matrix(
            permuted_genes,
            len(bp_names)
        )

        total_absolute_shap = np.abs(
            shap_values
        ).sum(axis=1)

        mapped_absolute_shap = np.abs(
            shap_values[
                :,
                permuted_mapped_mask
            ]
        ).sum(axis=1)

        mapping_coverage = np.divide(
            mapped_absolute_shap,
            total_absolute_shap,
            out=np.zeros_like(
                mapped_absolute_shap
            ),
            where=(
                total_absolute_shap > 0
            )
        )

        mapping_residual = shap_values[
            :,
            ~permuted_mapped_mask
        ].sum(axis=1)

        mapping_bp_shap = (
            permuted_allocation.T
            .dot(
                shap_values.T
            )
            .T
        )

        mapping_bp_shap = np.asarray(
            mapping_bp_shap
        )

        absolute_bp_shap = np.abs(
            mapping_bp_shap
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

        top_absolute_mass = (
            absolute_bp_shap[
                row_indices,
                top_indices
            ]
            .sum(axis=1)
        )

        total_bp_absolute_mass = (
            absolute_bp_shap.sum(axis=1)
        )

        top100_fraction = np.divide(
            top_absolute_mass,
            total_bp_absolute_mass,
            out=np.zeros_like(
                top_absolute_mass
            ),
            where=(
                total_bp_absolute_mass > 0
            )
        )

        mapping_coverages.extend(
            mapping_coverage.tolist()
        )

        mapping_signed_residuals.extend(
            mapping_residual.tolist()
        )

        mapping_top100_fractions.extend(
            top100_fraction.tolist()
        )

        # ----------------------------------------------------
        # Null B:
        # Randomly flip attribution directions while keeping
        # observed mapping and absolute SHAP magnitude
        # ----------------------------------------------------

        random_signs = rng.choice(
            np.asarray(
                [-1.0, 1.0]
            ),
            size=shap_values.shape,
            replace=True
        )

        sign_flipped_shap = (
            np.abs(
                shap_values
            )
            * random_signs
        )

        signflip_residual = (
            sign_flipped_shap[
                :,
                ~fold[
                    "observed_mapped_mask"
                ]
            ]
            .sum(axis=1)
        )

        signflip_signed_residuals.extend(
            signflip_residual.tolist()
        )

    mapping_coverages = np.asarray(
        mapping_coverages
    )

    mapping_signed_residuals = np.asarray(
        mapping_signed_residuals
    )

    mapping_top100_fractions = np.asarray(
        mapping_top100_fractions
    )

    signflip_signed_residuals = np.asarray(
        signflip_signed_residuals
    )

    null_records.append({
        "permutation_id":
            permutation_id,

        "mapping_mean_attribution_mass_coverage":
            float(
                np.mean(
                    mapping_coverages
                )
            ),

        "mapping_median_attribution_mass_coverage":
            float(
                np.median(
                    mapping_coverages
                )
            ),

        "mapping_mean_signed_residual":
            float(
                np.mean(
                    mapping_signed_residuals
                )
            ),

        "mapping_mean_absolute_signed_residual":
            float(
                np.mean(
                    np.abs(
                        mapping_signed_residuals
                    )
                )
            ),

        "mapping_fraction_negative_residual":
            float(
                np.mean(
                    mapping_signed_residuals < 0
                )
            ),

        "mapping_mean_top100_bp_mass_fraction":
            float(
                np.mean(
                    mapping_top100_fractions
                )
            ),

        "signflip_mean_signed_residual":
            float(
                np.mean(
                    signflip_signed_residuals
                )
            ),

        "signflip_mean_absolute_signed_residual":
            float(
                np.mean(
                    np.abs(
                        signflip_signed_residuals
                    )
                )
            ),

        "signflip_fraction_negative_residual":
            float(
                np.mean(
                    signflip_signed_residuals < 0
                )
            )
    })

    if (
        permutation_id == 1
        or permutation_id % 10 == 0
        or permutation_id == N_PERMUTATIONS
    ):

        elapsed_minutes = (
            time.time()
            - start_time
        ) / 60

        print(
            f"Permutation "
            f"{permutation_id:>3}/"
            f"{N_PERMUTATIONS} | "
            f"{elapsed_minutes:.2f} min"
        )


null_results = pd.DataFrame(
    null_records
)


# ============================================================
# 9. EMPIRICAL P-VALUE FUNCTIONS
# ============================================================

def empirical_two_sided_p(
    observed,
    null_values
):

    null_values = np.asarray(
        null_values,
        dtype=float
    )

    null_center = np.mean(
        null_values
    )

    observed_distance = abs(
        observed
        - null_center
    )

    null_distances = np.abs(
        null_values
        - null_center
    )

    return float(
        (
            1
            +
            np.sum(
                null_distances
                >= observed_distance
            )
        )
        /
        (
            len(null_values)
            + 1
        )
    )


def empirical_lower_p(
    observed,
    null_values
):

    null_values = np.asarray(
        null_values,
        dtype=float
    )

    return float(
        (
            1
            +
            np.sum(
                null_values <= observed
            )
        )
        /
        (
            len(null_values)
            + 1
        )
    )


def empirical_upper_p(
    observed,
    null_values
):

    null_values = np.asarray(
        null_values,
        dtype=float
    )

    return float(
        (
            1
            +
            np.sum(
                null_values >= observed
            )
        )
        /
        (
            len(null_values)
            + 1
        )
    )


# ============================================================
# 10. NULL COMPARISON SUMMARY
# ============================================================

comparison_records = []

comparison_definitions = [
    (
        "mean_attribution_mass_coverage",
        observed_metrics[
            "mean_attribution_mass_coverage"
        ],
        "mapping_mean_attribution_mass_coverage"
    ),
    (
        "median_attribution_mass_coverage",
        observed_metrics[
            "median_attribution_mass_coverage"
        ],
        "mapping_median_attribution_mass_coverage"
    ),
    (
        "mean_signed_residual",
        observed_metrics[
            "mean_signed_residual"
        ],
        "mapping_mean_signed_residual"
    ),
    (
        "mean_absolute_signed_residual",
        observed_metrics[
            "mean_absolute_signed_residual"
        ],
        "mapping_mean_absolute_signed_residual"
    ),
    (
        "fraction_negative_residual",
        observed_metrics[
            "fraction_negative_residual"
        ],
        "mapping_fraction_negative_residual"
    ),
    (
        "mean_top100_bp_mass_fraction",
        observed_metrics[
            "mean_top100_bp_mass_fraction"
        ],
        "mapping_mean_top100_bp_mass_fraction"
    ),
    (
        "mean_signed_residual_signflip_null",
        observed_metrics[
            "mean_signed_residual"
        ],
        "signflip_mean_signed_residual"
    ),
    (
        "fraction_negative_residual_signflip_null",
        observed_metrics[
            "fraction_negative_residual"
        ],
        "signflip_fraction_negative_residual"
    )
]

for (
    metric_name,
    observed_value,
    null_column
) in comparison_definitions:

    null_values = null_results[
        null_column
    ].to_numpy(dtype=float)

    null_mean = float(
        np.mean(
            null_values
        )
    )

    null_sd = float(
        np.std(
            null_values,
            ddof=1
        )
    )

    z_score = (
        (
            observed_value
            - null_mean
        )
        / null_sd
        if null_sd > 0
        else np.nan
    )

    comparison_records.append({
        "metric":
            metric_name,

        "null_column":
            null_column,

        "observed_value":
            observed_value,

        "null_mean":
            null_mean,

        "null_sd":
            null_sd,

        "observed_minus_null":
            observed_value
            - null_mean,

        "z_score":
            z_score,

        "empirical_two_sided_p":
            empirical_two_sided_p(
                observed_value,
                null_values
            ),

        "empirical_lower_tail_p":
            empirical_lower_p(
                observed_value,
                null_values
            ),

        "empirical_upper_tail_p":
            empirical_upper_p(
                observed_value,
                null_values
            )
    })

null_comparison = pd.DataFrame(
    comparison_records
)


# ============================================================
# 11. SAVE OUTPUTS
# ============================================================

null_results.to_csv(
    NULL_DIR
    / "residual_null_permutation_results.tsv",
    sep="\t",
    index=False
)

null_comparison.to_csv(
    NULL_DIR
    / "residual_null_comparison_summary.tsv",
    sep="\t",
    index=False
)

manifest = {
    "analysis":
        "AIDO-BBA residual null-model audit",

    "run_directory":
        str(RUN_DIR),

    "n_mapping_permutations":
        N_PERMUTATIONS,

    "random_seed":
        RANDOM_SEED,

    "top_k_bp":
        TOP_K_BP,

    "mapping_null":
        (
            "Gene identities were permuted across "
            "fold-selected SHAP columns."
        ),

    "signflip_null":
        (
            "Observed absolute SHAP values were retained "
            "while attribution signs were randomized."
        )
}

with open(
    NULL_DIR
    / "residual_null_manifest.json",
    "w",
    encoding="utf-8"
) as handle:

    json.dump(
        manifest,
        handle,
        indent=2
    )


# ============================================================
# 12. FINAL REPORT
# ============================================================

total_minutes = (
    time.time()
    - start_time
) / 60

print("\n" + "=" * 80)
print("CELL 22 COMPLETED")
print("=" * 80)

print(
    "Permutations:",
    N_PERMUTATIONS
)

print(
    "Duration:",
    round(
        total_minutes,
        2
    ),
    "minutes"
)

print("\nObserved versus null:")

display(
    null_comparison.round(8)
)

print("\nFirst null permutations:")

display(
    null_results.head(10)
)

print("\nOutput directory:")
print(NULL_DIR)