# ============================================================
# AIDO-BBA BRCA 1.0
# CELL 29B
# Resume CELL 29 after harmonized_gene_id suffix error
#
# Do NOT rerun CELL 29.
# Run this cell directly in the same Jupyter session.
# ============================================================

from pathlib import Path
from aido_bba.config import (
    data_root, brca_output_root, brca_run_dir, go_bp_gmt,
    hgnc_file, ncbi_gene_info, metabric_dir, gse96058_dir,
    kirc_dir, kirc_output_root,
)

import numpy as np
import pandas as pd
import json


print("=" * 80)
print("AIDO-BBA CELL 29B — RESUME RECOMMENDATION OUTPUT")
print("=" * 80)


# ============================================================
# 1. VERIFY REQUIRED IN-MEMORY OBJECTS
# ============================================================

required_objects = [
    "recommended_patient_genes",
    "patient_recommendation_report",
    "patient_core_recommendations",
    "primary_core_recommendations",
    "boundary_patient_ids",
    "RECOMMENDATION_DIR",
    "RECOMMENDATION_SUMMARY_DIR",
    "RECOMMENDATION_REPORT_DIR",
    "TOP_GENES_PER_PATIENT_REPORT"
]

missing_objects = [
    object_name
    for object_name in required_objects
    if object_name not in globals()
]

if missing_objects:

    raise RuntimeError(
        "Required objects are not available in memory:\n"
        + "\n".join(
            missing_objects
        )
        + "\n\nRun CELL 29 once until its original "
        "harmonized_gene_id error, then execute CELL 29B."
    )


print("\nRecommended patient-gene rows:")
print(
    recommended_patient_genes.shape
)

print("\nAvailable recommended-patient-gene columns:")
print(
    recommended_patient_genes.columns.tolist()
)


# ============================================================
# 2. ROBUSTLY IDENTIFY GENE COLUMNS
# ============================================================

def first_existing_column(
    dataframe,
    candidates,
    required=True
):

    for candidate in candidates:

        if candidate in dataframe.columns:

            return candidate

    if required:

        raise KeyError(
            "None of the expected columns were found:\n"
            + "\n".join(
                candidates
            )
        )

    return None


raw_gene_column = first_existing_column(
    recommended_patient_genes,
    [
        "raw_gene_id",
        "gene_id",
        "raw_gene_id_x",
        "raw_gene_id_y"
    ]
)

harmonized_gene_column = first_existing_column(
    recommended_patient_genes,
    [
        "harmonized_gene_id_y",
        "harmonized_gene_id",
        "harmonized_gene_id_x"
    ],
    required=False
)

core_name_column = first_existing_column(
    recommended_patient_genes,
    [
        "core_module_name",
        "core_module_name_x",
        "core_module_name_y"
    ]
)

patient_signed_column = first_existing_column(
    recommended_patient_genes,
    [
        "pipeline_mean_signed_shap_x",
        "pipeline_mean_signed_shap",
        "conditional_mean_signed_shap"
    ],
    required=False
)

patient_abs_column = first_existing_column(
    recommended_patient_genes,
    [
        "pipeline_mean_absolute_shap_x",
        "pipeline_mean_absolute_shap",
        "conditional_mean_absolute_shap"
    ],
    required=False
)

print("\nResolved columns:")
print("Raw gene:", raw_gene_column)
print("Harmonized gene:", harmonized_gene_column)
print("Core:", core_name_column)
print("Patient signed attribution:", patient_signed_column)
print("Patient absolute attribution:", patient_abs_column)


# ============================================================
# 3. CLEAN HARMONIZED GENE COLUMN
# ============================================================

if harmonized_gene_column is None:

    recommended_patient_genes[
        "resolved_harmonized_gene_id"
    ] = recommended_patient_genes[
        raw_gene_column
    ].astype(str)

else:

    recommended_patient_genes[
        "resolved_harmonized_gene_id"
    ] = (
        recommended_patient_genes[
            harmonized_gene_column
        ]
        .where(
            recommended_patient_genes[
                harmonized_gene_column
            ].notna(),
            recommended_patient_genes[
                raw_gene_column
            ]
        )
        .astype(str)
    )

recommended_patient_genes[
    "resolved_raw_gene_id"
] = (
    recommended_patient_genes[
        raw_gene_column
    ]
    .astype(str)
)

recommended_patient_genes[
    "resolved_core_module_name"
] = (
    recommended_patient_genes[
        core_name_column
    ]
    .astype(str)
)


# ============================================================
# 4. SUMMARIZE TOP GENES PER PATIENT
# ============================================================

patient_gene_text_records = []

recommended_patient_genes_sorted = (
    recommended_patient_genes
    .sort_values(
        [
            "patient_id",
            "gene_rank_within_patient"
        ]
    )
    .copy()
)

for patient_id, patient_df in (
    recommended_patient_genes_sorted
    .groupby(
        "patient_id",
        sort=False
    )
):

    gene_labels = []

    for _, row in patient_df.iterrows():

        raw_gene = str(
            row[
                "resolved_raw_gene_id"
            ]
        )

        harmonized_gene = str(
            row[
                "resolved_harmonized_gene_id"
            ]
        )

        core_name = str(
            row[
                "resolved_core_module_name"
            ]
        )

        invalid_harmonized_values = {
            "",
            "nan",
            "none",
            "null",
            "na"
        }

        if (
            harmonized_gene.lower()
            not in invalid_harmonized_values
            and
            harmonized_gene != raw_gene
        ):

            gene_label = (
                f"{raw_gene}->{harmonized_gene}"
            )

        else:

            gene_label = raw_gene

        gene_labels.append(
            f"{gene_label}[{core_name}]"
        )

    patient_gene_text_records.append({
        "patient_id":
            str(
                patient_id
            ),

        "recommended_gap_genes":
            "; ".join(
                gene_labels
            ),

        "n_recommended_gap_genes":
            len(
                gene_labels
            )
    })


patient_gene_text = pd.DataFrame(
    patient_gene_text_records
)

print("\nPatient gene summaries:")
print(
    patient_gene_text.shape
)

display(
    patient_gene_text.head()
)


# ============================================================
# 5. REMOVE OLD PARTIAL COLUMNS BEFORE MERGE
# ============================================================

columns_to_remove = [
    column
    for column in [
        "recommended_gap_genes",
        "n_recommended_gap_genes"
    ]
    if column in patient_recommendation_report.columns
]

if columns_to_remove:

    patient_recommendation_report = (
        patient_recommendation_report.drop(
            columns=columns_to_remove
        )
    )


patient_recommendation_report = (
    patient_recommendation_report.merge(
        patient_gene_text,
        on="patient_id",
        how="left",
        validate="one_to_one"
    )
)

patient_recommendation_report[
    "recommended_gap_genes"
] = (
    patient_recommendation_report[
        "recommended_gap_genes"
    ]
    .fillna(
        ""
    )
)

patient_recommendation_report[
    "n_recommended_gap_genes"
] = (
    patient_recommendation_report[
        "n_recommended_gap_genes"
    ]
    .fillna(0)
    .astype(int)
)


# ============================================================
# 6. RECOMMENDATION CONFIDENCE
# ============================================================

required_confidence_columns = [
    "repeat_state_consistency",
    "mean_core_membership_uncertainty",
    "primary_core_priority",
    "n_recommended_gap_genes"
]

missing_confidence_columns = [
    column
    for column in required_confidence_columns
    if column not in patient_recommendation_report.columns
]

if missing_confidence_columns:

    raise KeyError(
        "Missing confidence columns:\n"
        + "\n".join(
            missing_confidence_columns
        )
    )


patient_recommendation_report[
    "recommendation_confidence_score"
] = (
    0.30
    * patient_recommendation_report[
        "repeat_state_consistency"
    ]
    +
    0.25
    * (
        1.0
        -
        patient_recommendation_report[
            "mean_core_membership_uncertainty"
        ]
    )
    +
    0.25
    * patient_recommendation_report[
        "primary_core_priority"
    ]
    +
    0.20
    * (
        patient_recommendation_report[
            "n_recommended_gap_genes"
        ]
        .clip(
            upper=TOP_GENES_PER_PATIENT_REPORT
        )
        /
        TOP_GENES_PER_PATIENT_REPORT
    )
)

patient_recommendation_report[
    "recommendation_confidence_tier"
] = pd.cut(
    patient_recommendation_report[
        "recommendation_confidence_score"
    ],
    bins=[
        -np.inf,
        0.55,
        0.70,
        np.inf
    ],
    labels=[
        "limited_confidence",
        "moderate_confidence",
        "higher_confidence"
    ]
)


# ============================================================
# 7. BOUNDARY-PATIENT FINAL REPORT
# ============================================================

boundary_recommendation_report = (
    patient_recommendation_report[
        patient_recommendation_report[
            "patient_id"
        ]
        .astype(str)
        .isin(
            [
                str(patient_id)
                for patient_id
                in boundary_patient_ids
            ]
        )
    ]
    .copy()
)

boundary_order = {
    str(patient_id):
        rank

    for rank, patient_id
    in enumerate(
        boundary_patient_ids,
        start=1
    )
}

boundary_recommendation_report[
    "boundary_priority_rank"
] = (
    boundary_recommendation_report[
        "patient_id"
    ]
    .astype(str)
    .map(
        boundary_order
    )
)

boundary_recommendation_report = (
    boundary_recommendation_report
    .sort_values(
        "boundary_priority_rank"
    )
    .reset_index(
        drop=True
    )
)

print("\nBoundary recommendation reports:")
print(
    boundary_recommendation_report.shape
)


# ============================================================
# 8. RECOMMENDATION TYPE SUMMARY
# ============================================================

recommendation_type_summary = pd.DataFrame([
    {
        "recommendation_type":
            "representation_expansion",

        "n_patients":
            int(
                patient_recommendation_report[
                    "recommend_representation_expansion"
                ].sum()
            )
    },
    {
        "recommendation_type":
            "residual_direction_review",

        "n_patients":
            int(
                patient_recommendation_report[
                    "recommend_residual_direction_review"
                ].sum()
            )
    },
    {
        "recommendation_type":
            "model_arbitration",

        "n_patients":
            int(
                patient_recommendation_report[
                    "recommend_model_arbitration"
                ].sum()
            )
    },
    {
        "recommendation_type":
            "repeat_or_orthogonal_measurement",

        "n_patients":
            int(
                patient_recommendation_report[
                    "recommend_repeat_measurement"
                ].sum()
            )
    },
    {
        "recommendation_type":
            "clinical_molecular_reconciliation",

        "n_patients":
            int(
                patient_recommendation_report[
                    "recommend_clinical_molecular_reconciliation"
                ].sum()
            )
    }
])


# ============================================================
# 9. CORE RECOMMENDATION SUMMARY
# ============================================================

core_recommendation_summary = (
    primary_core_recommendations
    .groupby(
        [
            "core_module_name",
            "core_direction",
            "core_measurement_recommendation"
        ],
        as_index=False
    )
    .agg(
        n_patient_core_records=(
            "patient_id",
            "size"
        ),

        mean_core_priority=(
            "core_resolution_priority",
            "mean"
        ),

        mean_membership_uncertainty=(
            "membership_uncertainty",
            "mean"
        )
    )
)


# ============================================================
# 10. SAVE TABLE OUTPUTS
# ============================================================

RECOMMENDATION_DIR = Path(
    RECOMMENDATION_DIR
)

RECOMMENDATION_SUMMARY_DIR = Path(
    RECOMMENDATION_SUMMARY_DIR
)

RECOMMENDATION_REPORT_DIR = Path(
    RECOMMENDATION_REPORT_DIR
)

for directory in [
    RECOMMENDATION_DIR,
    RECOMMENDATION_SUMMARY_DIR,
    RECOMMENDATION_REPORT_DIR
]:

    directory.mkdir(
        parents=True,
        exist_ok=True
    )


patient_core_recommendations.to_csv(
    RECOMMENDATION_SUMMARY_DIR
    / "patient_core_measurement_recommendations.tsv",
    sep="\t",
    index=False
)

recommended_patient_genes.to_csv(
    RECOMMENDATION_SUMMARY_DIR
    / "patient_gap_gene_measurement_targets.tsv",
    sep="\t",
    index=False
)

patient_recommendation_report.to_csv(
    RECOMMENDATION_SUMMARY_DIR
    / "all_patient_measurement_recommendation_report.tsv",
    sep="\t",
    index=False
)

boundary_recommendation_report.to_csv(
    RECOMMENDATION_SUMMARY_DIR
    / "boundary_patient_measurement_recommendation_top100.tsv",
    sep="\t",
    index=False
)

recommendation_type_summary.to_csv(
    RECOMMENDATION_SUMMARY_DIR
    / "measurement_recommendation_type_summary.tsv",
    sep="\t",
    index=False
)

core_recommendation_summary.to_csv(
    RECOMMENDATION_SUMMARY_DIR
    / "core_measurement_recommendation_summary.tsv",
    sep="\t",
    index=False
)


# ============================================================
# 11. WRITE INDIVIDUAL PATIENT TEXT REPORTS
# ============================================================

n_reports_written = 0

for _, row in (
    boundary_recommendation_report.iterrows()
):

    patient_id = str(
        row[
            "patient_id"
        ]
    )

    primary_core_priority = row.get(
        "primary_core_priority",
        np.nan
    )

    confidence_score = row.get(
        "recommendation_confidence_score",
        np.nan
    )

    primary_core_priority_text = (
        f"{primary_core_priority:.4f}"
        if pd.notna(
            primary_core_priority
        )
        else "NA"
    )

    confidence_score_text = (
        f"{confidence_score:.4f}"
        if pd.notna(
            confidence_score
        )
        else "NA"
    )

    recommended_genes_text = str(
        row.get(
            "recommended_gap_genes",
            ""
        )
    ).strip()

    if (
        recommended_genes_text == ""
        or recommended_genes_text.lower()
        in {
            "nan",
            "none"
        }
    ):

        recommended_genes_text = (
            "No gene-level target passed the current "
            "support and ranking thresholds."
        )

    report_lines = [
        "AIDO-BBA PATIENT-SPECIFIC MEASUREMENT AUDIT",
        "=" * 60,
        "",
        f"Patient: {patient_id}",
        (
            "Clinical group: "
            f"{row.get('true_group', 'NA')}"
        ),
        (
            "Clinical-molecular rank state: "
            f"{row.get('clinical_molecular_rank_state', 'NA')}"
        ),
        (
            "Integrated BBA state: "
            f"{row.get('integrated_bba_state', 'NA')}"
        ),
        (
            "Model-dependence tier: "
            f"{row.get('model_dependence_tier', 'NA')}"
        ),
        (
            "Repeat-instability tier: "
            f"{row.get('repeat_instability_tier', 'NA')}"
        ),
        "",
        "PRIMARY UNRESOLVED EXPLANATORY AXIS",
        "-" * 60,
        (
            "Core: "
            f"{row.get('primary_unresolved_core', 'NA')}"
        ),
        (
            "Direction: "
            f"{row.get('primary_core_direction', 'NA')}"
        ),
        (
            "Core priority: "
            f"{primary_core_priority_text}"
        ),
        (
            "Core recommendation: "
            f"{row.get('primary_core_recommendation', 'NA')}"
        ),
        "",
        "AUDIT RECOMMENDATION",
        "-" * 60,
        str(
            row.get(
                "measurement_recommendation_summary",
                "NA"
            )
        ),
        "",
        "REPRESENTATION-GAP GENE TARGETS",
        "-" * 60,
        recommended_genes_text,
        "",
        "CONFIDENCE",
        "-" * 60,
        (
            "Score: "
            f"{confidence_score_text}"
        ),
        (
            "Tier: "
            f"{row.get('recommendation_confidence_tier', 'NA')}"
        ),
        "",
        "INTERPRETATION BOUNDARY",
        "-" * 60,
        (
            "This report identifies explanatory and measurement "
            "priorities within the current computational audit. "
            "It does not prescribe clinical testing, establish "
            "a biological subtype, infer prognosis, or recommend "
            "treatment."
        )
    ]

    report_path = (
        RECOMMENDATION_REPORT_DIR
        / f"{patient_id}_measurement_audit.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as handle:

        handle.write(
            "\n".join(
                report_lines
            )
        )

    n_reports_written += 1


# ============================================================
# 12. FINAL SUMMARY TABLE
# ============================================================

summary_table = pd.DataFrame([
    {
        "metric":
            "n_patients",

        "value":
            len(
                patient_recommendation_report
            )
    },
    {
        "metric":
            "n_boundary_reports",

        "value":
            len(
                boundary_recommendation_report
            )
    },
    {
        "metric":
            "n_text_reports_written",

        "value":
            n_reports_written
    },
    {
        "metric":
            "n_patient_core_recommendations",

        "value":
            len(
                patient_core_recommendations
            )
    },
    {
        "metric":
            "n_patient_gene_targets",

        "value":
            len(
                recommended_patient_genes
            )
    },
    {
        "metric":
            "patients_with_recommended_gene_targets",

        "value":
            int(
                (
                    patient_recommendation_report[
                        "n_recommended_gap_genes"
                    ]
                    > 0
                ).sum()
            )
    },
    {
        "metric":
            "patients_with_representation_expansion",

        "value":
            int(
                patient_recommendation_report[
                    "recommend_representation_expansion"
                ].sum()
            )
    },
    {
        "metric":
            "patients_with_residual_direction_review",

        "value":
            int(
                patient_recommendation_report[
                    "recommend_residual_direction_review"
                ].sum()
            )
    },
    {
        "metric":
            "patients_with_model_arbitration",

        "value":
            int(
                patient_recommendation_report[
                    "recommend_model_arbitration"
                ].sum()
            )
    },
    {
        "metric":
            "patients_with_repeat_measurement",

        "value":
            int(
                patient_recommendation_report[
                    "recommend_repeat_measurement"
                ].sum()
            )
    },
    {
        "metric":
            "patients_with_clinical_molecular_reconciliation",

        "value":
            int(
                patient_recommendation_report[
                    "recommend_clinical_molecular_reconciliation"
                ].sum()
            )
    },
    {
        "metric":
            "higher_confidence_boundary_reports",

        "value":
            int(
                (
                    boundary_recommendation_report[
                        "recommendation_confidence_tier"
                    ]
                    .astype(str)
                    ==
                    "higher_confidence"
                ).sum()
            )
    },
    {
        "metric":
            "moderate_confidence_boundary_reports",

        "value":
            int(
                (
                    boundary_recommendation_report[
                        "recommendation_confidence_tier"
                    ]
                    .astype(str)
                    ==
                    "moderate_confidence"
                ).sum()
            )
    },
    {
        "metric":
            "limited_confidence_boundary_reports",

        "value":
            int(
                (
                    boundary_recommendation_report[
                        "recommendation_confidence_tier"
                    ]
                    .astype(str)
                    ==
                    "limited_confidence"
                ).sum()
            )
    }
])

summary_table.to_csv(
    RECOMMENDATION_SUMMARY_DIR
    / "missing_measurement_recommendation_summary.tsv",
    sep="\t",
    index=False
)


# ============================================================
# 13. MANIFEST
# ============================================================

manifest = {
    "analysis":
        (
            "Patient-specific missing-measurement "
            "recommendation audit"
        ),

    "run_directory":
        str(
            RUN_DIR
        )
        if "RUN_DIR" in globals()
        else "in-memory continuation",

    "n_boundary_reports":
        int(
            len(
                boundary_recommendation_report
            )
        ),

    "n_text_reports_written":
        int(
            n_reports_written
        ),

    "resolved_raw_gene_column":
        raw_gene_column,

    "resolved_harmonized_gene_column":
        harmonized_gene_column,

    "resolved_core_column":
        core_name_column,

    "recommendation_classes": [
        "representation expansion",
        "residual-direction review",
        "model arbitration",
        "repeat or orthogonal measurement",
        "clinical-molecular reconciliation"
    ],

    "interpretation_boundary":
        (
            "Recommendations identify computational measurement "
            "priorities. They are not clinical prescriptions, "
            "diagnoses, prognostic conclusions, or treatment "
            "recommendations."
        )
}

with open(
    RECOMMENDATION_DIR
    / "missing_measurement_recommendation_manifest.json",
    "w",
    encoding="utf-8"
) as handle:

    json.dump(
        manifest,
        handle,
        indent=2
    )


# ============================================================
# 14. FINAL DISPLAY
# ============================================================

print("\n" + "=" * 80)
print("CELL 29B COMPLETED")
print("=" * 80)

display(
    summary_table
)

print("\nRecommendation types:")

display(
    recommendation_type_summary
)

print("\nCore recommendation summary:")

display(
    core_recommendation_summary
    .sort_values(
        [
            "mean_core_priority",
            "n_patient_core_records"
        ],
        ascending=[
            False,
            False
        ]
    )
)

print("\nTop boundary measurement recommendations:")

boundary_display_columns = [
    column
    for column in [
        "boundary_priority_rank",
        "patient_id",
        "true_group",
        "integrated_bba_state",
        "clinical_molecular_rank_state",
        "model_dependence_tier",
        "repeat_instability_tier",
        "primary_unresolved_core",
        "primary_core_direction",
        "primary_core_priority",
        "measurement_recommendation_summary",
        "recommended_gap_genes",
        "recommendation_confidence_score",
        "recommendation_confidence_tier"
    ]
    if column
    in boundary_recommendation_report.columns
]

display(
    boundary_recommendation_report[
        boundary_display_columns
    ].head(40)
)

print("\nHighest-priority patient-gene targets:")

gene_display_columns = [
    column
    for column in [
        "patient_id",
        "resolved_core_module_name",
        "resolved_raw_gene_id",
        "resolved_harmonized_gene_id",
        patient_signed_column,
        patient_abs_column,
        "repeat_selection_frequency",
        "attribution_sign_consistency",
        "fold_selection_frequency",
        "consensus_margin",
        "core_direction",
        "combined_patient_gene_priority",
        "gene_rank_within_patient"
    ]
    if (
        column is not None
        and
        column
        in recommended_patient_genes.columns
    )
]

display(
    recommended_patient_genes[
        gene_display_columns
    ]
    .sort_values(
        [
            "patient_id",
            "gene_rank_within_patient"
        ]
    )
    .head(100)
)

print("\nOutput directory:")
print(
    RECOMMENDATION_DIR
)

print("\nIndividual reports written:")
print(
    n_reports_written
)