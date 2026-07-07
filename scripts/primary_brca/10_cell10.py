from aido_bba.config import (
    data_root, brca_output_root, brca_run_dir, go_bp_gmt,
    hgnc_file, ncbi_gene_info, metabric_dir, gse96058_dir,
    kirc_dir, kirc_output_root,
)

# ============================================================
# CELL 10
# Match primary-tumour expression with patient-level endpoint
# ============================================================

# ------------------------------------------------------------
# 1. Prepare expression-side metadata
# ------------------------------------------------------------

expression_primary_metadata = (
    primary_sample_manifest[
        [
            "normalized_sample_id",
            "patient_id",
            "sample_type_code"
        ]
    ]
    .rename(
        columns={
            "normalized_sample_id": "expression_sample_id"
        }
    )
    .copy()
)

# Check uniqueness
assert (
    expression_primary_metadata["patient_id"].is_unique
), "Expression patient IDs are not unique."

assert (
    stage_patient["patient_id"].is_unique
), "Stage patient IDs are not unique."


# ------------------------------------------------------------
# 2. Match expression and endpoint
# ------------------------------------------------------------

analysis_cohort = expression_primary_metadata.merge(
    stage_patient[
        [
            "patient_id",
            "sample_id",
            "stage_raw",
            "stage_group",
            "stage_label",
            "endpoint_source"
        ]
    ].rename(
        columns={
            "sample_id": "stage_sample_id"
        }
    ),
    on="patient_id",
    how="inner",
    validate="one_to_one"
)

analysis_cohort = (
    analysis_cohort
    .sort_values("patient_id")
    .reset_index(drop=True)
)


# ------------------------------------------------------------
# 3. Audit unmatched endpoint patients
# ------------------------------------------------------------

stage_without_expression = (
    stage_patient[
        ~stage_patient["patient_id"].isin(
            expression_primary_metadata["patient_id"]
        )
    ]
    .copy()
    .sort_values("patient_id")
)

expression_without_stage = (
    expression_primary_metadata[
        ~expression_primary_metadata["patient_id"].isin(
            stage_patient["patient_id"]
        )
    ]
    .copy()
    .sort_values("patient_id")
)


# ------------------------------------------------------------
# 4. Add matching diagnostics
# ------------------------------------------------------------

analysis_cohort["sample_id_exact_match"] = (
    analysis_cohort["expression_sample_id"]
    == analysis_cohort["stage_sample_id"]
)

analysis_cohort["patient_id_from_expression_sample"] = (
    analysis_cohort["expression_sample_id"]
    .map(tcga_patient_id)
)

analysis_cohort["patient_id_consistent"] = (
    analysis_cohort["patient_id_from_expression_sample"]
    == analysis_cohort["patient_id"]
)


# ------------------------------------------------------------
# 5. Cohort summary
# ------------------------------------------------------------

analysis_cohort_summary = pd.DataFrame([
    {
        "metric": "primary_expression_patients",
        "value": len(expression_primary_metadata)
    },
    {
        "metric": "resolved_stage_patients",
        "value": len(stage_patient)
    },
    {
        "metric": "matched_analysis_patients",
        "value": len(analysis_cohort)
    },
    {
        "metric": "stage_patients_without_primary_expression",
        "value": len(stage_without_expression)
    },
    {
        "metric": "primary_expression_patients_without_stage",
        "value": len(expression_without_stage)
    },
    {
        "metric": "matched_early_patients",
        "value": int(
            (analysis_cohort["stage_label"] == 0).sum()
        )
    },
    {
        "metric": "matched_advanced_patients",
        "value": int(
            (analysis_cohort["stage_label"] == 1).sum()
        )
    },
    {
        "metric": "exact_sample_id_matches",
        "value": int(
            analysis_cohort["sample_id_exact_match"].sum()
        )
    },
    {
        "metric": "patient_id_consistent",
        "value": int(
            analysis_cohort["patient_id_consistent"].sum()
        )
    }
])


# ------------------------------------------------------------
# 6. Display results
# ------------------------------------------------------------

print("=" * 72)
print("EXPRESSION–ENDPOINT MATCHING AUDIT")
print("=" * 72)

display(analysis_cohort_summary)

print("\nMatched class counts:")
display(
    analysis_cohort["stage_group"]
    .value_counts(dropna=False)
    .rename_axis("stage_group")
    .reset_index(name="n")
)

print("\nStage patients without primary expression:")
display(
    stage_without_expression[
        [
            "sample_id",
            "patient_id",
            "stage_raw",
            "stage_group",
            "stage_label"
        ]
    ]
)

print("\nPrimary-expression patients without resolved stage:")
display(
    expression_without_stage.head(30)
)

print("\nExact sample-ID agreement:")
display(
    analysis_cohort["sample_id_exact_match"]
    .value_counts(dropna=False)
    .rename_axis("sample_id_exact_match")
    .reset_index(name="n")
)

print("\nMatched cohort preview:")
display(
    analysis_cohort.head(15)
)


# ------------------------------------------------------------
# 7. Save outputs
# ------------------------------------------------------------

analysis_cohort.to_csv(
    DIRS["endpoint"] / "analysis_cohort.tsv",
    sep="\t",
    index=False
)

analysis_cohort_summary.to_csv(
    DIRS["harmonization"]
    / "expression_endpoint_matching_summary.tsv",
    sep="\t",
    index=False
)

stage_without_expression.to_csv(
    DIRS["harmonization"]
    / "stage_patients_without_primary_expression.tsv",
    sep="\t",
    index=False
)

expression_without_stage.to_csv(
    DIRS["harmonization"]
    / "primary_expression_patients_without_stage.tsv",
    sep="\t",
    index=False
)


# ------------------------------------------------------------
# 8. Integrity checks
# ------------------------------------------------------------

if analysis_cohort["stage_label"].nunique() != 2:
    raise ValueError(
        "Matched cohort does not contain both endpoint classes."
    )

if not analysis_cohort["patient_id_consistent"].all():
    raise ValueError(
        "Patient-ID inconsistency detected after matching."
    )

logger.info(
    "Expression-stage matching completed: %s patients; "
    "%s Early; %s Advanced.",
    len(analysis_cohort),
    int((analysis_cohort["stage_label"] == 0).sum()),
    int((analysis_cohort["stage_label"] == 1).sum())
)