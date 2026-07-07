from aido_bba.config import (
    data_root, brca_output_root, brca_run_dir, go_bp_gmt,
    hgnc_file, ncbi_gene_info, metabric_dir, gse96058_dir,
    kirc_dir, kirc_output_root,
)

# ============================================================
# CELL 6
# Resolve endpoint records at patient level
# ============================================================

# ------------------------------------------------------------
# 1. Existing vs reconstructed stage-group disagreement
# ------------------------------------------------------------

stage_definition_disagreements = stage_endpoint[
    stage_endpoint["group_agreement"] == False
].copy()

stage_definition_disagreements.to_csv(
    DIRS["endpoint"] / "stage_definition_disagreements.tsv",
    sep="\t",
    index=False
)

print(
    "Existing/reconstructed disagreements:",
    len(stage_definition_disagreements)
)

if len(stage_definition_disagreements) > 0:
    display(
        stage_definition_disagreements[
            [
                "sample_id",
                "patient_id",
                "stage_raw",
                "stage_group_original",
                "stage_group_existing",
                "stage_group_reconstructed"
            ]
        ].head(30)
    )


# ------------------------------------------------------------
# 2. Retain resolved endpoint rows
# ------------------------------------------------------------

stage_resolved = stage_endpoint[
    stage_endpoint["stage_label"].notna()
    & stage_endpoint["patient_id"].notna()
].copy()

stage_resolved["stage_label"] = (
    stage_resolved["stage_label"]
    .astype(int)
)


# ------------------------------------------------------------
# 3. Detect patient-level label conflicts
# ------------------------------------------------------------

patient_label_counts = (
    stage_resolved
    .groupby("patient_id", as_index=False)
    .agg(
        n_records=("stage_label", "size"),
        n_unique_stage_labels=("stage_label", "nunique"),
        n_unique_stage_groups=("stage_group", "nunique")
    )
)

conflicting_patient_ids = patient_label_counts.loc[
    patient_label_counts["n_unique_stage_labels"] > 1,
    "patient_id"
].tolist()

patient_stage_conflicts = stage_resolved[
    stage_resolved["patient_id"].isin(conflicting_patient_ids)
].sort_values(
    ["patient_id", "sample_id"]
)

patient_stage_conflicts.to_csv(
    DIRS["endpoint"] / "stage_label_conflicts.tsv",
    sep="\t",
    index=False
)

print(
    "Patients with conflicting endpoint labels:",
    len(conflicting_patient_ids)
)

if len(patient_stage_conflicts) > 0:
    display(
        patient_stage_conflicts[
            [
                "sample_id",
                "patient_id",
                "stage_raw",
                "stage_group",
                "stage_label"
            ]
        ].head(30)
    )


# ------------------------------------------------------------
# 4. Exclude conflicting patients
# ------------------------------------------------------------

stage_resolved_clean = stage_resolved[
    ~stage_resolved["patient_id"].isin(conflicting_patient_ids)
].copy()


# ------------------------------------------------------------
# 5. Detect primary-tumour records
# ------------------------------------------------------------

stage_resolved_clean["sample_type_code"] = (
    stage_resolved_clean["sample_id"]
    .map(tcga_sample_type_code)
)

stage_resolved_clean["is_primary_tumour"] = (
    stage_resolved_clean["sample_type_code"] == "01"
)

print("\nResolved endpoint records by sample type:")

display(
    stage_resolved_clean["sample_type_code"]
    .value_counts(dropna=False)
    .rename_axis("sample_type_code")
    .reset_index(name="n")
)


# ------------------------------------------------------------
# 6. Deterministic patient-level record selection
# ------------------------------------------------------------
#
# Priority:
#   1. Primary solid tumour sample (01)
#   2. Alphabetically first sample ID
#
# No endpoint label is force-resolved.
# ------------------------------------------------------------

stage_resolved_clean = stage_resolved_clean.sort_values(
    [
        "patient_id",
        "is_primary_tumour",
        "sample_id"
    ],
    ascending=[
        True,
        False,
        True
    ]
)

stage_patient = (
    stage_resolved_clean
    .drop_duplicates(
        subset=["patient_id"],
        keep="first"
    )
    .reset_index(drop=True)
)


# ------------------------------------------------------------
# 7. Unresolved rows
# ------------------------------------------------------------

stage_unresolved = stage_endpoint[
    stage_endpoint["stage_label"].isna()
].copy()

stage_unresolved.to_csv(
    DIRS["endpoint"] / "stage_unresolved_records.tsv",
    sep="\t",
    index=False
)


# ------------------------------------------------------------
# 8. Duplicate record audit
# ------------------------------------------------------------

patient_record_audit = (
    stage_resolved
    .groupby("patient_id", as_index=False)
    .agg(
        n_stage_records=("sample_id", "size"),
        n_unique_samples=("sample_id", "nunique"),
        n_unique_labels=("stage_label", "nunique")
    )
)

patient_record_audit["has_multiple_records"] = (
    patient_record_audit["n_stage_records"] > 1
)

patient_record_audit.to_csv(
    DIRS["endpoint"] / "patient_stage_record_audit.tsv",
    sep="\t",
    index=False
)

print(
    "\nPatients with multiple resolved stage records:",
    int(patient_record_audit["has_multiple_records"].sum())
)


# ------------------------------------------------------------
# 9. Save patient-level endpoint
# ------------------------------------------------------------

stage_patient.to_csv(
    DIRS["endpoint"] / "stage_endpoint_manifest.tsv",
    sep="\t",
    index=False
)


# ------------------------------------------------------------
# 10. Summary
# ------------------------------------------------------------

stage_endpoint_summary = pd.DataFrame([
    {
        "metric": "raw_records",
        "value": len(stage_endpoint)
    },
    {
        "metric": "resolved_records",
        "value": len(stage_resolved)
    },
    {
        "metric": "unresolved_records",
        "value": len(stage_unresolved)
    },
    {
        "metric": "definition_disagreements",
        "value": len(stage_definition_disagreements)
    },
    {
        "metric": "patients_with_conflicting_labels",
        "value": len(conflicting_patient_ids)
    },
    {
        "metric": "final_unique_patients",
        "value": len(stage_patient)
    },
    {
        "metric": "final_early_patients",
        "value": int(
            (stage_patient["stage_label"] == 0).sum()
        )
    },
    {
        "metric": "final_advanced_patients",
        "value": int(
            (stage_patient["stage_label"] == 1).sum()
        )
    }
])

stage_endpoint_summary.to_csv(
    DIRS["endpoint"] / "stage_endpoint_summary.tsv",
    sep="\t",
    index=False
)


print("\n" + "=" * 72)
print("PATIENT-LEVEL ENDPOINT SUMMARY")
print("=" * 72)

display(stage_endpoint_summary)

print("\nFinal patient-level class counts:")

display(
    stage_patient["stage_group"]
    .value_counts(dropna=False)
    .rename_axis("stage_group")
    .reset_index(name="n")
)

print("\nPatient-level endpoint preview:")

display(
    stage_patient[
        [
            "sample_id",
            "patient_id",
            "stage_raw",
            "stage_group",
            "stage_label",
            "endpoint_source",
            "sample_type_code",
            "is_primary_tumour"
        ]
    ].head(15)
)

logger.info(
    "Stage endpoint completed: %s unique patients; "
    "%s Early; %s Advanced.",
    len(stage_patient),
    int((stage_patient["stage_label"] == 0).sum()),
    int((stage_patient["stage_label"] == 1).sum())
)