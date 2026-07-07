from aido_bba.config import (
    data_root, brca_output_root, brca_run_dir, go_bp_gmt,
    hgnc_file, ncbi_gene_info, metabric_dir, gse96058_dir,
    kirc_dir, kirc_output_root,
)

# ============================================================
# CELL 9
# Audit TCGA sample types and retain primary tumours
# ============================================================

expression_sample_manifest = pd.DataFrame({
    "sample_id": expression.index
})

expression_sample_manifest[
    "normalized_sample_id"
] = (
    expression_sample_manifest["sample_id"]
    .map(normalize_tcga_barcode)
)

expression_sample_manifest[
    "patient_id"
] = (
    expression_sample_manifest[
        "normalized_sample_id"
    ]
    .map(tcga_patient_id)
)

expression_sample_manifest[
    "sample_type_code"
] = (
    expression_sample_manifest[
        "normalized_sample_id"
    ]
    .map(tcga_sample_type_code)
)

expression_sample_manifest[
    "is_primary_tumour"
] = (
    expression_sample_manifest[
        "sample_type_code"
    ] == "01"
)

print("=" * 72)
print("EXPRESSION SAMPLE-TYPE AUDIT")
print("=" * 72)

print("\nExpression sample-type counts:")
display(
    expression_sample_manifest[
        "sample_type_code"
    ]
    .value_counts(dropna=False)
    .rename_axis("sample_type_code")
    .reset_index(name="n")
)

print("\nTotal expression samples:")
print(len(expression_sample_manifest))

print("\nPrimary-tumour expression samples:")
print(
    int(
        expression_sample_manifest[
            "is_primary_tumour"
        ].sum()
    )
)

# ------------------------------------------------------------
# Primary-tumour records only
# ------------------------------------------------------------

primary_sample_manifest = (
    expression_sample_manifest[
        expression_sample_manifest[
            "is_primary_tumour"
        ]
    ]
    .copy()
)

# ------------------------------------------------------------
# Audit multiple primary samples belonging to the same patient
# ------------------------------------------------------------

primary_patient_counts = (
    primary_sample_manifest
    .groupby(
        "patient_id",
        as_index=False
    )
    .agg(
        n_primary_samples=(
            "normalized_sample_id",
            "size"
        )
    )
)

duplicate_primary_patients = (
    primary_patient_counts[
        primary_patient_counts[
            "n_primary_samples"
        ] > 1
    ]
    .copy()
)

print(
    "\nPatients with multiple GE primary-tumour samples:",
    len(duplicate_primary_patients)
)

if len(duplicate_primary_patients) > 0:

    duplicate_primary_details = (
        primary_sample_manifest.merge(
            duplicate_primary_patients[
                ["patient_id"]
            ],
            on="patient_id",
            how="inner"
        )
        .sort_values(
            ["patient_id", "normalized_sample_id"]
        )
    )

    display(
        duplicate_primary_details.head(30)
    )

else:

    duplicate_primary_details = (
        primary_sample_manifest.iloc[0:0]
        .copy()
    )

duplicate_primary_details.to_csv(
    DIRS["harmonization"]
    / "duplicate_primary_expression_samples.tsv",
    sep="\t",
    index=False
)

# ------------------------------------------------------------
# Deterministic sample selection
#
# Priority:
#   alphabetically first primary sample per patient
# ------------------------------------------------------------

primary_sample_manifest = (
    primary_sample_manifest
    .sort_values(
        [
            "patient_id",
            "normalized_sample_id"
        ]
    )
    .drop_duplicates(
        subset=["patient_id"],
        keep="first"
    )
    .reset_index(drop=True)
)

selected_primary_sample_ids = (
    primary_sample_manifest[
        "normalized_sample_id"
    ]
    .tolist()
)

expression_primary = expression.loc[
    selected_primary_sample_ids
].copy()

print("\nFinal unique primary-tumour patients in GE:")
print(expression_primary.shape[0])

print("\nPrimary expression matrix:")
print(expression_primary.shape)

# ------------------------------------------------------------
# Save manifests
# ------------------------------------------------------------

expression_sample_manifest.to_csv(
    DIRS["harmonization"]
    / "expression_sample_manifest.tsv",
    sep="\t",
    index=False
)

primary_patient_counts.to_csv(
    DIRS["harmonization"]
    / "primary_expression_patient_counts.tsv",
    sep="\t",
    index=False
)

primary_sample_manifest.to_csv(
    DIRS["harmonization"]
    / "selected_primary_expression_samples.tsv",
    sep="\t",
    index=False
)

logger.info(
    "Primary-tumour GE matrix completed: %s patients x %s genes.",
    expression_primary.shape[0],
    expression_primary.shape[1]
)