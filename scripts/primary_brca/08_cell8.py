from aido_bba.config import (
    data_root, brca_output_root, brca_run_dir, go_bp_gmt,
    hgnc_file, ncbi_gene_info, metabric_dir, gse96058_dir,
    kirc_dir, kirc_output_root,
)

# ============================================================
# CELL 8
# Convert GE data into samples x genes
# ============================================================

if orientation == "genes_by_samples":

    original_gene_column = ge_raw.columns[0]

    print("Detected gene-identifier column:")
    print(repr(original_gene_column))

    ge_work = ge_raw.rename(
        columns={
            original_gene_column: "gene_id"
        }
    ).copy()

    # --------------------------------------------------------
    # Clean gene identifiers
    # --------------------------------------------------------

    ge_work["gene_id"] = (
        ge_work["gene_id"]
        .astype(str)
        .str.strip()
    )

    invalid_gene_id_mask = (
        ge_work["gene_id"].isna()
        | ge_work["gene_id"].str.lower().isin(
            [
                "",
                "nan",
                "none",
                "null",
                "na"
            ]
        )
    )

    n_invalid_gene_ids = int(
        invalid_gene_id_mask.sum()
    )

    ge_work = ge_work.loc[
        ~invalid_gene_id_mask
    ].copy()

    sample_columns = [
        column
        for column in ge_work.columns
        if column != "gene_id"
    ]

    print("\nNumber of raw expression sample columns:")
    print(len(sample_columns))

    # --------------------------------------------------------
    # Convert expression values to numeric
    # --------------------------------------------------------

    expression_numeric = ge_work[
        sample_columns
    ].apply(
        pd.to_numeric,
        errors="coerce"
    )

    expression_numeric.index = (
        ge_work["gene_id"].values
    )

    # --------------------------------------------------------
    # Audit duplicated gene identifiers
    # --------------------------------------------------------

    duplicated_gene_mask = (
        expression_numeric.index.duplicated(
            keep=False
        )
    )

    duplicate_gene_entry_count = int(
        duplicated_gene_mask.sum()
    )

    duplicated_gene_ids = pd.Index(
        expression_numeric.index[
            duplicated_gene_mask
        ]
    ).unique()

    unique_duplicated_gene_count = int(
        len(duplicated_gene_ids)
    )

    duplicate_gene_audit = pd.DataFrame({
        "gene_id": duplicated_gene_ids
    })

    if len(duplicate_gene_audit) > 0:

        duplicate_gene_audit[
            "n_occurrences"
        ] = duplicate_gene_audit[
            "gene_id"
        ].map(
            pd.Series(
                expression_numeric.index
            ).value_counts()
        )

    duplicate_gene_audit.to_csv(
        DIRS["harmonization"]
        / "duplicate_gene_identifier_audit.tsv",
        sep="\t",
        index=False
    )

    # --------------------------------------------------------
    # Aggregate duplicate genes by arithmetic mean
    # --------------------------------------------------------

    expression_gene_by_sample = (
        expression_numeric
        .groupby(
            level=0,
            sort=False
        )
        .mean()
    )

    expression = (
        expression_gene_by_sample.T
    )

    expression.index.name = "sample_id"


else:

    original_sample_column = ge_raw.columns[0]

    print("Detected sample-identifier column:")
    print(repr(original_sample_column))

    expression = ge_raw.rename(
        columns={
            original_sample_column: "sample_id"
        }
    ).copy()

    expression["sample_id"] = (
        expression["sample_id"]
        .astype(str)
        .str.strip()
    )

    expression = expression.set_index(
        "sample_id"
    )

    expression = expression.apply(
        pd.to_numeric,
        errors="coerce"
    )

    duplicated_gene_mask = (
        expression.columns.duplicated(
            keep=False
        )
    )

    duplicate_gene_entry_count = int(
        duplicated_gene_mask.sum()
    )

    duplicated_gene_ids = pd.Index(
        expression.columns[
            duplicated_gene_mask
        ]
    ).unique()

    unique_duplicated_gene_count = int(
        len(duplicated_gene_ids)
    )

    n_invalid_gene_ids = 0

    duplicate_gene_audit = pd.DataFrame({
        "gene_id": duplicated_gene_ids
    })

    duplicate_gene_audit.to_csv(
        DIRS["harmonization"]
        / "duplicate_gene_identifier_audit.tsv",
        sep="\t",
        index=False
    )

    expression = (
        expression.T
        .groupby(
            level=0,
            sort=False
        )
        .mean()
        .T
    )


# ------------------------------------------------------------
# Normalize TCGA sample barcodes
# ------------------------------------------------------------

expression.index = [
    normalize_tcga_barcode(sample)
    for sample in expression.index
]


# ------------------------------------------------------------
# Audit duplicated sample identifiers
# ------------------------------------------------------------

duplicated_sample_mask = (
    pd.Index(expression.index)
    .duplicated(keep=False)
)

duplicate_sample_entry_count = int(
    duplicated_sample_mask.sum()
)

if duplicate_sample_entry_count > 0:

    duplicate_sample_ids = pd.Index(
        expression.index[
            duplicated_sample_mask
        ]
    ).unique()

    duplicate_sample_audit = pd.DataFrame({
        "sample_id": duplicate_sample_ids
    })

    duplicate_sample_audit[
        "n_occurrences"
    ] = duplicate_sample_audit[
        "sample_id"
    ].map(
        pd.Series(
            expression.index
        ).value_counts()
    )

else:

    duplicate_sample_audit = pd.DataFrame(
        columns=[
            "sample_id",
            "n_occurrences"
        ]
    )

duplicate_sample_audit.to_csv(
    DIRS["harmonization"]
    / "duplicate_expression_sample_ids.tsv",
    sep="\t",
    index=False
)


# ------------------------------------------------------------
# Remove genes with all values missing
# ------------------------------------------------------------

all_missing_gene_mask = (
    expression.isna().all(axis=0)
)

n_all_missing_genes = int(
    all_missing_gene_mask.sum()
)

expression = expression.loc[
    :,
    ~all_missing_gene_mask
].copy()


# ------------------------------------------------------------
# Audit genes with partial missingness
# ------------------------------------------------------------

gene_missingness = (
    expression.isna()
    .mean(axis=0)
    .rename("missing_fraction")
    .reset_index()
    .rename(columns={"index": "gene_id"})
)

gene_missingness[
    "n_missing"
] = (
    expression.isna()
    .sum(axis=0)
    .values
)

gene_missingness = gene_missingness.sort_values(
    [
        "missing_fraction",
        "gene_id"
    ],
    ascending=[
        False,
        True
    ]
)

gene_missingness.to_csv(
    DIRS["harmonization"]
    / "gene_missingness_audit.tsv",
    sep="\t",
    index=False
)


# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

remaining_missing_values = int(
    expression.isna().sum().sum()
)

genes_with_any_missing = int(
    (gene_missingness["n_missing"] > 0).sum()
)

print("\n" + "=" * 72)
print("EXPRESSION MATRIX SUMMARY")
print("=" * 72)

print("\nSamples x genes:")
print(expression.shape)

print("\nInvalid gene identifiers removed:")
print(n_invalid_gene_ids)

print("\nDuplicate gene entries before aggregation:")
print(duplicate_gene_entry_count)

print("\nUnique duplicated gene identifiers:")
print(unique_duplicated_gene_count)

print("\nDuplicate sample entries:")
print(duplicate_sample_entry_count)

print("\nAll-missing genes removed:")
print(n_all_missing_genes)

print("\nGenes with any remaining missing values:")
print(genes_with_any_missing)

print("\nTotal remaining missing values:")
print(remaining_missing_values)

print("\nFirst five normalized sample IDs:")
print(expression.index[:5].tolist())

print("\nFirst ten gene identifiers:")
print(expression.columns[:10].tolist())


expression_conversion_summary = pd.DataFrame([
    {
        "metric": "raw_rows",
        "value": int(ge_raw.shape[0])
    },
    {
        "metric": "raw_columns",
        "value": int(ge_raw.shape[1])
    },
    {
        "metric": "raw_expression_samples",
        "value": int(len(sample_columns))
        if orientation == "genes_by_samples"
        else int(ge_raw.shape[0])
    },
    {
        "metric": "invalid_gene_identifiers_removed",
        "value": n_invalid_gene_ids
    },
    {
        "metric": "duplicate_gene_entries_before_aggregation",
        "value": duplicate_gene_entry_count
    },
    {
        "metric": "unique_duplicated_gene_identifiers",
        "value": unique_duplicated_gene_count
    },
    {
        "metric": "duplicate_sample_entries",
        "value": duplicate_sample_entry_count
    },
    {
        "metric": "all_missing_genes_removed",
        "value": n_all_missing_genes
    },
    {
        "metric": "final_samples",
        "value": int(expression.shape[0])
    },
    {
        "metric": "final_unique_genes",
        "value": int(expression.shape[1])
    },
    {
        "metric": "genes_with_any_missing_values",
        "value": genes_with_any_missing
    },
    {
        "metric": "remaining_missing_values",
        "value": remaining_missing_values
    }
])

display(expression_conversion_summary)

if unique_duplicated_gene_count > 0:

    print("\nDuplicated gene identifiers:")
    display(
        duplicate_gene_audit.head(30)
    )

if genes_with_any_missing > 0:

    print("\nGenes with the highest missing fractions:")
    display(
        gene_missingness.head(20)
    )


expression_conversion_summary.to_csv(
    DIRS["input_audit"]
    / "expression_conversion_summary.tsv",
    sep="\t",
    index=False
)

logger.info(
    "Expression matrix constructed: %s samples x %s genes; "
    "%s unique duplicated gene identifiers aggregated.",
    expression.shape[0],
    expression.shape[1],
    unique_duplicated_gene_count
)