from aido_bba.config import (
    data_root, brca_output_root, brca_run_dir, go_bp_gmt,
    hgnc_file, ncbi_gene_info, metabric_dir, gse96058_dir,
    kirc_dir, kirc_output_root,
)

# ============================================================
# CELL 4
# Read and inspect BRCA stage endpoint file
# ============================================================

stage_raw = read_table_auto(STAGE_FILE)
stage_raw = clean_column_names(stage_raw)

logger.info(
    "Stage file loaded: %s rows x %s columns",
    stage_raw.shape[0],
    stage_raw.shape[1]
)

print("=" * 70)
print("STAGE FILE DIAGNOSTIC")
print("=" * 70)

print("Stage file path:")
print(STAGE_FILE)

print("\nStage file dimensions:")
print(stage_raw.shape)

print("\nStage file columns:")
for index, column in enumerate(stage_raw.columns):
    print(f"{index:>3} | {repr(column)}")

print("\nFirst 10 rows:")
display(stage_raw.head(10))

print("\nData types:")
display(
    stage_raw.dtypes
    .astype(str)
    .rename("dtype")
    .reset_index()
    .rename(columns={"index": "column"})
)

print("\nMissing-value counts:")
display(
    stage_raw.isna()
    .sum()
    .sort_values(ascending=False)
    .rename("n_missing")
    .reset_index()
    .rename(columns={"index": "column"})
    .head(30)
)

# Save diagnostic output
stage_raw.head(100).to_csv(
    DIRS["input_audit"] / "stage_file_raw_preview.tsv",
    sep="\t",
    index=False
)

stage_column_diagnostics = pd.DataFrame({
    "column_index": range(len(stage_raw.columns)),
    "column_name": stage_raw.columns,
    "dtype": [
        str(stage_raw[column].dtype)
        for column in stage_raw.columns
    ],
    "n_nonmissing": [
        int(stage_raw[column].notna().sum())
        for column in stage_raw.columns
    ],
    "n_unique": [
        int(stage_raw[column].nunique(dropna=True))
        for column in stage_raw.columns
    ]
})

stage_column_diagnostics.to_csv(
    DIRS["input_audit"] / "stage_column_diagnostics.tsv",
    sep="\t",
    index=False
)

logger.info("Stage-file diagnostic completed.")