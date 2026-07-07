from aido_bba.config import (
    data_root, brca_output_root, brca_run_dir, go_bp_gmt,
    hgnc_file, ncbi_gene_info, metabric_dir, gse96058_dir,
    kirc_dir, kirc_output_root,
)

# ============================================================
# CELL 7
# Load and diagnose TCGA-BRCA gene-expression matrix
# ============================================================

logger.info("Reading expression file: %s", GE_FILE)

ge_raw = pd.read_csv(
    GE_FILE,
    sep="\t",
    low_memory=False
)

ge_raw = clean_column_names(ge_raw)

print("=" * 72)
print("GENE-EXPRESSION FILE DIAGNOSTIC")
print("=" * 72)

print("\nGE file path:")
print(GE_FILE)

print("\nRaw GE dimensions:")
print(ge_raw.shape)

print("\nFirst 15 column names:")
for index, column in enumerate(ge_raw.columns[:15]):
    print(f"{index:>3} | {repr(column)}")

print("\nFirst five rows and first eight columns:")
display(ge_raw.iloc[:5, :8])

print("\nFirst-column preview:")
display(
    ge_raw.iloc[:15, [0]]
)

logger.info(
    "Raw GE matrix loaded: %s rows x %s columns.",
    ge_raw.shape[0],
    ge_raw.shape[1]
)


def detect_expression_orientation(df):
    """
    Detect whether the expression matrix is:

    1. genes_by_samples:
       rows = genes
       columns = TCGA samples

    2. samples_by_genes:
       rows = TCGA samples
       columns = genes
    """

    column_names = pd.Series(
        [str(column).strip().upper() for column in df.columns]
    )

    column_tcga_fraction = (
        column_names.str.startswith("TCGA-").mean()
    )

    first_column_values = (
        df.iloc[:, 0]
        .dropna()
        .astype(str)
        .str.strip()
        .str.upper()
        .head(2000)
    )

    first_column_tcga_fraction = (
        first_column_values
        .str.startswith("TCGA-")
        .mean()
        if len(first_column_values) > 0
        else 0
    )

    diagnostics = {
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "column_tcga_fraction": float(column_tcga_fraction),
        "first_column_tcga_fraction": float(
            first_column_tcga_fraction
        )
    }

    if column_tcga_fraction >= 0.20:
        orientation = "genes_by_samples"

    elif first_column_tcga_fraction >= 0.20:
        orientation = "samples_by_genes"

    else:
        orientation = "uncertain"

    return orientation, diagnostics


orientation, orientation_diagnostics = (
    detect_expression_orientation(ge_raw)
)

print("\nDetected expression orientation:")
print(orientation)

print("\nOrientation diagnostics:")
display(
    pd.DataFrame(
        [orientation_diagnostics]
    )
)

orientation_audit = pd.DataFrame([
    {
        "orientation": orientation,
        **orientation_diagnostics
    }
])

orientation_audit.to_csv(
    DIRS["input_audit"]
    / "expression_orientation_audit.tsv",
    sep="\t",
    index=False
)

ge_raw.iloc[:100, :20].to_csv(
    DIRS["input_audit"]
    / "expression_raw_preview.tsv",
    sep="\t",
    index=False
)

if orientation == "uncertain":
    raise ValueError(
        "Expression orientation could not be determined. "
        "Inspect the printed GE column names and first-column values."
    )

logger.info(
    "Expression orientation detected: %s",
    orientation
)