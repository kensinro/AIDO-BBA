from aido_bba.config import (
    data_root, brca_output_root, brca_run_dir, go_bp_gmt,
    hgnc_file, ncbi_gene_info, metabric_dir, gse96058_dir,
    kirc_dir, kirc_output_root,
)

# ============================================================
# Utility functions
# ============================================================

LOG_FILE = DIRS["logs"] / "AIDO_BBA_BRCA.log"

logger = logging.getLogger("AIDO_BBA_BRCA")
logger.setLevel(logging.INFO)
logger.handlers.clear()

file_handler = logging.FileHandler(
    LOG_FILE,
    mode="w",
    encoding="utf-8"
)
console_handler = logging.StreamHandler()

formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(message)s"
)

file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)


def audit_input_files(path_dict):
    """
    Audit whether required input files exist.
    """
    records = []

    for name, path in path_dict.items():
        path = Path(path)

        records.append({
            "input_name": name,
            "path": str(path),
            "exists": path.exists(),
            "size_bytes": (
                path.stat().st_size
                if path.exists()
                else np.nan
            ),
            "size_mb": (
                round(path.stat().st_size / 1024**2, 3)
                if path.exists()
                else np.nan
            )
        })

    audit_df = pd.DataFrame(records)

    audit_df.to_csv(
        DIRS["input_audit"] / "input_file_audit.tsv",
        sep="\t",
        index=False
    )

    return audit_df


def read_table_auto(path, nrows=None):
    """
    Read TSV/CSV files using delimiter inference.
    Falls back to tab-separated loading.
    """
    path = Path(path)

    try:
        df = pd.read_csv(
            path,
            sep=None,
            engine="python",
            nrows=nrows,
            low_memory=False
        )

    except Exception:
        df = pd.read_csv(
            path,
            sep="\t",
            nrows=nrows,
            low_memory=False
        )

    return df


def clean_column_names(df):
    """
    Remove surrounding spaces and normalize unnamed columns.
    """
    df = df.copy()

    df.columns = [
        str(column).strip()
        for column in df.columns
    ]

    return df


def normalize_tcga_barcode(value):
    """
    Normalize common TCGA barcode formats.

    Example:
    TCGA-XX-YYYY-01A-01R
    remains TCGA-XX-YYYY-01A-01R.
    """
    if pd.isna(value):
        return np.nan

    value = str(value).strip()
    value = value.replace(".", "-")
    value = value.upper()

    if not value.startswith("TCGA-"):
        return value

    return value


def tcga_patient_id(value):
    """
    Return the 12-character TCGA participant identifier.

    Example:
    TCGA-XX-YYYY-01A -> TCGA-XX-YYYY
    """
    value = normalize_tcga_barcode(value)

    if pd.isna(value):
        return np.nan

    if str(value).startswith("TCGA-") and len(str(value)) >= 12:
        return str(value)[:12]

    return str(value)


def tcga_sample_type_code(value):
    """
    Extract TCGA sample-type code.

    For TCGA-XX-YYYY-01A, the code is 01.
    """
    value = normalize_tcga_barcode(value)

    if pd.isna(value):
        return np.nan

    parts = str(value).split("-")

    if len(parts) >= 4 and len(parts[3]) >= 2:
        return parts[3][:2]

    return np.nan


def is_primary_tumour_barcode(value):
    """
    TCGA sample type 01 = Primary Solid Tumor.
    """
    return tcga_sample_type_code(value) == "01"


def safe_auc(y_true, y_prob):
    if len(np.unique(y_true)) < 2:
        return np.nan

    return roc_auc_score(y_true, y_prob)


logger.info("Utility functions initialized.")