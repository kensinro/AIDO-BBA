from aido_bba.config import (
    data_root, brca_output_root, brca_run_dir, go_bp_gmt,
    hgnc_file, ncbi_gene_info, metabric_dir, gse96058_dir,
    kirc_dir, kirc_output_root,
)

# ============================================================
# CELL 5
# Construct and audit Early-versus-Advanced stage endpoint
#
# 0 = Early     : Stage I / II
# 1 = Advanced  : Stage III / IV
# ============================================================

SAMPLE_COLUMN = "sampleID"
PATIENT_COLUMN = "_PATIENT"
RAW_STAGE_COLUMN = "stage_raw"
GROUP_COLUMN = "stage_group"


def normalize_text(value):
    """
    Normalize text while preserving missing values.
    """
    if pd.isna(value):
        return np.nan

    text = str(value).strip()

    if text == "":
        return np.nan

    return text


def normalize_stage_group(value):
    """
    Convert the existing stage_group field into:
        Early
        Advanced
        NaN
    """
    text = normalize_text(value)

    if pd.isna(text):
        return np.nan

    text_lower = text.lower()

    if text_lower in {
        "early",
        "early stage",
        "stage i/ii",
        "i/ii"
    }:
        return "Early"

    if text_lower in {
        "advanced",
        "advanced stage",
        "late",
        "late stage",
        "stage iii/iv",
        "iii/iv"
    }:
        return "Advanced"

    return np.nan


def raw_stage_to_group(value):
    """
    Independently reconstruct the stage group from stage_raw.

    Early:
        Stage I, IA, IB, IC
        Stage II, IIA, IIB, IIC

    Advanced:
        Stage III, IIIA, IIIB, IIIC
        Stage IV, IVA, IVB, IVC

    Stage X, unknown, missing, and unrecognized values remain NaN.
    """
    text = normalize_text(value)

    if pd.isna(text):
        return np.nan

    text = (
        text.upper()
        .replace("PATHOLOGIC", "")
        .replace("PATHOLOGICAL", "")
        .replace("CLINICAL", "")
        .replace("AJCC", "")
        .replace("STAGE", "")
        .strip()
    )

    text = (
        text.replace("-", "")
            .replace("_", "")
            .replace(".", "")
            .replace(" ", "")
    )

    unresolved_values = {
        "",
        "X",
        "NA",
        "NAN",
        "NONE",
        "UNKNOWN",
        "NOTREPORTED",
        "NOTAVAILABLE"
    }

    if text in unresolved_values:
        return np.nan

    # Check advanced first because III and IV also begin with I
    if text.startswith("III") or text.startswith("IV"):
        return "Advanced"

    if text.startswith("II") or text.startswith("I"):
        return "Early"

    return np.nan


stage_endpoint = stage_raw[
    [
        SAMPLE_COLUMN,
        PATIENT_COLUMN,
        RAW_STAGE_COLUMN,
        GROUP_COLUMN
    ]
].copy()

stage_endpoint.columns = [
    "sample_id",
    "patient_id",
    "stage_raw",
    "stage_group_original"
]

# Normalize identifiers
stage_endpoint["sample_id"] = (
    stage_endpoint["sample_id"]
    .map(normalize_tcga_barcode)
)

stage_endpoint["patient_id"] = (
    stage_endpoint["patient_id"]
    .map(normalize_tcga_barcode)
    .map(tcga_patient_id)
)

# Existing group supplied in the file
stage_endpoint["stage_group_existing"] = (
    stage_endpoint["stage_group_original"]
    .map(normalize_stage_group)
)

# Independent reconstruction from stage_raw
stage_endpoint["stage_group_reconstructed"] = (
    stage_endpoint["stage_raw"]
    .map(raw_stage_to_group)
)

# Compare the two definitions where both are available
stage_endpoint["group_agreement"] = np.where(
    stage_endpoint["stage_group_existing"].notna()
    & stage_endpoint["stage_group_reconstructed"].notna(),
    stage_endpoint["stage_group_existing"]
    == stage_endpoint["stage_group_reconstructed"],
    np.nan
)

# Primary endpoint:
# Prefer the existing curated stage_group.
# Use reconstructed stage only when the curated value is missing.
stage_endpoint["stage_group"] = (
    stage_endpoint["stage_group_existing"]
    .fillna(stage_endpoint["stage_group_reconstructed"])
)

stage_endpoint["stage_label"] = (
    stage_endpoint["stage_group"]
    .map({
        "Early": 0,
        "Advanced": 1
    })
)

stage_endpoint["endpoint_source"] = np.select(
    [
        stage_endpoint["stage_group_existing"].notna(),
        stage_endpoint["stage_group_existing"].isna()
        & stage_endpoint["stage_group_reconstructed"].notna()
    ],
    [
        "existing_stage_group",
        "reconstructed_from_stage_raw"
    ],
    default="unresolved"
)

print("=" * 72)
print("STAGE ENDPOINT AUDIT")
print("=" * 72)

print("\nOriginal stage_group counts:")
display(
    stage_endpoint["stage_group_original"]
    .value_counts(dropna=False)
    .rename_axis("stage_group_original")
    .reset_index(name="n")
)

print("\nReconstructed group counts from stage_raw:")
display(
    stage_endpoint["stage_group_reconstructed"]
    .value_counts(dropna=False)
    .rename_axis("stage_group_reconstructed")
    .reset_index(name="n")
)

print("\nFinal endpoint counts before expression matching:")
display(
    stage_endpoint["stage_group"]
    .value_counts(dropna=False)
    .rename_axis("stage_group")
    .reset_index(name="n")
)

print("\nEndpoint source:")
display(
    stage_endpoint["endpoint_source"]
    .value_counts(dropna=False)
    .rename_axis("endpoint_source")
    .reset_index(name="n")
)

print("\nExisting-versus-reconstructed agreement:")
display(
    stage_endpoint["group_agreement"]
    .value_counts(dropna=False)
    .rename_axis("agreement")
    .reset_index(name="n")
)