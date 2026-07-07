from aido_bba.config import (
    data_root, brca_output_root, brca_run_dir, go_bp_gmt,
    hgnc_file, ncbi_gene_info, metabric_dir, gse96058_dir,
    kirc_dir, kirc_output_root,
)

# ============================================================
# Input file audit
# ============================================================

INPUT_FILES = {
    "TCGA_BRCA_GE": GE_FILE,
    "TCGA_BRCA_STAGE": STAGE_FILE,
    "TCGA_BRCA_PHENOTYPE": PHENOTYPE_FILE,
    "TCGA_BRCA_CLINICAL_MATRIX": CLINICAL_MATRIX_FILE,
}

input_audit = audit_input_files(INPUT_FILES)

display(input_audit)

required_inputs = [
    GE_FILE,
    STAGE_FILE
]

missing_required = [
    str(path)
    for path in required_inputs
    if not Path(path).exists()
]

if missing_required:
    raise FileNotFoundError(
        "Required input files are missing:\n"
        + "\n".join(missing_required)
    )

logger.info("Required input files were found.")