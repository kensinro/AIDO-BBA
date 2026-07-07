from aido_bba.config import (
    data_root, brca_output_root, brca_run_dir, go_bp_gmt,
    hgnc_file, ncbi_gene_info, metabric_dir, gse96058_dir,
    kirc_dir, kirc_output_root,
)

# ============================================================
# CELL 14
# Save raw repeated-CV outputs immediately
# ============================================================

fold_performance.to_csv(
    DIRS["blackbox"] / "fold_performance_all_models.tsv",
    sep="\t",
    index=False
)

oof_predictions_all.to_csv(
    DIRS["blackbox"]
    / "oof_predictions_all_models_all_repeats.tsv",
    sep="\t",
    index=False
)

selected_genes_by_fold.to_csv(
    DIRS["blackbox"]
    / "selected_genes_by_model_and_fold.tsv",
    sep="\t",
    index=False
)

model_features_by_fold.to_csv(
    DIRS["blackbox"]
    / "model_feature_values_by_fold.tsv",
    sep="\t",
    index=False
)

fit_timing.to_csv(
    DIRS["blackbox"]
    / "model_fit_timing.tsv",
    sep="\t",
    index=False
)

print("=" * 72)
print("RAW MODEL OUTPUTS SAVED")
print("=" * 72)

print("Fold performance rows:", len(fold_performance))
print("OOF prediction rows:", len(oof_predictions_all))
print("Selected-gene rows:", len(selected_genes_by_fold))
print("Model-feature rows:", len(model_features_by_fold))

print("\nOutput directory:")
print(DIRS["blackbox"])

logger.info(
    "Raw model outputs saved successfully."
)