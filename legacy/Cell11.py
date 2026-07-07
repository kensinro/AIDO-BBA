# ============================================================
# CELL 11
# Construct the final model matrix and audit gene eligibility
# ============================================================

# ------------------------------------------------------------
# 1. Align expression samples to the final cohort order
# ------------------------------------------------------------

matched_sample_ids = (
    analysis_cohort["expression_sample_id"]
    .tolist()
)

matched_patient_ids = (
    analysis_cohort["patient_id"]
    .tolist()
)

X_raw = expression_primary.loc[
    matched_sample_ids
].copy()

# Use patient IDs as the modeling index
X_raw.index = matched_patient_ids
X_raw.index.name = "patient_id"

y = pd.Series(
    analysis_cohort["stage_label"].to_numpy(dtype=int),
    index=matched_patient_ids,
    name="stage_label"
)

stage_group_series = pd.Series(
    analysis_cohort["stage_group"].to_numpy(),
    index=matched_patient_ids,
    name="stage_group"
)

# ------------------------------------------------------------
# 2. Integrity checks
# ------------------------------------------------------------

assert X_raw.shape[0] == len(y)
assert X_raw.index.equals(y.index)
assert y.nunique() == 2
assert int((y == 0).sum()) == 803
assert int((y == 1).sum()) == 270

print("=" * 72)
print("MODEL MATRIX CONSTRUCTION")
print("=" * 72)

print("\nRaw matched matrix:")
print(X_raw.shape)

print("\nClass counts:")
display(
    pd.DataFrame({
        "stage_label": [0, 1],
        "stage_group": ["Early", "Advanced"],
        "n": [
            int((y == 0).sum()),
            int((y == 1).sum())
        ]
    })
)


# ------------------------------------------------------------
# 3. Numeric and finite-value audit
# ------------------------------------------------------------

X_numeric = X_raw.apply(
    pd.to_numeric,
    errors="coerce"
)

X_array = X_numeric.to_numpy(dtype=np.float64)

gene_names_raw = X_numeric.columns.to_numpy()

n_nonfinite_by_gene = (
    ~np.isfinite(X_array)
).sum(axis=0)

finite_gene_mask = (
    n_nonfinite_by_gene < X_numeric.shape[0]
)

# Remove only genes that contain no finite values at all
genes_all_nonfinite = gene_names_raw[
    ~finite_gene_mask
]

X_finite = X_numeric.loc[
    :,
    finite_gene_mask
].copy()


# ------------------------------------------------------------
# 4. Gene-level descriptive audit
# ------------------------------------------------------------

gene_mean = X_finite.mean(axis=0, skipna=True)
gene_sd = X_finite.std(axis=0, ddof=1, skipna=True)
gene_variance = X_finite.var(axis=0, ddof=1, skipna=True)
gene_minimum = X_finite.min(axis=0, skipna=True)
gene_maximum = X_finite.max(axis=0, skipna=True)

gene_missing_count = X_finite.isna().sum(axis=0)
gene_missing_fraction = (
    gene_missing_count / X_finite.shape[0]
)

gene_unique_values = X_finite.nunique(
    axis=0,
    dropna=True
)

gene_audit = pd.DataFrame({
    "gene_id_original": X_finite.columns,
    "n_missing": gene_missing_count.values,
    "missing_fraction": gene_missing_fraction.values,
    "n_unique_values": gene_unique_values.values,
    "mean": gene_mean.values,
    "sd": gene_sd.values,
    "variance": gene_variance.values,
    "minimum": gene_minimum.values,
    "maximum": gene_maximum.values
})


# ------------------------------------------------------------
# 5. Label-independent eligibility rules
# ------------------------------------------------------------
#
# These filters do not use the endpoint labels.
#
# Rule A:
#   Remove zero-variance genes.
#
# Rule B:
#   Remove genes with fewer than 2 unique finite values.
#
# Extremely small but nonzero variance is retained here.
# Supervised selection will occur inside CV.
# ------------------------------------------------------------

gene_audit["is_zero_variance"] = (
    gene_audit["variance"].fillna(0) <= 0
)

gene_audit["has_fewer_than_2_unique_values"] = (
    gene_audit["n_unique_values"] < 2
)

gene_audit["eligible_for_blackbox"] = (
    ~gene_audit["is_zero_variance"]
    & ~gene_audit["has_fewer_than_2_unique_values"]
)

gene_audit["exclusion_reason"] = np.select(
    [
        gene_audit["is_zero_variance"],
        gene_audit["has_fewer_than_2_unique_values"]
    ],
    [
        "zero_variance",
        "fewer_than_2_unique_values"
    ],
    default="retained"
)

eligible_genes = gene_audit.loc[
    gene_audit["eligible_for_blackbox"],
    "gene_id_original"
].tolist()

X = X_finite.loc[
    :,
    eligible_genes
].copy()


# ------------------------------------------------------------
# 6. Final matrix audit
# ------------------------------------------------------------

remaining_nonfinite = int(
    (~np.isfinite(
        X.to_numpy(dtype=np.float64)
    )).sum()
)

gene_universe_summary = pd.DataFrame([
    {
        "metric": "matched_patients",
        "value": int(X_raw.shape[0])
    },
    {
        "metric": "raw_gene_identifiers",
        "value": int(X_raw.shape[1])
    },
    {
        "metric": "genes_all_nonfinite_removed",
        "value": int(len(genes_all_nonfinite))
    },
    {
        "metric": "zero_variance_genes_removed",
        "value": int(
            gene_audit["is_zero_variance"].sum()
        )
    },
    {
        "metric": "genes_with_fewer_than_2_unique_values_removed",
        "value": int(
            gene_audit[
                "has_fewer_than_2_unique_values"
            ].sum()
        )
    },
    {
        "metric": "final_blackbox_gene_universe",
        "value": int(X.shape[1])
    },
    {
        "metric": "remaining_nonfinite_values",
        "value": remaining_nonfinite
    },
    {
        "metric": "early_patients",
        "value": int((y == 0).sum())
    },
    {
        "metric": "advanced_patients",
        "value": int((y == 1).sum())
    }
])

print("\nGene-universe summary:")
display(gene_universe_summary)

print("\nFinal black-box model matrix:")
print(X.shape)

print("\nGene audit preview:")
display(
    gene_audit.head(20)
)

print("\nExcluded genes:")
display(
    gene_audit.loc[
        ~gene_audit["eligible_for_blackbox"],
        [
            "gene_id_original",
            "n_unique_values",
            "variance",
            "exclusion_reason"
        ]
    ].head(30)
)


# ------------------------------------------------------------
# 7. Save audit outputs
# ------------------------------------------------------------

gene_audit.to_csv(
    DIRS["harmonization"]
    / "blackbox_gene_universe_audit.tsv",
    sep="\t",
    index=False
)

gene_universe_summary.to_csv(
    DIRS["blackbox"]
    / "model_input_summary.tsv",
    sep="\t",
    index=False
)

pd.DataFrame({
    "patient_id": X.index,
    "stage_label": y.loc[X.index].values,
    "stage_group": stage_group_series.loc[X.index].values
}).to_csv(
    DIRS["blackbox"]
    / "model_patient_manifest.tsv",
    sep="\t",
    index=False
)

pd.DataFrame({
    "gene_id": X.columns
}).to_csv(
    DIRS["blackbox"]
    / "model_gene_manifest.tsv",
    sep="\t",
    index=False
)

logger.info(
    "Model matrix completed: %s patients x %s genes; "
    "%s Early; %s Advanced.",
    X.shape[0],
    X.shape[1],
    int((y == 0).sum()),
    int((y == 1).sum())
)