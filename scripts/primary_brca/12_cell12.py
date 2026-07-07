from aido_bba.config import (
    data_root, brca_output_root, brca_run_dir, go_bp_gmt,
    hgnc_file, ncbi_gene_info, metabric_dir, gse96058_dir,
    kirc_dir, kirc_output_root,
)

# ============================================================
# CELL 12
# Fixed repeated-CV folds and model definitions
# ============================================================

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.base import clone

# ------------------------------------------------------------
# 1. Analysis settings
# ------------------------------------------------------------

# Quick baseline:
N_SPLITS = 5
N_REPEATS = 5

# Formal run later:
# N_REPEATS = 20

N_SELECTED_GENES = 1500

# Elastic-net settings
ELASTICNET_C = 0.1
ELASTICNET_L1_RATIO = 0.5
ELASTICNET_MAX_ITER = 5000

# ExtraTrees settings
N_TREES = 400
MIN_SAMPLES_LEAF = 3
MAX_DEPTH = None
MAX_FEATURES = "sqrt"
N_JOBS = -1

# Use the random seed already defined in CELL 1
print("Random seed:", RANDOM_SEED)


# ------------------------------------------------------------
# 2. Convert model data into arrays
# ------------------------------------------------------------

X_values = X.to_numpy(dtype=np.float32)
y_values = y.loc[X.index].to_numpy(dtype=int)

patient_ids = X.index.to_numpy()
gene_names = X.columns.to_numpy()

effective_k = min(
    N_SELECTED_GENES,
    X.shape[1]
)

print("\nModel input:")
print("Patients:", X_values.shape[0])
print("Genes:", X_values.shape[1])
print("Selected genes per training fold:", effective_k)


# ------------------------------------------------------------
# 3. Generate fixed repeated stratified folds
# ------------------------------------------------------------

cv_generator = RepeatedStratifiedKFold(
    n_splits=N_SPLITS,
    n_repeats=N_REPEATS,
    random_state=RANDOM_SEED
)

cv_splits = list(
    cv_generator.split(
        X_values,
        y_values
    )
)

print("\nCV configuration:")
print("Splits per repeat:", N_SPLITS)
print("Repeats:", N_REPEATS)
print("Total model fits per model:", len(cv_splits))
print("Total model fits for two models:", 2 * len(cv_splits))


# ------------------------------------------------------------
# 4. Create fold-assignment manifest
# ------------------------------------------------------------

fold_assignment_records = []

for split_number, (train_index, test_index) in enumerate(
    cv_splits,
    start=1
):

    repeat_id = (
        (split_number - 1) // N_SPLITS
    ) + 1

    fold_id = (
        (split_number - 1) % N_SPLITS
    ) + 1

    for index in train_index:
        fold_assignment_records.append({
            "patient_id": patient_ids[index],
            "repeat_id": repeat_id,
            "fold_id": fold_id,
            "partition": "train",
            "stage_label": int(y_values[index]),
            "stage_group": (
                "Advanced"
                if y_values[index] == 1
                else "Early"
            )
        })

    for index in test_index:
        fold_assignment_records.append({
            "patient_id": patient_ids[index],
            "repeat_id": repeat_id,
            "fold_id": fold_id,
            "partition": "test",
            "stage_label": int(y_values[index]),
            "stage_group": (
                "Advanced"
                if y_values[index] == 1
                else "Early"
            )
        })

cv_fold_assignments = pd.DataFrame(
    fold_assignment_records
)

cv_fold_assignments.to_csv(
    DIRS["blackbox"] / "cv_fold_assignments.tsv",
    sep="\t",
    index=False
)


# ------------------------------------------------------------
# 5. Fold-balance audit
# ------------------------------------------------------------

cv_fold_balance = (
    cv_fold_assignments
    .groupby(
        [
            "repeat_id",
            "fold_id",
            "partition",
            "stage_group"
        ],
        as_index=False
    )
    .size()
    .rename(columns={"size": "n"})
)

cv_fold_balance.to_csv(
    DIRS["blackbox"] / "cv_fold_balance.tsv",
    sep="\t",
    index=False
)

print("\nFold-balance preview:")
display(
    cv_fold_balance.head(20)
)


# ------------------------------------------------------------
# 6. Transparent comparator
# ------------------------------------------------------------
#
# Imputation and supervised feature selection occur inside each
# training fold.
#
# StandardScaler is needed for elastic-net logistic regression.
# ------------------------------------------------------------

elasticnet_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),
        (
            "feature_selection",
            SelectKBest(
                score_func=f_classif,
                k=effective_k
            )
        ),
        (
            "scaler",
            StandardScaler()
        ),
        (
            "classifier",
            LogisticRegression(
                penalty="elasticnet",
                solver="saga",
                C=ELASTICNET_C,
                l1_ratio=ELASTICNET_L1_RATIO,
                class_weight="balanced",
                max_iter=ELASTICNET_MAX_ITER,
                random_state=RANDOM_SEED,
                n_jobs=N_JOBS
            )
        )
    ]
)


# ------------------------------------------------------------
# 7. Primary black-box model
# ------------------------------------------------------------

extratrees_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),
        (
            "feature_selection",
            SelectKBest(
                score_func=f_classif,
                k=effective_k
            )
        ),
        (
            "classifier",
            ExtraTreesClassifier(
                n_estimators=N_TREES,
                max_depth=MAX_DEPTH,
                min_samples_leaf=MIN_SAMPLES_LEAF,
                max_features=MAX_FEATURES,
                class_weight="balanced",
                random_state=RANDOM_SEED,
                n_jobs=N_JOBS
            )
        )
    ]
)


# ------------------------------------------------------------
# 8. Model registry
# ------------------------------------------------------------

MODEL_REGISTRY = {
    "ElasticNet_Logistic": elasticnet_pipeline,
    "ExtraTrees_BlackBox": extratrees_pipeline
}

model_configuration = pd.DataFrame([
    {
        "model_name": "ElasticNet_Logistic",
        "model_role": "transparent_comparator",
        "feature_selection": "SelectKBest_f_classif_inside_fold",
        "n_selected_genes": effective_k,
        "class_weight": "balanced",
        "main_parameters": (
            f"C={ELASTICNET_C}; "
            f"l1_ratio={ELASTICNET_L1_RATIO}; "
            f"max_iter={ELASTICNET_MAX_ITER}"
        )
    },
    {
        "model_name": "ExtraTrees_BlackBox",
        "model_role": "primary_blackbox",
        "feature_selection": "SelectKBest_f_classif_inside_fold",
        "n_selected_genes": effective_k,
        "class_weight": "balanced",
        "main_parameters": (
            f"n_estimators={N_TREES}; "
            f"min_samples_leaf={MIN_SAMPLES_LEAF}; "
            f"max_features={MAX_FEATURES}; "
            f"max_depth={MAX_DEPTH}"
        )
    }
])

model_configuration.to_csv(
    DIRS["blackbox"] / "model_configuration.tsv",
    sep="\t",
    index=False
)

print("\nModel configuration:")
display(model_configuration)


# ------------------------------------------------------------
# 9. Integrity checks
# ------------------------------------------------------------

expected_test_appearances = N_REPEATS

test_appearance_counts = (
    cv_fold_assignments[
        cv_fold_assignments["partition"] == "test"
    ]
    .groupby("patient_id")
    .size()
)

assert (
    test_appearance_counts == expected_test_appearances
).all(), (
    "At least one patient does not appear exactly once "
    "per repeat in the held-out test partitions."
)

assert len(cv_splits) == N_SPLITS * N_REPEATS

logger.info(
    "CV and model configuration completed: "
    "%s folds per model; %s selected genes per fold.",
    len(cv_splits),
    effective_k
)

print("\n" + "=" * 72)
print("CV AND MODEL CONFIGURATION COMPLETED")
print("=" * 72)
print("Every patient has held-out predictions:", N_REPEATS)
print("Models:", list(MODEL_REGISTRY.keys()))