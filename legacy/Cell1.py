# ============================================================
# AIDO-BBA BRCA 1.0
# M0-M3: Input audit, cohort construction and black-box baseline
# ============================================================

from pathlib import Path
from datetime import datetime
import json
import logging
import platform
import sys
import warnings

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    balanced_accuracy_score,
    accuracy_score,
    average_precision_score,
    log_loss,
    brier_score_loss,
    confusion_matrix
)

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# 1. Run settings
# ------------------------------------------------------------

RANDOM_SEED = 20260701

# Quick test:
#   N_REPEATS = 3 or 5
#
# Full planned run:
#   N_REPEATS = 20
N_SPLITS = 5
N_REPEATS = 5

# Number of genes selected inside each training fold.
# Feature selection is performed within CV to prevent leakage.
N_SELECTED_GENES = 1500

# ExtraTrees black-box parameters
N_TREES = 400
MAX_DEPTH = None
MIN_SAMPLES_LEAF = 3
N_JOBS = -1

# ------------------------------------------------------------
# 2. Input paths
# ------------------------------------------------------------

DATA_ROOT = Path(r"D:\AIDO-Data")

TCGA_BRCA_DIR = (
    DATA_ROOT
    / "UCSC_XENA"
    / "Breast Cancer (BRCA)"
)

GE_FILE = TCGA_BRCA_DIR / "GE.tsv"

STAGE_FILE = (
    TCGA_BRCA_DIR
    / "BRCA_stage_groups_from_survival.tsv"
)

PHENOTYPE_FILE = TCGA_BRCA_DIR / "Phenotype.tsv"

CLINICAL_MATRIX_FILE = (
    TCGA_BRCA_DIR
    / "TCGA.BRCA.sampleMap_BRCA_clinicalMatrix"
)

# ------------------------------------------------------------
# 3. Output paths
# ------------------------------------------------------------

OUTPUT_ROOT = Path(
    r"D:\AIDO-Temp\AIDO_BBA_BRCA_1_0"
)

RUN_NAME = (
    f"RUN_BASELINE_GE_STAGE_"
    f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
)

RUN_DIR = OUTPUT_ROOT / RUN_NAME

DIRS = {
    "manifest": RUN_DIR / "00_manifest",
    "input_audit": RUN_DIR / "01_input_audit",
    "harmonization": RUN_DIR / "02_harmonization",
    "endpoint": RUN_DIR / "03_endpoint",
    "blackbox": RUN_DIR / "04_blackbox",
    "logs": RUN_DIR / "logs",
}

for directory in DIRS.values():
    directory.mkdir(parents=True, exist_ok=True)

print("AIDO-BBA BRCA run directory:")
print(RUN_DIR)