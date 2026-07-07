import json
from pathlib import Path

import pandas as pd
import pytest

from aido_bba.demo_pipeline import PATIENT_OUTPUT_COLUMNS, run_demo


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "demo" / "demo_config.json"
OUTPUT = ROOT / "demo" / "demo_outputs"


def test_demo_outputs_are_deterministic_and_complete():
    summary = run_demo(CONFIG)
    assert summary["status"] == "completed"
    assert summary["n_matched_patients"] == 60
    assert summary["n_input_genes"] == 24
    assert summary["class_counts"] == {"0": 30, "1": 30}
    assert summary["cv_folds"] == 3
    assert summary["cv_repeats"] == 2
    assert summary["oof_rows"] == 240
    assert summary["patient_rows"] == 60
    assert 0.5 <= summary["model_auc"]["logistic"] <= 1.0
    assert 0.5 <= summary["model_auc"]["extratrees"] <= 1.0
    assert 0.0 <= summary["representation_coverage"] <= 1.0
    assert summary["gap_gene_count"] == 6

    expected_files = {
        "demo_summary.json",
        "exclusion_log.csv",
        "fold_metrics.csv",
        "matched_samples.csv",
        "oof_probabilities.csv",
        "patient_reliability.csv",
        "representation_audit.csv",
        "representation_gap_genes.csv",
        "run_manifest.json",
    }
    assert expected_files.issubset({path.name for path in OUTPUT.iterdir()})

    patient = pd.read_csv(OUTPUT / "patient_reliability.csv")
    assert list(patient.columns) == PATIENT_OUTPUT_COLUMNS
    assert patient["mean_prob_logistic"].between(0, 1).all()
    assert patient["mean_prob_extratrees"].between(0, 1).all()

    first_summary = json.loads((OUTPUT / "demo_summary.json").read_text())
    second_summary = run_demo(CONFIG)
    assert second_summary == first_summary


def test_reference_metrics_with_reasonable_tolerance():
    summary = run_demo(CONFIG)
    assert summary["model_auc"]["logistic"] == pytest.approx(0.80, abs=0.15)
    assert summary["model_auc"]["extratrees"] == pytest.approx(0.75, abs=0.18)
