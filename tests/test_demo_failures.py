import json
from pathlib import Path

import pandas as pd
import pytest

from aido_bba.demo_pipeline import run_demo
from aido_bba.demo_schema import SchemaError


ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "demo"


def test_failure_state_is_written_for_invalid_input(tmp_path):
    bad_clinical = pd.read_csv(DEMO / "demo_clinical.csv").drop(columns=["stage_group"])
    bad_clinical_path = tmp_path / "bad_clinical.csv"
    bad_clinical.to_csv(bad_clinical_path, index=False)
    output_dir = tmp_path / "outputs"
    config = {
        "seed": 20260707,
        "expression_path": str(DEMO / "demo_expression.csv"),
        "clinical_path": str(bad_clinical_path),
        "gene_set_path": str(DEMO / "demo_gene_sets.csv"),
        "output_dir": str(output_dir),
        "gene_id_column": "gene_id",
        "patient_id_column": "patient_id",
        "endpoint_column": "stage_group",
        "class_mapping": {"early": 0, "advanced": 1},
        "cv_folds": 3,
        "cv_repeats": 2,
    }
    config_path = tmp_path / "demo_config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(SchemaError):
        run_demo(config_path)

    failure = pd.read_csv(output_dir / "failure_log.csv")
    assert failure.loc[0, "failure_code"] == "SCHEMA_VALIDATION_FAILED"
    manifest = json.loads((output_dir / "run_manifest.json").read_text())
    assert manifest["status"] == "failed"
