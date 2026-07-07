from pathlib import Path

import pandas as pd
import pytest

from aido_bba.demo_schema import SchemaError, validate_and_align_inputs


ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "demo"


def _validate(expression: Path, clinical: Path, gene_sets: Path):
    return validate_and_align_inputs(
        expression,
        clinical,
        gene_sets,
        "gene_id",
        "patient_id",
        "stage_group",
        {"early": 0, "advanced": 1},
    )


def test_valid_demo_inputs_pass():
    inputs = _validate(
        DEMO / "demo_expression.csv",
        DEMO / "demo_clinical.csv",
        DEMO / "demo_gene_sets.csv",
    )
    assert len(inputs.matched_patients) == 60
    assert inputs.extra_expression_patients == []


def test_missing_endpoint_column_fails(tmp_path):
    clinical = pd.read_csv(DEMO / "demo_clinical.csv").drop(columns=["stage_group"])
    path = tmp_path / "clinical.csv"
    clinical.to_csv(path, index=False)
    with pytest.raises(SchemaError, match="missing required column"):
        _validate(DEMO / "demo_expression.csv", path, DEMO / "demo_gene_sets.csv")


def test_duplicate_patient_id_fails(tmp_path):
    clinical = pd.read_csv(DEMO / "demo_clinical.csv")
    clinical.loc[1, "patient_id"] = clinical.loc[0, "patient_id"]
    path = tmp_path / "clinical.csv"
    clinical.to_csv(path, index=False)
    with pytest.raises(SchemaError, match="duplicate patient"):
        _validate(DEMO / "demo_expression.csv", path, DEMO / "demo_gene_sets.csv")


def test_unmatched_clinical_patient_fails(tmp_path):
    clinical = pd.read_csv(DEMO / "demo_clinical.csv")
    clinical.loc[0, "patient_id"] = "MISSING_PATIENT"
    path = tmp_path / "clinical.csv"
    clinical.to_csv(path, index=False)
    with pytest.raises(SchemaError, match="missing from expression"):
        _validate(DEMO / "demo_expression.csv", path, DEMO / "demo_gene_sets.csv")


def test_invalid_endpoint_value_fails(tmp_path):
    clinical = pd.read_csv(DEMO / "demo_clinical.csv")
    clinical.loc[0, "stage_group"] = "unknown"
    path = tmp_path / "clinical.csv"
    clinical.to_csv(path, index=False)
    with pytest.raises(SchemaError, match="invalid endpoint"):
        _validate(DEMO / "demo_expression.csv", path, DEMO / "demo_gene_sets.csv")


def test_non_numeric_expression_fails(tmp_path):
    expression = pd.read_csv(DEMO / "demo_expression.csv")
    expression["P001"] = expression["P001"].astype(object)
    expression.loc[0, "P001"] = "not_numeric"
    path = tmp_path / "expression.csv"
    expression.to_csv(path, index=False)
    with pytest.raises(SchemaError, match="must be numeric"):
        _validate(path, DEMO / "demo_clinical.csv", DEMO / "demo_gene_sets.csv")


def test_reversed_expression_orientation_fails(tmp_path):
    expression = pd.read_csv(DEMO / "demo_expression.csv").set_index("gene_id").T.reset_index()
    expression = expression.rename(columns={"index": "gene_id"})
    path = tmp_path / "expression.csv"
    expression.to_csv(path, index=False)
    with pytest.raises(SchemaError, match="orientation appears invalid"):
        _validate(path, DEMO / "demo_clinical.csv", DEMO / "demo_gene_sets.csv")
