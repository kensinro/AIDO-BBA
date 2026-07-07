"""Schema validation utilities for the compact AIDO-BBA demonstration."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


class SchemaError(ValueError):
    """Raised when a demo input or output violates the documented contract."""


@dataclass(frozen=True)
class DemoInputs:
    expression: pd.DataFrame
    clinical: pd.DataFrame
    gene_sets: pd.DataFrame
    matched_patients: list[str]
    extra_expression_patients: list[str]


def _require_columns(frame: pd.DataFrame, required: Iterable[str], table_name: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise SchemaError(
            f"{table_name}: missing required column(s): {', '.join(missing)}"
        )


def validate_expression_table(frame: pd.DataFrame, gene_id_column: str) -> None:
    _require_columns(frame, [gene_id_column], "expression")
    if frame.empty:
        raise SchemaError("expression: table is empty")
    if frame[gene_id_column].isna().any() or (frame[gene_id_column].astype(str).str.strip() == "").any():
        raise SchemaError("expression: gene identifiers must be non-empty")
    duplicated = frame[gene_id_column].astype(str).duplicated(keep=False)
    if duplicated.any():
        genes = sorted(frame.loc[duplicated, gene_id_column].astype(str).unique())
        raise SchemaError(f"expression: duplicate gene identifier(s): {', '.join(genes[:5])}")

    patient_columns = [column for column in frame.columns if column != gene_id_column]
    if len(patient_columns) < 6:
        raise SchemaError("expression: at least six patient columns are required")

    numeric = frame[patient_columns].apply(pd.to_numeric, errors="coerce")
    originally_missing = frame[patient_columns].isna()
    conversion_failed = numeric.isna() & ~originally_missing
    if conversion_failed.any().any():
        raise SchemaError("expression: all expression values must be numeric")
    if not np.isfinite(numeric.to_numpy(dtype=float, na_value=np.nan)).all():
        raise SchemaError("expression: missing or infinite values are not permitted in the demo")


def validate_clinical_table(
    frame: pd.DataFrame,
    patient_id_column: str,
    endpoint_column: str,
    allowed_labels: set[str],
) -> None:
    _require_columns(frame, [patient_id_column, endpoint_column], "clinical")
    if frame.empty:
        raise SchemaError("clinical: table is empty")
    patient_ids = frame[patient_id_column].astype(str)
    if patient_ids.str.strip().eq("").any() or frame[patient_id_column].isna().any():
        raise SchemaError("clinical: patient identifiers must be non-empty")
    duplicated = patient_ids.duplicated(keep=False)
    if duplicated.any():
        patients = sorted(patient_ids[duplicated].unique())
        raise SchemaError(f"clinical: duplicate patient identifier(s): {', '.join(patients[:5])}")

    observed = set(frame[endpoint_column].astype(str))
    invalid = sorted(observed - allowed_labels)
    if invalid:
        raise SchemaError(f"clinical: invalid endpoint value(s): {', '.join(invalid)}")
    if len(observed) < 2:
        raise SchemaError("clinical: endpoint must contain at least two classes")


def validate_gene_sets(frame: pd.DataFrame) -> None:
    _require_columns(frame, ["process_id", "gene_id"], "gene_sets")
    if frame.empty:
        raise SchemaError("gene_sets: table is empty")
    if frame[["process_id", "gene_id"]].isna().any().any():
        raise SchemaError("gene_sets: process_id and gene_id must be non-empty")


def validate_and_align_inputs(
    expression_path: Path,
    clinical_path: Path,
    gene_set_path: Path,
    gene_id_column: str,
    patient_id_column: str,
    endpoint_column: str,
    class_mapping: dict[str, int],
) -> DemoInputs:
    expression = pd.read_csv(expression_path)
    clinical = pd.read_csv(clinical_path)
    gene_sets = pd.read_csv(gene_set_path)

    validate_expression_table(expression, gene_id_column)
    validate_clinical_table(
        clinical,
        patient_id_column,
        endpoint_column,
        set(class_mapping),
    )
    validate_gene_sets(gene_sets)

    expression_patients = [column for column in expression.columns if column != gene_id_column]
    expression_patient_set = set(expression_patients)
    clinical_patients = clinical[patient_id_column].astype(str).tolist()

    # Detect the common accidental transpose: clinical IDs appear down the first column.
    first_column_values = set(expression[gene_id_column].astype(str))
    if set(clinical_patients).issubset(first_column_values):
        raise SchemaError(
            "expression: orientation appears invalid; expected rows=genes and columns=patients"
        )

    missing_from_expression = sorted(set(clinical_patients) - expression_patient_set)
    if missing_from_expression:
        raise SchemaError(
            "cross_table: clinical patient(s) missing from expression: "
            + ", ".join(missing_from_expression[:5])
        )

    matched_patients = [patient for patient in clinical_patients if patient in expression_patient_set]
    extra_expression_patients = sorted(expression_patient_set - set(clinical_patients))

    return DemoInputs(
        expression=expression,
        clinical=clinical,
        gene_sets=gene_sets,
        matched_patients=matched_patients,
        extra_expression_patients=extra_expression_patients,
    )


def validate_output_columns(frame: pd.DataFrame, expected_columns: list[str], table_name: str) -> None:
    actual = list(frame.columns)
    if actual != expected_columns:
        raise SchemaError(
            f"{table_name}: output schema mismatch; expected {expected_columns}, observed {actual}"
        )
