"""Compact deterministic demonstration of the AIDO-BBA software contract."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .demo_schema import SchemaError, validate_and_align_inputs, validate_output_columns


@dataclass(frozen=True)
class DemoConfig:
    seed: int
    expression_path: Path
    clinical_path: Path
    gene_set_path: Path
    output_dir: Path
    gene_id_column: str
    patient_id_column: str
    endpoint_column: str
    class_mapping: dict[str, int]
    cv_folds: int
    cv_repeats: int


PATIENT_OUTPUT_COLUMNS = [
    "patient_id",
    "true_label",
    "mean_prob_logistic",
    "mean_prob_extratrees",
    "cross_model_difference",
    "resampling_sd_logistic",
    "resampling_sd_extratrees",
    "model_disagreement",
]

OOF_COLUMNS = [
    "patient_id",
    "repeat",
    "fold",
    "model",
    "true_label",
    "oof_probability",
    "predicted_label",
]


def load_demo_config(config_path: Path) -> DemoConfig:
    with config_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    base = config_path.parent.parent.resolve()

    def resolve(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else (base / path).resolve()

    return DemoConfig(
        seed=int(raw["seed"]),
        expression_path=resolve(raw["expression_path"]),
        clinical_path=resolve(raw["clinical_path"]),
        gene_set_path=resolve(raw["gene_set_path"]),
        output_dir=resolve(raw["output_dir"]),
        gene_id_column=str(raw["gene_id_column"]),
        patient_id_column=str(raw["patient_id_column"]),
        endpoint_column=str(raw["endpoint_column"]),
        class_mapping={str(k): int(v) for k, v in raw["class_mapping"].items()},
        cv_folds=int(raw["cv_folds"]),
        cv_repeats=int(raw["cv_repeats"]),
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _models(seed: int) -> dict[str, Any]:
    return {
        "logistic": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        solver="liblinear",
                        C=0.5,
                        class_weight="balanced",
                        random_state=seed,
                        max_iter=2000,
                    ),
                ),
            ]
        ),
        "extratrees": ExtraTreesClassifier(
            n_estimators=80,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            random_state=seed,
            n_jobs=1,
        ),
    }


def _fit_repeated_oof(
    X: pd.DataFrame,
    y: np.ndarray,
    patient_ids: list[str],
    seed: int,
    cv_folds: int,
    cv_repeats: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    splitter = RepeatedStratifiedKFold(
        n_splits=cv_folds,
        n_repeats=cv_repeats,
        random_state=seed,
    )
    rows: list[dict[str, Any]] = []
    fold_metrics: list[dict[str, Any]] = []
    for split_index, (train_idx, test_idx) in enumerate(splitter.split(X, y)):
        repeat = split_index // cv_folds + 1
        fold = split_index % cv_folds + 1
        for model_name, model in _models(seed + split_index).items():
            model.fit(X.iloc[train_idx], y[train_idx])
            probability = model.predict_proba(X.iloc[test_idx])[:, 1]
            prediction = (probability >= 0.5).astype(int)
            fold_auc = roc_auc_score(y[test_idx], probability)
            fold_metrics.append(
                {
                    "repeat": repeat,
                    "fold": fold,
                    "model": model_name,
                    "roc_auc": float(fold_auc),
                    "n_test": int(len(test_idx)),
                }
            )
            for local_index, patient_index in enumerate(test_idx):
                rows.append(
                    {
                        "patient_id": patient_ids[patient_index],
                        "repeat": repeat,
                        "fold": fold,
                        "model": model_name,
                        "true_label": int(y[patient_index]),
                        "oof_probability": float(probability[local_index]),
                        "predicted_label": int(prediction[local_index]),
                    }
                )
    oof = pd.DataFrame(rows, columns=OOF_COLUMNS)
    metrics = pd.DataFrame(fold_metrics)
    return oof, metrics


def _patient_reliability(oof: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        oof.groupby(["patient_id", "true_label", "model"], as_index=False)
        .agg(mean_probability=("oof_probability", "mean"), resampling_sd=("oof_probability", "std"))
    )
    mean_pivot = grouped.pivot(index=["patient_id", "true_label"], columns="model", values="mean_probability")
    sd_pivot = grouped.pivot(index=["patient_id", "true_label"], columns="model", values="resampling_sd")
    result = mean_pivot.reset_index()
    result = result.rename(
        columns={"logistic": "mean_prob_logistic", "extratrees": "mean_prob_extratrees"}
    )
    sd_frame = sd_pivot.reset_index().rename(
        columns={"logistic": "resampling_sd_logistic", "extratrees": "resampling_sd_extratrees"}
    )
    result = result.merge(sd_frame, on=["patient_id", "true_label"], validate="one_to_one")
    result["cross_model_difference"] = (
        result["mean_prob_logistic"] - result["mean_prob_extratrees"]
    ).abs()
    result["model_disagreement"] = (
        (result["mean_prob_logistic"] >= 0.5)
        != (result["mean_prob_extratrees"] >= 0.5)
    ).astype(int)
    result = result[PATIENT_OUTPUT_COLUMNS].sort_values("patient_id").reset_index(drop=True)
    validate_output_columns(result, PATIENT_OUTPUT_COLUMNS, "patient_reliability")
    return result


def _representation_audit(
    X: pd.DataFrame,
    y: np.ndarray,
    gene_sets: pd.DataFrame,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    solver="liblinear",
                    C=0.5,
                    class_weight="balanced",
                    random_state=seed,
                    max_iter=2000,
                ),
            ),
        ]
    )
    model.fit(X, y)
    scaled = model.named_steps["scale"].transform(X)
    coefficients = model.named_steps["model"].coef_[0]
    contributions = scaled * coefficients
    mean_absolute = np.abs(contributions).mean(axis=0)
    mean_signed = contributions.mean(axis=0)

    represented = set(gene_sets["gene_id"].astype(str))
    rows = []
    for gene, abs_value, signed_value in zip(X.columns, mean_absolute, mean_signed):
        rows.append(
            {
                "gene_id": gene,
                "represented": int(gene in represented),
                "mean_absolute_contribution": float(abs_value),
                "mean_signed_contribution": float(signed_value),
            }
        )
    gene_table = pd.DataFrame(rows).sort_values("gene_id").reset_index(drop=True)
    total_abs = float(gene_table["mean_absolute_contribution"].sum())
    represented_abs = float(
        gene_table.loc[gene_table["represented"] == 1, "mean_absolute_contribution"].sum()
    )
    represented_signed = float(
        gene_table.loc[gene_table["represented"] == 1, "mean_signed_contribution"].sum()
    )
    total_signed = float(gene_table["mean_signed_contribution"].sum())
    coverage = represented_abs / total_abs if total_abs else 0.0
    summary = pd.DataFrame(
        [
            {
                "n_genes": int(len(gene_table)),
                "n_represented_genes": int(gene_table["represented"].sum()),
                "n_gap_genes": int((gene_table["represented"] == 0).sum()),
                "absolute_contribution_coverage": float(coverage),
                "signed_residual": float(total_signed - represented_signed),
            }
        ]
    )
    gap_table = gene_table.loc[gene_table["represented"] == 0].copy()
    return summary, gap_table


def run_demo(config_path: Path) -> dict[str, Any]:
    config = load_demo_config(config_path)
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    failure_log = output_dir / "failure_log.csv"

    for old_file in output_dir.glob("*"):
        if old_file.is_file():
            old_file.unlink()

    try:
        inputs = validate_and_align_inputs(
            config.expression_path,
            config.clinical_path,
            config.gene_set_path,
            config.gene_id_column,
            config.patient_id_column,
            config.endpoint_column,
            config.class_mapping,
        )
        clinical = inputs.clinical.set_index(config.patient_id_column).loc[inputs.matched_patients]
        y = clinical[config.endpoint_column].astype(str).map(config.class_mapping).to_numpy(dtype=int)
        expression = inputs.expression.set_index(config.gene_id_column)
        X = expression[inputs.matched_patients].T
        X.columns = X.columns.astype(str)
        X = X.astype(float)

        matched = pd.DataFrame(
            {
                "patient_id": inputs.matched_patients,
                "endpoint_label": clinical[config.endpoint_column].astype(str).to_list(),
                "endpoint_code": y,
            }
        )
        matched.to_csv(output_dir / "matched_samples.csv", index=False)
        pd.DataFrame(
            {
                "patient_id": inputs.extra_expression_patients,
                "reason": "expression_only_sample",
            }
        ).to_csv(output_dir / "exclusion_log.csv", index=False)

        oof, fold_metrics = _fit_repeated_oof(
            X,
            y,
            inputs.matched_patients,
            config.seed,
            config.cv_folds,
            config.cv_repeats,
        )
        oof.to_csv(output_dir / "oof_probabilities.csv", index=False, float_format="%.10f")
        fold_metrics.to_csv(output_dir / "fold_metrics.csv", index=False, float_format="%.10f")

        patient_reliability = _patient_reliability(oof)
        patient_reliability.to_csv(
            output_dir / "patient_reliability.csv", index=False, float_format="%.10f"
        )

        representation_summary, gap_genes = _representation_audit(
            X,
            y,
            inputs.gene_sets,
            config.seed,
        )
        representation_summary.to_csv(
            output_dir / "representation_audit.csv", index=False, float_format="%.10f"
        )
        gap_genes.to_csv(
            output_dir / "representation_gap_genes.csv", index=False, float_format="%.10f"
        )

        model_auc = {
            name: float(roc_auc_score(group["true_label"], group["oof_probability"]))
            for name, group in oof.groupby("model")
        }
        summary = {
            "status": "completed",
            "seed": config.seed,
            "n_input_genes": int(X.shape[1]),
            "n_matched_patients": int(X.shape[0]),
            "class_counts": {str(key): int(value) for key, value in pd.Series(y).value_counts().sort_index().items()},
            "cv_folds": config.cv_folds,
            "cv_repeats": config.cv_repeats,
            "models": ["logistic", "extratrees"],
            "oof_rows": int(len(oof)),
            "patient_rows": int(len(patient_reliability)),
            "model_auc": model_auc,
            "model_disagreement_count": int(patient_reliability["model_disagreement"].sum()),
            "representation_coverage": float(
                representation_summary.loc[0, "absolute_contribution_coverage"]
            ),
            "gap_gene_count": int(representation_summary.loc[0, "n_gap_genes"]),
        }
        _write_json(output_dir / "demo_summary.json", summary)
        manifest = {
            **summary,
            "software": "AIDO-BBA",
            "mode": "compact_demo",
            "python": platform.python_version(),
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "config": {
                **asdict(config),
                "expression_path": str(config.expression_path),
                "clinical_path": str(config.clinical_path),
                "gene_set_path": str(config.gene_set_path),
                "output_dir": str(config.output_dir),
            },
            "output_files": sorted(path.name for path in output_dir.iterdir() if path.is_file()),
        }
        _write_json(output_dir / "run_manifest.json", manifest)
        return summary
    except Exception as exc:
        failure_code = "SCHEMA_VALIDATION_FAILED" if isinstance(exc, SchemaError) else "DEMO_EXECUTION_FAILED"
        pd.DataFrame(
            [
                {
                    "module": "M0" if isinstance(exc, SchemaError) else "DEMO",
                    "failure_code": failure_code,
                    "severity": "error",
                    "message": str(exc),
                    "affected_count": 1,
                }
            ]
        ).to_csv(failure_log, index=False)
        _write_json(
            output_dir / "run_manifest.json",
            {
                "software": "AIDO-BBA",
                "mode": "compact_demo",
                "status": "failed",
                "failure_code": failure_code,
                "message": str(exc),
            },
        )
        raise
