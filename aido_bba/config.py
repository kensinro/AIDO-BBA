"""Portable path configuration for AIDO-BBA scripts.

Resolution order:
1. Environment variable named by each helper.
2. ``config.local.json`` in the repository root (or AIDO_BBA_CONFIG).
3. A repository-relative fallback.
"""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path
import json
import os
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

@lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    config_path = Path(os.environ.get("AIDO_BBA_CONFIG", REPO_ROOT / "config.local.json"))
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    return {}

def get_path(config_key: str, env_key: str, fallback: str | Path) -> Path:
    env_value = os.environ.get(env_key)
    if env_value:
        return Path(env_value).expanduser()
    value = load_config().get(config_key)
    if value:
        return Path(value).expanduser()
    fallback_path = Path(fallback)
    return fallback_path if fallback_path.is_absolute() else REPO_ROOT / fallback_path

def data_root() -> Path:
    return get_path("data_root", "AIDO_DATA_ROOT", "data")

def brca_output_root() -> Path:
    return get_path("brca_output_root", "AIDO_BRCA_OUTPUT_ROOT", "outputs/AIDO_BBA_BRCA_1_0")

def kirc_output_root() -> Path:
    return get_path("kirc_output_root", "AIDO_KIRC_OUTPUT_ROOT", "outputs/AIDO_BBA_KIRC_1_0")

def brca_run_dir() -> Path:
    configured = get_path("brca_run_dir", "AIDO_BRCA_RUN_DIR", "outputs/AIDO_BBA_BRCA_1_0")
    if configured.is_dir() and configured.name.startswith("RUN_"):
        return configured
    candidates = sorted(
        [p for p in configured.glob("RUN_*") if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No BRCA run directory found under {configured}. "
            "Set brca_run_dir in config.local.json or AIDO_BRCA_RUN_DIR."
        )
    return candidates[0]

def go_bp_gmt() -> Path:
    return get_path("go_bp_gmt", "AIDO_GO_BP_GMT", data_root() / "GSEA/c5.go.bp.v2026.1.Hs.symbols.gmt")

def hgnc_file() -> Path:
    return get_path("hgnc_file", "AIDO_HGNC_FILE", data_root() / "HGNC/hgnc_complete_set.txt")

def ncbi_gene_info() -> Path:
    return get_path("ncbi_gene_info", "AIDO_NCBI_GENE_INFO", data_root() / "NCBI_Gene/Homo_sapiens.gene_info")

def metabric_dir() -> Path:
    return get_path("metabric_dir", "AIDO_METABRIC_DIR", data_root() / "External/brca_metabric")

def gse96058_dir() -> Path:
    return get_path("gse96058_dir", "AIDO_GSE96058_DIR", data_root() / "External/GSE96058")

def kirc_dir() -> Path:
    return get_path("kirc_dir", "AIDO_KIRC_DIR", data_root() / "UCSC_XENA/Kidney Clear Cell Carcinoma (KIRC)")
