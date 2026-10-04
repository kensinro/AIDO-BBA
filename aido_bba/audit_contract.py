"""Canonical repaired audit definitions for AIDO-BBA v1.1.0.

These functions encode the scientific definitions locked during the 2026-10-04
forensic repair. They are intentionally small and dependency-light so that the
manuscript-facing quantities can be tested independently of the legacy pipeline.

Historical v1.0.1 outputs remain provenance artifacts and are not silently
rewritten by this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


REPRESENTATION_EXPANSION_THRESHOLD = -0.005
RESIDUAL_DIRECTION_THRESHOLD = 0.005
REPEAT_CONSISTENCY_THRESHOLD = 0.60


@dataclass(frozen=True)
class StatisticalFamilyManifest:
    """Repaired multiplicity-family counts after invalid groupings were removed."""

    representation_gap_omnibus_tested: int = 18
    representation_gap_omnibus_significant: int = 14
    representation_gap_pairwise_tested: int = 84
    representation_gap_pairwise_significant: int = 44
    representation_gap_pairwise_medium_large: int = 16

    anchor_boundary_omnibus_tested: int = 36
    anchor_boundary_omnibus_significant: int = 23
    anchor_boundary_pairwise_tested: int = 168
    anchor_boundary_pairwise_significant: int = 95
    anchor_boundary_pairwise_medium_large: int = 47


REPAIRED_STATISTICAL_FAMILIES = StatisticalFamilyManifest()


def attribution_mass_coverage(
    gene_shap: Sequence[float] | np.ndarray,
    mapped_mask: Sequence[bool] | np.ndarray,
) -> float:
    """Return canonical gene-level absolute attribution-mass coverage C_i.

    C_i = sum_g(mapped_g * |phi_ig|) / sum_g(|phi_ig|)

    This is a gene-level mass fraction. It is not the absolute mass of a
    reconstructed process-level sum.
    """

    shap = np.asarray(gene_shap, dtype=float)
    mask = np.asarray(mapped_mask, dtype=bool)
    if shap.ndim != 1 or mask.ndim != 1 or shap.shape != mask.shape:
        raise ValueError("gene_shap and mapped_mask must be one-dimensional and aligned")

    total = float(np.abs(shap).sum())
    if total <= 0.0:
        return float("nan")
    mapped = float(np.abs(shap[mask]).sum())
    return mapped / total


def rank_probability_separation(
    model_a_percentile: Sequence[float] | np.ndarray,
    model_b_percentile: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Return calibration-insensitive cross-model rank disagreement D_rank.

    D_rank_i = |rankpct_A_i - rankpct_B_i|.

    The input percentile vectors must be patient-aligned. Unlike raw probability
    separation, this quantity is invariant to monotone rescaling of each model's
    probability output.
    """

    a = np.asarray(model_a_percentile, dtype=float)
    b = np.asarray(model_b_percentile, dtype=float)
    if a.shape != b.shape:
        raise ValueError("model percentile vectors must be patient-aligned")
    return np.abs(a - b)


def fuzzy_state_label(
    memberships: Iterable[float],
    *,
    high: float = 0.75,
    low: float = 0.25,
    dominance_margin: float = 0.15,
) -> str:
    """Classify independent per-axis fuzzy membership magnitudes.

    Memberships are *not* constrained to a simplex and need not sum to one.
    Precedence reproduces the source-verified execution contract:
      1. >=2 memberships >= high -> shared_high
      2. exactly one >= high and top-second >= dominance_margin -> dominant
      3. all memberships <= low -> shared_low
      4. otherwise -> blended_intermediate
    """

    values = np.asarray(list(memberships), dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("memberships must be a finite one-dimensional vector")

    high_idx = np.flatnonzero(values >= high)
    if high_idx.size >= 2:
        pair = "_".join(str(i + 1) for i in high_idx[:2])
        return f"shared_high_{pair}"

    order = np.sort(values)[::-1]
    if high_idx.size == 1 and (values.size == 1 or order[0] - order[1] >= dominance_margin):
        return f"dominant_GapCore_{int(high_idx[0]) + 1:02d}"

    if np.all(values <= low):
        return "shared_low"

    return "blended_intermediate"


def measurement_triage_flags(
    *,
    excess_coverage_mean: float,
    excess_signed_residual_mean: float,
    repeat_state_consistency: float,
    repeat_instability_tier: str,
    label_model_discordance: bool,
) -> dict[str, bool]:
    """Return the four source-verified repaired measurement-triage flags.

    The historical ``model_arbitration`` action is deliberately absent because
    it was driven by calibration-confounded raw probability separation and is
    superseded by continuous D_rank review.
    """

    return {
        "representation_expansion": bool(
            excess_coverage_mean < REPRESENTATION_EXPANSION_THRESHOLD
        ),
        "residual_direction_review": bool(
            abs(excess_signed_residual_mean) > RESIDUAL_DIRECTION_THRESHOLD
        ),
        "repeat_or_orthogonal_measurement": bool(
            repeat_state_consistency < REPEAT_CONSISTENCY_THRESHOLD
            or repeat_instability_tier == "high_repeat_instability"
        ),
        "label_model_discrepancy_review": bool(label_model_discordance),
    }
