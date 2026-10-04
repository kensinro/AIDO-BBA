import math

import numpy as np

from aido_bba.audit_contract import (
    REPAIRED_STATISTICAL_FAMILIES,
    attribution_mass_coverage,
    fuzzy_state_label,
    measurement_triage_flags,
    rank_probability_separation,
)


def test_attribution_mass_coverage_is_gene_level_absolute_mass_fraction():
    shap = np.array([1.0, -2.0, 3.0, -4.0])
    mapped = np.array([True, False, True, False])
    assert math.isclose(attribution_mass_coverage(shap, mapped), 0.4)


def test_rank_probability_separation():
    a = np.array([0.1, 0.7, 0.9])
    b = np.array([0.4, 0.6, 0.2])
    np.testing.assert_allclose(rank_probability_separation(a, b), [0.3, 0.1, 0.7])


def test_fuzzy_memberships_are_not_simplex_constrained():
    # This historically observed pattern is satisfiable only when memberships
    # are independent per-axis magnitudes rather than proportions summing to 1.
    assert fuzzy_state_label([0.954, 0.782, 0.507]) == "shared_high_1_2"


def test_fuzzy_dominant_and_blended_precedence():
    assert fuzzy_state_label([0.83, 0.61, 0.42]) == "dominant_GapCore_01"
    assert fuzzy_state_label([0.70, 0.62, 0.45]) == "blended_intermediate"


def test_repaired_measurement_triage_contract():
    flags = measurement_triage_flags(
        excess_coverage_mean=-0.006,
        excess_signed_residual_mean=0.006,
        repeat_state_consistency=0.75,
        repeat_instability_tier="high_repeat_instability",
        label_model_discordance=True,
    )
    assert flags == {
        "representation_expansion": True,
        "residual_direction_review": True,
        "repeat_or_orthogonal_measurement": True,
        "label_model_discrepancy_review": True,
    }
    assert "model_arbitration" not in flags


def test_repaired_statistical_family_manifest():
    m = REPAIRED_STATISTICAL_FAMILIES
    assert (m.representation_gap_omnibus_tested, m.representation_gap_omnibus_significant) == (18, 14)
    assert (m.representation_gap_pairwise_tested, m.representation_gap_pairwise_significant) == (84, 44)
    assert m.representation_gap_pairwise_medium_large == 16
    assert (m.anchor_boundary_omnibus_tested, m.anchor_boundary_omnibus_significant) == (36, 23)
    assert (m.anchor_boundary_pairwise_tested, m.anchor_boundary_pairwise_significant) == (168, 95)
    assert m.anchor_boundary_pairwise_medium_large == 47
