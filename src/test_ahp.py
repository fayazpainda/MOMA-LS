"""Fast unit tests for the AHP preference-weighting module.

No optimisation is involved -- these are pure matrix -> weights -> CR checks
and run in well under a second.
"""

import warnings

import numpy as np
import pytest

from ahp import (
    DEFAULT_PREFERENCE_MATRIX,
    MANAGERIAL_PRIORITY_ORDER,
    MANAGERIAL_SCENARIOS,
    OBJECTIVE_ORDER,
    consistency_ratio,
    derive_weights,
    geometric_mean_weights,
    managerial_scenario_report,
    managerial_scenario_weights,
    named_weights,
)


def test_weights_sum_to_one_and_are_positive():
    weights = derive_weights(DEFAULT_PREFERENCE_MATRIX)

    assert len(weights) == 3
    assert np.all(weights > 0.0)
    assert pytest.approx(1.0, abs=1e-12) == float(weights.sum())


def test_default_matrix_ordering_reflects_preference():
    # Example matrix: Cmax > TWET > NEC in priority.
    weights = derive_weights(DEFAULT_PREFERENCE_MATRIX)

    w_cmax, w_twet, w_nec = weights
    assert w_cmax > w_twet > w_nec, (
        f"Expected Cmax > TWET > NEC priority ordering; got {weights.tolist()}"
    )


def test_geometric_mean_matches_hand_computation():
    # A simple, perfectly reciprocal, exactly-consistent matrix:
    #   A over (x, y, z) with x twice y, y twice z, x four times z.
    matrix = [
        [1.0, 2.0, 4.0],
        [0.5, 1.0, 2.0],
        [0.25, 0.5, 1.0],
    ]

    weights = geometric_mean_weights(matrix)

    # For a consistent matrix the geometric-mean weights are exact:
    # proportional to (4, 2, 1) -> /7.
    expected = np.array([4.0, 2.0, 1.0]) / 7.0
    assert np.allclose(weights, expected, atol=1e-12)


def test_consistent_matrix_has_zero_cr():
    # The (4,2,1) matrix above is perfectly consistent -> CR ~ 0, CI ~ 0,
    # lambda_max ~ n = 3.
    matrix = [
        [1.0, 2.0, 4.0],
        [0.5, 1.0, 2.0],
        [0.25, 0.5, 1.0],
    ]

    diagnostics = consistency_ratio(matrix)

    assert diagnostics["n"] == 3
    assert diagnostics["RI"] == pytest.approx(0.58)
    assert diagnostics["lambda_max"] == pytest.approx(3.0, abs=1e-9)
    assert diagnostics["CI"] == pytest.approx(0.0, abs=1e-9)
    assert diagnostics["CR"] == pytest.approx(0.0, abs=1e-9)


def test_default_matrix_is_reasonably_consistent():
    # The placeholder default should pass the CR < 0.1 check (no warning path).
    _weights, diagnostics = derive_weights(
        DEFAULT_PREFERENCE_MATRIX, return_diagnostics=True
    )
    assert diagnostics["CR"] < 0.1


def test_inconsistent_matrix_warns_but_still_returns_weights():
    # Deliberately intransitive: x >> y, y >> z, yet z >> x. High CR.
    inconsistent = [
        [1.0, 9.0, 1.0 / 9.0],
        [1.0 / 9.0, 1.0, 9.0],
        [9.0, 1.0 / 9.0, 1.0],
    ]

    with pytest.warns(UserWarning, match="consistency ratio"):
        weights = derive_weights(inconsistent)

    # Still a valid weight vector despite the inconsistency.
    assert pytest.approx(1.0, abs=1e-12) == float(weights.sum())
    assert np.all(weights > 0.0)

    diagnostics = consistency_ratio(inconsistent)
    assert diagnostics["CR"] >= 0.1


def test_named_weights_keys_and_alignment():
    named = named_weights(DEFAULT_PREFERENCE_MATRIX)

    assert tuple(named.keys()) == OBJECTIVE_ORDER
    assert pytest.approx(1.0, abs=1e-12) == sum(named.values())
    # Cmax is the highest-priority objective in the default matrix.
    assert named["Cmax"] == max(named.values())


def test_bad_matrix_shape_rejected():
    with pytest.raises(ValueError):
        derive_weights([[1.0, 2.0], [0.5, 1.0], [1.0, 1.0]])


def test_non_positive_entry_rejected():
    with pytest.raises(ValueError):
        derive_weights([
            [1.0, 0.0, 5.0],
            [1.0, 1.0, 3.0],
            [0.2, 1.0 / 3.0, 1.0],
        ])


def test_non_unit_diagonal_rejected():
    with pytest.raises(ValueError):
        derive_weights([
            [2.0, 3.0, 5.0],
            [1.0 / 3.0, 1.0, 3.0],
            [0.2, 1.0 / 3.0, 1.0],
        ])


def test_reciprocity_is_not_structurally_required():
    # A non-reciprocal but positive, unit-diagonal matrix must be *accepted*
    # (no ValueError): reciprocity is not enforced. Whether it warns is a
    # separate CR question, so we silence warnings and only check the matrix
    # is turned into a valid weight vector.
    matrix = [
        [1.0, 3.0, 5.0],
        [0.4, 1.0, 3.0],   # not exactly 1/3
        [0.25, 0.4, 1.0],  # not exactly 1/5, 1/3
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        weights = derive_weights(matrix)
    assert pytest.approx(1.0, abs=1e-12) == float(weights.sum())
    assert np.all(weights > 0.0)


def test_managerial_scenarios_cover_all_six_priority_orders_plus_balanced():
    orders = {
        tuple(order) for name, order in MANAGERIAL_PRIORITY_ORDER.items()
        if name != "balanced_operations"
    }
    assert len(orders) == 6  # all 3! strict orderings of (Cmax, TWET, NEC)
    assert len(MANAGERIAL_SCENARIOS) == 7


def test_managerial_scenarios_are_consistent_and_sum_to_one():
    for name, weights in managerial_scenario_weights().items():
        assert pytest.approx(1.0, abs=1e-9) == float(np.sum(weights))
        assert np.all(np.asarray(weights) > 0.0)
        diagnostics = consistency_ratio(MANAGERIAL_SCENARIOS[name])
        assert diagnostics["CR"] < 0.1


def test_managerial_scenario_weights_match_stated_priority_order():
    weights = managerial_scenario_weights()
    for name, order in MANAGERIAL_PRIORITY_ORDER.items():
        if name == "balanced_operations":
            continue
        w = dict(zip(OBJECTIVE_ORDER, weights[name]))
        ranked = sorted(order, key=lambda obj: -w[obj])
        assert list(order) == ranked


def test_managerial_scenario_report_has_one_row_per_scenario():
    rows = managerial_scenario_report()
    assert len(rows) == len(MANAGERIAL_SCENARIOS)
    names = {row["scenario"] for row in rows}
    assert names == set(MANAGERIAL_SCENARIOS)
    for row in rows:
        assert pytest.approx(1.0, abs=1e-9) == (
            row["w_Cmax"] + row["w_TWET"] + row["w_NEC"]
        )
