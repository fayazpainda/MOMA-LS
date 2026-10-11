"""Fast unit tests for structured reference-point generation.

No optimisation involved -- pure grid construction and filtering, runs in
well under a second.
"""

import warnings

import numpy as np
import pytest

from ahp import derive_weights
from reference_points import (
    DEFAULT_RADIUS,
    MIN_RECOMMENDED_POINTS,
    das_dennis_grid,
    generate_reference_points,
    reference_target_from_weights,
)
from unsga3 import generate_reference_points_3d


def test_uniform_matches_validated_baseline_exactly():
    # mode="uniform" must be byte-for-byte identical to the historical grid so
    # the validated U-NSGA-III baseline is unchanged.
    baseline = generate_reference_points_3d(12)
    uniform = generate_reference_points(mode="uniform", p=12)
    assert uniform == baseline
    assert len(uniform) == 91  # H = C(14, 2)


def test_uniform_points_sum_to_one():
    for point in generate_reference_points(mode="uniform", p=12):
        assert pytest.approx(1.0, abs=1e-9) == sum(point)


def test_das_dennis_count_formula():
    # H = C(p+2, 2) for M = 3.
    for p in (4, 6, 12):
        expected = (p + 2) * (p + 1) // 2
        assert len(das_dennis_grid(p)) == expected


def test_ahp_guided_is_a_strict_subset_within_radius():
    w = derive_weights()  # default (Cmax-dominant) weights
    grid = generate_reference_points(mode="uniform", p=12)
    filtered = generate_reference_points(
        mode="ahp_guided", weight_vector=w, radius=DEFAULT_RADIUS
    )

    # Subset of the grid.
    grid_set = {tuple(round(x, 12) for x in p) for p in grid}
    for point in filtered:
        assert tuple(round(x, 12) for x in point) in grid_set

    # Strictly fewer than the full grid, but non-trivial.
    assert 0 < len(filtered) < len(grid)

    # Every retained direction is within the radius of the weight vector.
    target = np.asarray(w, dtype=float)
    for point in filtered:
        assert np.linalg.norm(np.asarray(point) - target) <= DEFAULT_RADIUS + 1e-9


def test_ahp_guided_points_sum_to_one():
    w = derive_weights()
    for point in generate_reference_points(
        mode="ahp_guided", weight_vector=w, radius=DEFAULT_RADIUS
    ):
        assert pytest.approx(1.0, abs=1e-9) == sum(point)


def test_default_radius_keeps_N_greater_than_H_and_meets_floor():
    # With N = 100, the retained set must stay below N and (for the default
    # weights + radius) at or above the diversity floor.
    w = derive_weights()
    filtered = generate_reference_points(
        mode="ahp_guided", weight_vector=w, radius=DEFAULT_RADIUS
    )
    assert len(filtered) < 100
    assert len(filtered) >= MIN_RECOMMENDED_POINTS


def test_small_radius_warns_about_diversity():
    w = derive_weights()
    with pytest.warns(UserWarning, match="diversity floor"):
        filtered = generate_reference_points(
            mode="ahp_guided", weight_vector=w, radius=0.12
        )
    assert 0 < len(filtered) < MIN_RECOMMENDED_POINTS


def test_tiny_radius_falls_back_to_nearest_and_warns():
    w = derive_weights()
    with pytest.warns(UserWarning, match="retained no grid directions"):
        filtered = generate_reference_points(
            mode="ahp_guided", weight_vector=w, radius=1e-6
        )
    assert len(filtered) == 1


def test_ahp_guided_requires_weight_vector():
    with pytest.raises(ValueError, match="requires a weight_vector"):
        generate_reference_points(mode="ahp_guided", radius=DEFAULT_RADIUS)


def test_unknown_mode_rejected():
    with pytest.raises(ValueError, match="Unknown mode"):
        generate_reference_points(mode="not_a_mode")


def test_angular_metric_requires_explicit_radius():
    w = derive_weights()
    with pytest.raises(ValueError, match="radius must be supplied"):
        generate_reference_points(
            mode="ahp_guided", weight_vector=w, metric="angular"
        )


def test_reference_target_inverts_priority():
    # A Cmax-priority weight (large w_Cmax) must map to a reference target with
    # the SMALLEST Cmax coordinate, so the guided search attracts LOW-Cmax
    # (good) solutions.
    w = derive_weights()  # cmax-dominant default ~ (0.637, 0.258, 0.105)
    target = reference_target_from_weights(w)

    assert pytest.approx(1.0, abs=1e-9) == sum(target)
    # Priority ordering is inverted: most-important objective -> smallest coord.
    assert target[0] < target[1] < target[2]
    # Argmin of the target aligns with the argmax of the weights.
    assert target.index(min(target)) == list(w).index(max(w))


def test_reference_target_leaves_balanced_centered():
    target = reference_target_from_weights([1 / 3, 1 / 3, 1 / 3])
    assert target == pytest.approx([1 / 3, 1 / 3, 1 / 3])


def test_angular_metric_filters_when_radius_given():
    w = derive_weights()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        filtered = generate_reference_points(
            mode="ahp_guided", weight_vector=w, radius=0.35, metric="angular"
        )
    assert 0 < len(filtered) < 91
