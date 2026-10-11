"""Fast unit tests for the 3-D hypervolume convergence indicator."""

import pytest

from hypervolume import hypervolume, normalized_hypervolume


def test_single_point_is_the_box_volume():
    # One point at origin, reference at (1,1,1) -> full unit cube.
    assert hypervolume([(0.0, 0.0, 0.0)], (1.0, 1.0, 1.0)) == pytest.approx(1.0)


def test_single_point_partial_box():
    # Point (2,3,4), reference (5,5,5) -> box 3*2*1 = 6.
    assert hypervolume([(2.0, 3.0, 4.0)], (5.0, 5.0, 5.0)) == pytest.approx(6.0)


def test_point_outside_reference_contributes_zero():
    # Point not strictly below reference in every objective.
    assert hypervolume([(5.0, 1.0, 1.0)], (5.0, 5.0, 5.0)) == pytest.approx(0.0)
    assert hypervolume([(6.0, 1.0, 1.0)], (5.0, 5.0, 5.0)) == pytest.approx(0.0)


def test_dominated_point_adds_nothing():
    # (1,1,1) dominates (2,2,2) under minimisation; adding the dominated point
    # must not change the HV.
    ref = (5.0, 5.0, 5.0)
    hv_one = hypervolume([(1.0, 1.0, 1.0)], ref)
    hv_two = hypervolume([(1.0, 1.0, 1.0), (2.0, 2.0, 2.0)], ref)
    assert hv_one == pytest.approx(hv_two)
    assert hv_one == pytest.approx(4.0 ** 3)


def test_two_non_dominated_points_inclusion_exclusion():
    # Minimisation, reference (5,5,5). Two trade-off points along z fixed:
    #   a = (1, 4, 1) -> box (4,1,4)  volume 16
    #   b = (4, 1, 1) -> box (1,4,4)  volume 16
    # overlap box = (1,1,4) volume 4 ; union = 16 + 16 - 4 = 28.
    ref = (5.0, 5.0, 5.0)
    hv = hypervolume([(1.0, 4.0, 1.0), (4.0, 1.0, 1.0)], ref)
    assert hv == pytest.approx(28.0)


def test_hv_is_monotone_in_convergence():
    # A front closer to the ideal must have >= HV than a worse front.
    ref = (10.0, 10.0, 10.0)
    good = [(1.0, 1.0, 1.0), (2.0, 1.0, 3.0)]
    bad = [(5.0, 5.0, 5.0), (6.0, 5.0, 7.0)]
    assert hypervolume(good, ref) > hypervolume(bad, ref)


def test_normalized_hv_in_unit_interval_and_ordering():
    ideal = (0.0, 0.0, 0.0)
    nadir = (10.0, 10.0, 10.0)

    good = normalized_hypervolume([(1.0, 1.0, 1.0)], ideal, nadir)
    bad = normalized_hypervolume([(9.0, 9.0, 9.0)], ideal, nadir)

    assert 0.0 <= bad < good <= 1.0


def test_normalized_hv_handles_degenerate_objective():
    # Third objective is constant (nadir == ideal): should not blow up.
    ideal = (0.0, 0.0, 5.0)
    nadir = (10.0, 10.0, 5.0)
    value = normalized_hypervolume([(1.0, 1.0, 5.0)], ideal, nadir)
    assert 0.0 <= value <= 1.0


def test_empty_front_is_zero():
    assert hypervolume([], (1.0, 1.0, 1.0)) == 0.0
    assert normalized_hypervolume([], (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)) == 0.0
