"""Sparsity-based subset selection (ENSGA-II baseline driver): Eq. 15-17
scoring and subset-selection logic."""

import pytest

from sparsity_local_search import SparsityLocalSearch


def _r(cmax, twet, nec, chromosome=None):
    return {"chromosome": chromosome, "Cmax": cmax, "TWET": twet, "NEC": nec}


# ------------------------------------------------------------
# Construction
# ------------------------------------------------------------

def test_rejects_out_of_range_threshold():
    with pytest.raises(ValueError):
        SparsityLocalSearch(judgment_threshold=0.0)
    with pytest.raises(ValueError):
        SparsityLocalSearch(judgment_threshold=1.0)
    with pytest.raises(ValueError):
        SparsityLocalSearch(judgment_threshold=-0.1)


def test_default_threshold_matches_paper_tuned_value():
    sls = SparsityLocalSearch()
    assert sls.judgment_threshold == 0.2


# ------------------------------------------------------------
# Sparsity scoring (Eqs. 15-17)
# ------------------------------------------------------------

def test_identical_points_are_maximally_crowded():
    sls = SparsityLocalSearch(judgment_threshold=0.5)
    results = [_r(10, 10, 10) for _ in range(4)]
    scores = sls._sparsity_scores(results)
    # Every point coincides with every other -> distance 0 < k for all pairs.
    assert scores == [3 / 4] * 4


def test_isolated_point_has_zero_sparsity_score():
    sls = SparsityLocalSearch(judgment_threshold=0.1)
    # Three points clustered near the origin, one far away on its own axis.
    results = [
        _r(0, 0, 0),
        _r(0, 0, 1),
        _r(0, 1, 0),
        _r(1000, 1000, 1000),
    ]
    scores = sls._sparsity_scores(results)
    # The far point has no neighbors within the (small) threshold.
    assert scores[3] == 0.0
    # The far point is strictly the sparsest (smallest score).
    assert scores[3] == min(scores)


def test_select_subset_picks_sparsest_individuals():
    sls = SparsityLocalSearch(subset_size=1, judgment_threshold=0.1)
    results = [
        _r(0, 0, 0),
        _r(0, 0, 1),     # crowded cluster with index 0
        _r(1000, 1000, 1000),  # sparsest
    ]
    subset = sls._select_subset(results)
    assert subset == [2]


def test_select_subset_no_trim_when_population_not_larger_than_budget():
    sls = SparsityLocalSearch(subset_size=5)
    results = [_r(i, i, i) for i in range(3)]
    assert sls._select_subset(results) == [0, 1, 2]
