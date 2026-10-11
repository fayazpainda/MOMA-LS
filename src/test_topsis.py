"""Fast unit tests for TOPSIS a-posteriori selection (cost criteria)."""

import pytest

from topsis import topsis_scores, topsis_select


def test_empty_and_single():
    assert topsis_scores([], [1, 1, 1]) == []
    assert topsis_scores([(1.0, 2.0, 3.0)], [1, 1, 1]) == [1.0]
    idx, alt, scores = topsis_select([(1.0, 2.0, 3.0)], [1, 1, 1])
    assert idx == 0 and alt == (1.0, 2.0, 3.0) and scores == [1.0]


def test_dominant_alternative_wins():
    # One alternative is best on every (cost) objective -> it must be selected
    # and score 1.0, the fully-dominated one score 0.0.
    alts = [(1.0, 1.0, 1.0), (9.0, 9.0, 9.0)]
    idx, alt, scores = topsis_select(alts, [1, 1, 1])
    assert idx == 0
    assert alt == (1.0, 1.0, 1.0)
    assert scores[0] == pytest.approx(1.0)
    assert scores[1] == pytest.approx(0.0)


def test_weights_shift_the_choice():
    # Two trade-off alternatives: A is cheap on objective 0, expensive on 1;
    # B is the reverse. Objective 2 identical. Weighting objective 0 heavily
    # should pick the one that is best on objective 0 (A); weighting objective
    # 1 heavily should pick B.
    a = (1.0, 9.0, 5.0)
    b = (9.0, 1.0, 5.0)
    alts = [a, b]

    idx_obj0, _, _ = topsis_select(alts, [0.9, 0.05, 0.05])
    idx_obj1, _, _ = topsis_select(alts, [0.05, 0.9, 0.05])

    assert idx_obj0 == 0  # prioritising objective 0 picks A
    assert idx_obj1 == 1  # prioritising objective 1 picks B


def test_scores_in_unit_interval():
    alts = [(2.0, 3.0, 4.0), (3.0, 2.0, 5.0), (4.0, 4.0, 2.0)]
    scores = topsis_scores(alts, [0.4, 0.4, 0.2])
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_constant_objective_is_ignored_safely():
    # Third objective constant across alternatives -> zero denominator path.
    alts = [(1.0, 4.0, 7.0), (4.0, 1.0, 7.0)]
    scores = topsis_scores(alts, [0.5, 0.5, 0.0])
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_weights_renormalised():
    # Unnormalised weights must give the same ranking as normalised ones.
    alts = [(1.0, 9.0, 5.0), (9.0, 1.0, 5.0)]
    a = topsis_scores(alts, [2.0, 1.0, 1.0])
    b = topsis_scores(alts, [0.5, 0.25, 0.25])
    assert a == pytest.approx(b)


def test_zero_weight_sum_rejected():
    with pytest.raises(ValueError):
        topsis_scores([(1.0, 2.0, 3.0), (2.0, 1.0, 3.0)], [0.0, 0.0, 0.0])
