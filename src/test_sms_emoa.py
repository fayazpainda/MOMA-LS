"""SMS-EMOA comparator: hypervolume-contribution front reduction and
environmental selection correctness, plus end-to-end integration."""

import random
import time

from wei_extended_loader import load_wei_extended_instance
from pareto import get_pareto_front
from pareto_local_search import ParetoLocalSearch
from sms_emoa import (
    reduce_front_by_hv_contribution,
    sms_emoa_environmental_selection,
    run_SMS_EMOA,
)


def _r(cmax, twet, nec, chromosome=None):
    return {"chromosome": chromosome, "Cmax": cmax, "TWET": twet, "NEC": nec}


def _required_multiset(instance):
    return sorted(
        job.job_id for job in instance.jobs for _ in job.operations
    )


# ------------------------------------------------------------
# Hypervolume-contribution front reduction
# ------------------------------------------------------------

def test_reduce_front_keeps_target_size():
    front = [_r(10, 30, 20), _r(20, 10, 30), _r(30, 20, 10), _r(15, 15, 15)]
    reduced = reduce_front_by_hv_contribution(front, 2)
    assert len(reduced) == 2


def test_reduce_front_no_op_when_already_small_enough():
    front = [_r(10, 30, 20), _r(20, 10, 30)]
    reduced = reduce_front_by_hv_contribution(front, 2)
    assert reduced == front


def test_reduce_large_front_stays_fast():
    # Regression test: a splitting front approaching the combined-population
    # size (2 * population_size, e.g. 200) made the naive all-exact
    # hypervolume-contribution removal take several minutes for a single
    # generation during development. The two-stage crowding-precut strategy
    # must keep this within a couple of seconds.
    random.seed(0)
    front = [
        _r(random.uniform(0, 5000), random.uniform(0, 500000), random.uniform(0, 9000))
        for _ in range(180)
    ]
    start = time.time()
    reduced = reduce_front_by_hv_contribution(front, 100)
    elapsed = time.time() - start
    assert len(reduced) == 100
    assert elapsed < 5.0, f"large-front reduction took {elapsed:.1f}s, expected < 5s"


def test_reduce_front_removes_near_duplicate_first():
    # A point nearly identical to one extreme is almost entirely redundant
    # with it (their hypervolume coverage overlaps heavily), so it should
    # have the smaller marginal contribution of the two and be removed
    # before the genuinely distinct extreme.
    extreme_a = _r(0, 100, 100)
    extreme_b = _r(100, 0, 100)
    near_duplicate_of_a = _r(5, 95, 100)
    front = [extreme_a, extreme_b, near_duplicate_of_a]
    reduced = reduce_front_by_hv_contribution(front, 2)
    assert extreme_a not in reduced
    assert extreme_b in reduced
    assert near_duplicate_of_a in reduced


# ------------------------------------------------------------
# Environmental selection
# ------------------------------------------------------------

def test_environmental_selection_keeps_whole_fronts_that_fit():
    # front 0 (2 mutually non-dominated) + front 1 (1 dominated by both).
    a = _r(0, 10, 10)
    b = _r(10, 0, 10)
    c = _r(20, 20, 20)  # dominated by both a and b
    selected = sms_emoa_environmental_selection([a, b, c], 2)
    assert len(selected) == 2
    assert a in selected and b in selected
    assert c not in selected


def test_environmental_selection_reduces_splitting_front():
    front0 = [_r(0, 100, 100), _r(100, 0, 100), _r(50, 50, 100)]
    selected = sms_emoa_environmental_selection(front0, 2)
    assert len(selected) == 2


# ------------------------------------------------------------
# End-to-end integration
# ------------------------------------------------------------

def test_run_sms_emoa_produces_feasible_population():
    instance, due_dates, weights, machines = load_wei_extended_instance("ta01")

    random.seed(7)
    population = run_SMS_EMOA(
        instance,
        generations=2,
        population_size=12,
        due_dates=due_dates,
        weights=weights,
        machines=machines,
        verbose=False,
    )

    assert len(population) == 12
    required = _required_multiset(instance)
    for chromo in population:
        assert sorted(chromo.operation_sequence) == required


def test_run_sms_emoa_with_local_search_hook():
    instance, due_dates, weights, machines = load_wei_extended_instance("ta01")
    pls = ParetoLocalSearch(subset_size=4, max_moves=3)

    random.seed(13)
    population = run_SMS_EMOA(
        instance,
        generations=2,
        population_size=12,
        due_dates=due_dates,
        weights=weights,
        machines=machines,
        local_search=pls,
        verbose=False,
    )

    assert len(population) == 12
    required = _required_multiset(instance)
    for chromo in population:
        assert sorted(chromo.operation_sequence) == required
    assert pls.evaluations > 0
