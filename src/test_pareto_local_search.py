"""Pareto Local Search driver: acceptance/archive logic and integration."""

import random

from claude_dataset import load_claude_extended_instance
from chromosome import generate_chromosome
from evaluation import evaluate_individual
from unsga3 import run_UNSGAIII
from pareto_local_search import (
    ParetoLocalSearch,
    dominance_accept,
)


def _r(cmax, twet, nec, chromosome=None):
    return {"chromosome": chromosome, "Cmax": cmax, "TWET": twet, "NEC": nec}


def _required_multiset(instance):
    return sorted(
        job.job_id
        for job in instance.jobs
        for _ in job.operations
    )


# ------------------------------------------------------------
# Pure-logic pieces (no decoding)
# ------------------------------------------------------------

def test_dominance_accept():
    current = _r(100, 100, 100)
    assert dominance_accept(current, _r(90, 100, 100)) is True    # dominates
    assert dominance_accept(current, _r(100, 100, 100)) is False  # equal
    assert dominance_accept(current, _r(90, 110, 100)) is False   # trade-off
    assert dominance_accept(current, _r(110, 110, 110)) is False  # dominated


def test_filter_archive_dedupes_and_keeps_nondominated():
    pls = ParetoLocalSearch()
    archive = [
        _r(10, 10, 10),
        _r(10, 10, 10),   # duplicate objective vector
        _r(20, 5, 10),    # non-dominated w.r.t the first
        _r(30, 30, 30),   # dominated by the first -> dropped
    ]
    front = pls._filter_archive(archive)
    vectors = sorted((r["Cmax"], r["TWET"], r["NEC"]) for r in front)
    assert vectors == [(10, 10, 10), (20, 5, 10)]


def test_filter_archive_cap():
    pls = ParetoLocalSearch(max_archive=2)
    # Three mutually non-dominated points.
    archive = [_r(10, 30, 20), _r(20, 10, 30), _r(30, 20, 10)]
    assert len(pls._filter_archive(archive)) == 2


def test_select_subset_includes_front_then_fills():
    pls = ParetoLocalSearch(subset_size=3)
    results = [
        _r(10, 10, 10),   # 0: non-dominated
        _r(50, 50, 50),   # 1: dominated by 0
        _r(60, 60, 60),   # 2: dominated
        _r(70, 70, 70),   # 3: dominated
    ]
    random.seed(0)
    subset = pls._select_subset(results)
    assert 0 in subset            # the front member is always included
    assert len(subset) == 3       # filled up to subset_size


# ------------------------------------------------------------
# End-to-end improve()
# ------------------------------------------------------------

def test_improve_whole_population_never_worsens_best_objectives():
    # subset = whole population, so no injection: every solution can only move
    # to a dominating point. Hence each objective's best cannot worsen.
    instance, due_dates, weights, machines = load_claude_extended_instance("ta01")
    random.seed(7)
    pop = [generate_chromosome(instance) for _ in range(12)]

    def best(population):
        rs = [evaluate_individual(instance, c, due_dates, weights, machines)
              for c in population]
        return (min(r["Cmax"] for r in rs),
                min(r["TWET"] for r in rs),
                min(r["NEC"] for r in rs))

    before = best(pop)

    pls = ParetoLocalSearch(subset_size=len(pop), max_moves=5)
    improved = pls.improve(instance, pop, due_dates, weights, machines)

    assert len(improved) == len(pop)
    after = best(improved)

    assert after[0] <= before[0]     # Cmax best not worse
    assert after[1] <= before[1]     # TWET best not worse
    assert after[2] <= before[2]     # NEC best not worse
    assert pls.evaluations > 0       # decodes were counted


def test_improve_preserves_size_and_feasibility_with_injection():
    instance, due_dates, weights, machines = load_claude_extended_instance("ta01")
    required = _required_multiset(instance)
    random.seed(3)
    pop = [generate_chromosome(instance) for _ in range(16)]

    pls = ParetoLocalSearch(subset_size=4, max_moves=4, inject=True)
    improved = pls.improve(instance, pop, due_dates, weights, machines)

    assert len(improved) == len(pop)
    for child in improved:
        assert sorted(child.operation_sequence) == required


# ------------------------------------------------------------
# Integration through run_UNSGAIII
# ------------------------------------------------------------

def test_unsga3_runs_with_local_search_hook():
    instance, due_dates, weights, machines = load_claude_extended_instance("ta01")
    pls = ParetoLocalSearch(subset_size=4, max_moves=3)

    random.seed(11)
    population = run_UNSGAIII(
        instance,
        generations=2,
        population_size=12,
        due_dates=due_dates,
        weights=weights,
        machines=machines,
        divisions=3,          # H = C(5,2) = 10 < N = 12
        local_search=pls,
    )

    assert len(population) == 12
    required = _required_multiset(instance)
    for chromo in population:
        assert sorted(chromo.operation_sequence) == required
    assert pls.evaluations > 0
