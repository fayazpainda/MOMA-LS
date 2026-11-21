"""Tardiness (CET) local search: sink ranking, feasibility, TWET focus."""

import random

import pytest

from wei_extended_loader import load_wei_extended_instance
from chromosome import generate_chromosome
from decoder import decode, ScheduledOperation, Schedule
from ls_tardiness import tardy_job_sinks, tardiness_neighbors


def _required_multiset(instance):
    return sorted(job.job_id for job in instance.jobs for _ in job.operations)


# ------------------------------------------------------------
# Sink selection (synthetic schedule)
# ------------------------------------------------------------

def test_tardy_sinks_ranked_by_weighted_tardiness():
    # job0 last finish 10, due 5, alpha 2 -> weighted 10
    # job1 last finish 20, due 5, alpha 1 -> weighted 15  (ranked first)
    # job2 last finish 8,  due 100        -> not tardy
    ops = [
        ScheduledOperation(0, 0, 0, 0.0, 10.0),
        ScheduledOperation(1, 0, 1, 0.0, 20.0),
        ScheduledOperation(2, 0, 2, 0.0, 8.0),
    ]
    schedule = Schedule(operations=ops, makespan=20.0)
    due_dates = {0: 5, 1: 5, 2: 100}
    weights = {0: {"alpha": 2, "beta": 1},
               1: {"alpha": 1, "beta": 1},
               2: {"alpha": 1, "beta": 1}}

    sinks = tardy_job_sinks(None, schedule, due_dates, weights)
    assert [s.job_id for s in sinks] == [1, 0]        # weighted order, job2 out

    top1 = tardy_job_sinks(None, schedule, due_dates, weights, top_k=1)
    assert [s.job_id for s in top1] == [1]


def test_no_tardy_jobs_gives_no_sinks():
    ops = [ScheduledOperation(0, 0, 0, 0.0, 10.0)]
    schedule = Schedule(operations=ops, makespan=10.0)
    sinks = tardy_job_sinks(None, schedule, {0: 1000}, {0: {"alpha": 1, "beta": 1}})
    assert sinks == []


# ------------------------------------------------------------
# Neighborhood
# ------------------------------------------------------------

def test_tardiness_neighbors_require_context():
    instance, *_ = load_wei_extended_instance("ta01")
    random.seed(0)
    chromo = generate_chromosome(instance)
    with pytest.raises(ValueError):
        tardiness_neighbors(instance, chromo)  # no due_dates / weights


def test_tardiness_neighbors_feasible_and_decodable():
    instance, due_dates, weights, machines = load_wei_extended_instance("ta01")
    required = _required_multiset(instance)

    neighbors = []
    for seed in range(30):
        random.seed(seed)
        chromo = generate_chromosome(instance)
        neighbors = tardiness_neighbors(instance, chromo, due_dates, weights, machines)
        if neighbors:
            break

    assert neighbors, "expected tardy-job CET moves within 30 seeds"
    for child in neighbors:
        assert sorted(child.operation_sequence) == required
        assert decode(instance, child).makespan > 0
