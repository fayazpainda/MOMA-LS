"""Energy (idle-targeted resequencing) local search."""

import random

from wei_extended_loader import load_wei_extended_instance
from chromosome import generate_chromosome
from decoder import decode
from ls_energy import (
    _swap_positions,
    machines_by_idle,
    energy_neighbors,
    DEFAULT_MAX_CANDIDATES,
)


def _required_multiset(instance):
    return sorted(job.job_id for job in instance.jobs for _ in job.operations)


def test_swap_positions_preserves_multiset():
    instance, *_ = load_wei_extended_instance("ta01")
    random.seed(0)
    chromo = generate_chromosome(instance)
    child = _swap_positions(chromo, 0, 5)
    assert sorted(child.operation_sequence) == sorted(chromo.operation_sequence)
    assert child.operation_sequence[0] == chromo.operation_sequence[5]
    assert child.operation_sequence[5] == chromo.operation_sequence[0]
    assert child.operation_sequence is not chromo.operation_sequence


def test_machines_by_idle_sorted_descending():
    instance, *_ = load_wei_extended_instance("ta01")
    random.seed(1)
    chromo = generate_chromosome(instance)
    schedule = decode(instance, chromo)
    ordered = machines_by_idle(instance, schedule)
    # Each listed machine has positive idle, and totals are non-increasing.
    from energy import calculate_machine_idle_periods
    totals = [sum(calculate_machine_idle_periods(schedule, mid)) for mid in ordered]
    assert all(t > 0 for t in totals)
    assert totals == sorted(totals, reverse=True)


def test_energy_neighbors_feasible_bounded_decodable():
    instance, due_dates, weights, machines = load_wei_extended_instance("ta01")
    required = _required_multiset(instance)
    random.seed(2)
    chromo = generate_chromosome(instance)

    neighbors = energy_neighbors(instance, chromo, due_dates, weights, machines)

    assert neighbors, "a random schedule should have idle to target"
    assert len(neighbors) <= DEFAULT_MAX_CANDIDATES
    for child in neighbors:
        assert sorted(child.operation_sequence) == required
        assert decode(instance, child).makespan > 0
