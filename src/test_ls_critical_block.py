"""N5 critical-block local search: move enumeration and feasibility."""

import random

from benchmark import load_taillard
from chromosome import generate_chromosome
from decoder import decode, ScheduledOperation
from ls_critical_block import (
    operation_positions,
    swap_operations,
    n5_move_pairs,
    critical_block_neighbors,
)


def _op(job_id, operation_id, machine):
    # Times are irrelevant to n5_move_pairs; only block structure matters.
    return ScheduledOperation(
        job_id=job_id, operation_id=operation_id, machine=machine,
        start=0.0, finish=0.0,
    )


# ------------------------------------------------------------
# N5 border rule
# ------------------------------------------------------------

def test_n5_rule_first_internal_last_blocks():
    # r = 3: first [o1,o2], internal [o3,o4,o5], last [o6,o7].
    o1, o2 = _op(0, 0, 0), _op(1, 0, 0)
    o3, o4, o5 = _op(2, 0, 1), _op(3, 0, 1), _op(4, 0, 1)
    o6, o7 = _op(5, 0, 2), _op(6, 0, 2)
    blocks = [[o1, o2], [o3, o4, o5], [o6, o7]]
    # first -> last two; internal -> first two AND last two; last -> first two.
    assert n5_move_pairs(blocks) == [(o1, o2), (o3, o4), (o4, o5), (o6, o7)]


def test_internal_block_of_size_two_is_not_duplicated():
    o1, o2 = _op(0, 0, 0), _op(1, 0, 0)
    o3, o4 = _op(2, 0, 1), _op(3, 0, 1)   # internal, size 2
    o5, o6 = _op(4, 0, 2), _op(5, 0, 2)
    blocks = [[o1, o2], [o3, o4], [o5, o6]]
    # internal first-two and last-two coincide -> a single move.
    assert n5_move_pairs(blocks) == [(o1, o2), (o3, o4), (o5, o6)]


def test_single_block_is_n5_optimal():
    o1, o2, o3 = _op(0, 0, 0), _op(1, 0, 0), _op(2, 0, 0)
    assert n5_move_pairs([[o1, o2, o3]]) == []
    assert n5_move_pairs([]) == []


# ------------------------------------------------------------
# Swap realization
# ------------------------------------------------------------

def test_operation_positions_track_occurrences():
    instance = load_taillard("data/taillard/test.txt")
    random.seed(0)
    chromo = generate_chromosome(instance)
    positions = operation_positions(chromo)
    for job_id, idxs in positions.items():
        # k-th occurrence of the job sits at positions[job_id][k].
        occurrences = [i for i, j in enumerate(chromo.operation_sequence) if j == job_id]
        assert idxs == occurrences


def test_swap_preserves_job_multiset():
    instance = load_taillard("data/taillard/test.txt")
    random.seed(1)
    chromo = generate_chromosome(instance)
    schedule = decode(instance, chromo)
    # Swap the first two operations on some machine via their scheduled ops.
    op_a, op_b = schedule.operations[0], schedule.operations[1]
    child = swap_operations(chromo, op_a, op_b)
    assert sorted(child.operation_sequence) == sorted(chromo.operation_sequence)
    assert child is not chromo
    assert child.operation_sequence is not chromo.operation_sequence


# ------------------------------------------------------------
# End-to-end neighborhood
# ------------------------------------------------------------

def test_neighbors_are_feasible_and_decodable():
    instance = load_taillard("data/taillard/test.txt")
    original = sorted(
        j for job in instance.jobs for j in [job.job_id] * len(job.operations)
    )

    # Find a seed that yields a non-empty N5 neighborhood (r >= 2 with a
    # size-2+ block); 3x3 instances reach this within a few tries.
    neighbors = []
    for seed in range(50):
        random.seed(seed)
        chromo = generate_chromosome(instance)
        neighbors = critical_block_neighbors(instance, chromo)
        if neighbors:
            break

    assert neighbors, "expected at least one N5 neighbor within 50 seeds"

    for child in neighbors:
        # Feasible: same job multiset as the instance requires.
        assert sorted(child.operation_sequence) == original
        # Decodes into a complete schedule.
        schedule = decode(instance, child)
        assert schedule.makespan > 0
