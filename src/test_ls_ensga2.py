"""ENSGA-II baseline neighborhoods (N1 swap, N2 remove-insert): feasibility
and move correctness."""

import random

from benchmark import load_taillard
from chromosome import generate_chromosome
from decoder import decode
from ls_ensga2 import (
    _swap_positions,
    _insert_position,
    ensga2_n1_neighbors,
    ensga2_n2_neighbors,
)


# ------------------------------------------------------------
# N1: position-exchange
# ------------------------------------------------------------

def test_swap_positions_exchanges_values():
    seq = [0, 1, 2, 3]
    result = _swap_positions(seq, 0, 3, random)
    assert result == [3, 1, 2, 0]
    assert seq == [0, 1, 2, 3]  # original untouched


def test_n1_neighbors_feasible_and_decodable():
    instance = load_taillard("data/taillard/test.txt")
    random.seed(0)
    chromo = generate_chromosome(instance)
    original = sorted(chromo.operation_sequence)

    neighbors = ensga2_n1_neighbors(instance, chromo, num_samples=5, rng=random)

    assert len(neighbors) == 5
    for child in neighbors:
        assert sorted(child.operation_sequence) == original
        assert child is not chromo
        schedule = decode(instance, child)
        assert schedule.makespan > 0


def test_n1_neighbors_empty_for_degenerate_sequence():
    instance = load_taillard("data/taillard/test.txt")
    chromo = generate_chromosome(instance)
    chromo.operation_sequence = [0]
    assert ensga2_n1_neighbors(instance, chromo) == []


# ------------------------------------------------------------
# N2: remove-insert
# ------------------------------------------------------------

def test_insert_position_relocates_gene_forward():
    # Move the gene at index 0 to directly behind index 2.
    seq = ["a", "b", "c", "d"]
    result = _insert_position(seq, 0, 2)
    assert result == ["b", "c", "a", "d"]


def test_insert_position_relocates_gene_backward():
    # Move the gene at index 3 to directly behind index 0.
    seq = ["a", "b", "c", "d"]
    result = _insert_position(seq, 3, 0)
    assert result == ["a", "d", "b", "c"]


def test_n2_neighbors_feasible_and_decodable():
    instance = load_taillard("data/taillard/test.txt")
    random.seed(1)
    chromo = generate_chromosome(instance)
    original = sorted(chromo.operation_sequence)

    neighbors = ensga2_n2_neighbors(instance, chromo, num_samples=5, rng=random)

    assert len(neighbors) == 5
    for child in neighbors:
        assert sorted(child.operation_sequence) == original
        schedule = decode(instance, child)
        assert schedule.makespan > 0


def test_n2_neighbors_empty_for_degenerate_sequence():
    instance = load_taillard("data/taillard/test.txt")
    chromo = generate_chromosome(instance)
    chromo.operation_sequence = [0]
    assert ensga2_n2_neighbors(instance, chromo) == []
