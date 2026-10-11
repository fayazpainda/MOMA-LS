"""OS crossover preserves the operation multiset (produces a valid child)."""

from benchmark import load_taillard
from chromosome import generate_chromosome
from genetic import crossover_OS


def test_crossover_os_preserves_operations():
    instance = load_taillard("data/taillard/test.txt")
    p1 = generate_chromosome(instance)
    p2 = generate_chromosome(instance)

    child1, child2 = crossover_OS(p1, p2)

    # Each child is a permutation of the same repeated job ids as the parents.
    for child in (child1, child2):
        assert len(child.operation_sequence) == len(p1.operation_sequence)
        assert sorted(child.operation_sequence) == sorted(p1.operation_sequence)
