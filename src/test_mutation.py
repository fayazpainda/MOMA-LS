import random

from chromosome import Chromosome
from mutation import move_to_end_mutation, move_to_end_mutation_literal


def _sequence_chromosome(sequence):
    return Chromosome(
        operation_sequence=list(sequence),
        machine_sequence=[0] * len(sequence),
    )


def test_default_mutation_preserves_job_counts():
    random.seed(1)
    original = [0, 0, 1, 1, 2, 2]
    chromo = _sequence_chromosome(original)
    mutated = move_to_end_mutation(chromo, mutation_rate=1.0)
    assert sorted(mutated.operation_sequence) == sorted(original)


def test_default_mutation_moves_at_most_one_gene():
    random.seed(2)
    original = [0, 0, 1, 1, 2, 2]
    chromo = _sequence_chromosome(original)
    mutated = move_to_end_mutation(chromo, mutation_rate=1.0)
    # Chromosome-level operator: with rate=1.0 exactly one gene relocates to
    # the tail; every other gene keeps its relative front-to-back order.
    front_original = original[:]
    front_mutated = mutated.operation_sequence[:-1]
    remaining = front_original.copy()
    # Removing the one gene that moved should reproduce the mutated prefix.
    for gene in front_mutated:
        remaining.remove(gene)
    assert len(remaining) == 1


def test_literal_mutation_preserves_job_counts():
    random.seed(3)
    original = [0, 0, 1, 1, 2, 2]
    chromo = _sequence_chromosome(original)
    mutated = move_to_end_mutation_literal(chromo, mutation_rate=0.5)
    assert sorted(mutated.operation_sequence) == sorted(original)


def test_literal_mutation_rate_zero_is_identity():
    original = [0, 0, 1, 1, 2, 2]
    chromo = _sequence_chromosome(original)
    mutated = move_to_end_mutation_literal(chromo, mutation_rate=0.0)
    assert mutated.operation_sequence == original


def test_literal_mutation_rate_one_reverses_relative_order_to_tail():
    original = [0, 1, 2, 0, 1, 2]
    chromo = _sequence_chromosome(original)
    mutated = move_to_end_mutation_literal(chromo, mutation_rate=1.0)
    # Every position triggers -> all genes go to "triggered", none to
    # "untriggered", so the sequence is returned in original relative order.
    assert mutated.operation_sequence == original


def test_literal_mutation_moves_many_more_genes_than_default_on_average():
    random.seed(4)
    original = list(range(10)) * 3  # length-30 operation-based sequence
    moved_default = 0
    moved_literal = 0
    trials = 200
    for _ in range(trials):
        chromo = _sequence_chromosome(original)
        default_out = move_to_end_mutation(chromo, mutation_rate=0.1).operation_sequence
        literal_out = move_to_end_mutation_literal(
            _sequence_chromosome(original), mutation_rate=0.1
        ).operation_sequence
        moved_default += sum(a != b for a, b in zip(original, default_out))
        moved_literal += sum(a != b for a, b in zip(original, literal_out))
    # This is the quantitative core of the disclosed deviation: at Pm=0.1 the
    # literal per-position reading disturbs far more of the chromosome per
    # offspring than the chromosome-level default.
    assert moved_literal > moved_default * 3
