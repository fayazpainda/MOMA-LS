"""NEH heuristic yields a complete chromosome."""

from benchmark import load_taillard
from neh import create_NEH_chromosome


def test_create_neh_chromosome():
    instance = load_taillard("data/taillard/test.txt")
    chromosome = create_NEH_chromosome(instance)

    total_ops = sum(len(job.operations) for job in instance.jobs)
    assert len(chromosome.operation_sequence) == total_ops
    assert len(chromosome.machine_sequence) == total_ops
