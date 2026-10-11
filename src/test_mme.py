"""MME initialization returns a population of the requested size."""

from benchmark import load_taillard
from mme import initialize_MME_population


def test_initialize_mme_population():
    instance = load_taillard("data/taillard/test.txt")
    population = initialize_MME_population(instance, population_size=10)

    total_ops = sum(len(job.operations) for job in instance.jobs)
    assert len(population) == 10
    for chromosome in population:
        assert len(chromosome.operation_sequence) == total_ops
        assert len(chromosome.machine_sequence) == total_ops
