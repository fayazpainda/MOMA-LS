"""Random chromosome generation produces a valid encoding."""

from benchmark import load_taillard
from chromosome import generate_chromosome


def test_generate_chromosome():
    instance = load_taillard("data/taillard/test.txt")
    chromosome = generate_chromosome(instance)

    total_ops = sum(len(job.operations) for job in instance.jobs)

    assert len(chromosome.operation_sequence) == total_ops
    assert len(chromosome.machine_sequence) == total_ops
    # Operation sequence must reference only valid job ids.
    assert set(chromosome.operation_sequence) <= {job.job_id for job in instance.jobs}
