"""Decoder turns a chromosome into a feasible schedule."""

from benchmark import load_taillard
from chromosome import generate_chromosome
from decoder import decode


def test_decode_schedule():
    instance = load_taillard("data/taillard/test.txt")
    chromosome = generate_chromosome(instance)
    schedule = decode(instance, chromosome)

    total_ops = sum(len(job.operations) for job in instance.jobs)

    assert schedule.makespan > 0
    assert len(schedule.operations) == total_ops
    for op in schedule.operations:
        assert 0 <= op.machine < instance.num_machines
        assert op.start <= op.finish
        assert op.finish <= schedule.makespan
