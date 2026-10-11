"""Taillard loader: instance shape is consistent."""

from benchmark import load_taillard


def test_load_taillard_shape():
    instance = load_taillard("data/taillard/test.txt")

    assert instance.num_jobs > 0
    assert instance.num_machines > 0
    assert len(instance.jobs) == instance.num_jobs

    for job in instance.jobs:
        assert len(job.operations) == instance.num_machines
        for op in job.operations:
            assert 0 <= op.machine < instance.num_machines
            assert op.processing_time >= 0
