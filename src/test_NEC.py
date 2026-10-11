"""Non-processing energy consumption (NEC) is well-defined and non-negative."""

from benchmark import load_taillard
from chromosome import generate_chromosome
from decoder import decode
from energy import MachineEnergy, calculate_NEC


def _machine():
    return MachineEnergy(
        processing_power=0.96,
        idle_power=0.68,
        standby_power=0.40,
        ramp_standby_power=0.8,
        ramp_standby_time=24.0,
        ramp_off_power=0.8,
        ramp_off_time=48.0,
    )


def test_calculate_nec():
    instance = load_taillard("data/taillard/test.txt")
    schedule = decode(instance, generate_chromosome(instance))
    machines = [_machine() for _ in range(instance.num_machines)]

    nec = calculate_NEC(schedule, machines)

    assert nec >= 0
