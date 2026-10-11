"""Energy-aware right-shift decoder (MEMETIC_DESIGN.md §3b/§4.3a).

The right-shift refinement must be a *pure NEC lever*: it may only redistribute
each machine's idle time, never change the schedule's makespan, any job's
completion time, or the operation sequence. These tests pin those invariants on
real instances across several random schedules, plus feasibility of the shifted
schedule.
"""

import random

from claude_dataset import load_claude_extended_instance
from chromosome import generate_chromosome
from decoder import decode, decode_right_shift
from objectives import job_completion_times, calculate_makespan
from energy import calculate_NEC, get_machine_operations
from evaluation import evaluate_individual


EPS = 1e-6


def _feasible(instance, schedule):
    """No machine overlap; job precedence respected; non-negative starts."""

    # Per-machine: operations do not overlap in time.
    for machine_id in range(instance.num_machines):
        ops = get_machine_operations(schedule, machine_id)
        for a, b in zip(ops, ops[1:]):
            assert a.finish <= b.start + EPS, "machine overlap after right-shift"

    # Per-job: operation k+1 starts no earlier than operation k finishes.
    by_job = {}
    for op in schedule.operations:
        by_job.setdefault(op.job_id, []).append(op)
    for ops in by_job.values():
        ops.sort(key=lambda o: o.operation_id)
        for a, b in zip(ops, ops[1:]):
            assert a.finish <= b.start + EPS, "job precedence violated"

    for op in schedule.operations:
        assert op.start >= -EPS
        assert op.finish >= op.start - EPS


def test_right_shift_preserves_makespan_and_completions_and_is_feasible():
    """The raw transform must keep Cmax and every job completion exactly, and
    produce a feasible schedule. (NEC safety is asserted separately, at the
    evaluation level, because the raw transform can fragment idle — see
    decode_right_shift's docstring.)"""

    instance, *_ = load_claude_extended_instance("ta01")

    for seed in range(20):
        random.seed(seed)
        chromo = generate_chromosome(instance)

        left = decode(instance, chromo)
        shifted = decode_right_shift(instance, chromo)

        # (1) Makespan is identical.
        assert abs(calculate_makespan(left) - calculate_makespan(shifted)) < EPS

        # (2) Every job's completion time is identical -> Cmax and TWET unchanged.
        c_left = job_completion_times(left)
        c_shift = job_completion_times(shifted)
        assert c_left.keys() == c_shift.keys()
        for job_id in c_left:
            assert abs(c_left[job_id] - c_shift[job_id]) < EPS, (
                f"job {job_id} completion moved: "
                f"{c_left[job_id]} -> {c_shift[job_id]}"
            )

        # (3) The shifted schedule is feasible.
        _feasible(instance, shifted)


def test_evaluate_right_shift_is_pareto_safe():
    """evaluate_individual(right_shift=True) is the Pareto-safe NEC lever: Cmax
    and TWET are identical to the semi-active evaluation, and NEC never exceeds
    it (a pure decrease or no-op)."""

    instance, due_dates, weights, machines = load_claude_extended_instance("ta01")

    improved_at_least_once = False

    for seed in range(20):
        random.seed(seed)
        chromo = generate_chromosome(instance)

        base = evaluate_individual(instance, chromo, due_dates, weights, machines)
        rs = evaluate_individual(
            instance, chromo, due_dates, weights, machines, right_shift=True
        )

        # Cmax and TWET are untouched by right-shift.
        assert abs(base["Cmax"] - rs["Cmax"]) < EPS
        assert abs(base["TWET"] - rs["TWET"]) < EPS

        # NEC is never worsened (Pareto-safe by the min() safeguard).
        assert rs["NEC"] <= base["NEC"] + EPS, (
            f"seed {seed}: right-shift raised NEC "
            f"{base['NEC']} -> {rs['NEC']}"
        )

        if rs["NEC"] < base["NEC"] - EPS:
            improved_at_least_once = True

    # The lever must actually fire on at least one random schedule, otherwise it
    # is inert and not worth an experiment arm.
    assert improved_at_least_once, "right-shift never lowered NEC on any seed"


def test_right_shift_only_moves_operations_later():
    """A right shift never starts an operation earlier than the semi-active
    decode did (it is a pure delay within slack)."""

    instance, *_ = load_claude_extended_instance("ta01")

    for seed in range(10):
        random.seed(100 + seed)
        chromo = generate_chromosome(instance)

        left = {
            (op.job_id, op.operation_id): op.start
            for op in decode(instance, chromo).operations
        }
        for op in decode_right_shift(instance, chromo).operations:
            assert op.start >= left[(op.job_id, op.operation_id)] - EPS


def test_right_shift_does_not_touch_the_sequence():
    """decode_right_shift changes start times only; the operation sequence (the
    decision variable) is untouched, so critical-path structure is unaffected."""

    instance, *_ = load_claude_extended_instance("ta01")
    random.seed(7)
    chromo = generate_chromosome(instance)

    before = list(chromo.operation_sequence)
    decode_right_shift(instance, chromo)
    assert chromo.operation_sequence == before
