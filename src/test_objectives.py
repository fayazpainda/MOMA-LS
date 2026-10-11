"""Objective-function test on the Wei-style extended TA11 dataset.

Verifies that the corrected per-job TWET formula is the default and is a
genuinely different objective from the (inconsistent) makespan reading.
"""

from chromosome import generate_chromosome
from decoder import decode
from objectives import calculate_makespan, calculate_twet
from claude_dataset import load_claude_extended_instance


def test_twet_per_job_is_default_and_distinct():
    instance, due_dates, weights, _machines = load_claude_extended_instance("ta11")

    schedule = decode(instance, generate_chromosome(instance))

    cmax = calculate_makespan(schedule)
    twet = calculate_twet(schedule, due_dates, weights)
    twet_per_job = calculate_twet(
        schedule, due_dates, weights, completion_mode="per_job"
    )
    twet_makespan = calculate_twet(
        schedule, due_dates, weights, completion_mode="makespan"
    )

    # 1. The default must be the corrected per_job formula.
    assert twet == twet_per_job, (
        "Default calculate_twet() no longer matches per_job mode; "
        "the corrected formula is not the default."
    )
    # 2. per_job must be a genuinely different objective from the makespan
    #    reading, otherwise the per-job fix is not in effect.
    assert twet_per_job != twet_makespan, (
        "per_job and makespan TWET are identical; the per-job "
        "completion-time fix is not taking effect."
    )
    # 3. Basic sanity.
    assert twet >= 0.0
    assert cmax > 0.0
