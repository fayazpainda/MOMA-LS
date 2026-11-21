"""
Wei2021 Reproduction

Ramp-time fix test (NEC).

The NEC overshoot on large instances is caused by Eq. 20 being
read PER-MACHINE-LOAD, so ramp_off_time = machine_load / m grows
with job count and locks large instances out of STANDBY/OFF.

Wei's NEC is nearly flat vs instance size, which is only possible
if the ramp times are SIZE-INDEPENDENT. The natural size-
independent reading of Eq. 20 divides total work by the operation
COUNT as well, i.e. it is based on the MEAN operation processing
time p_bar (~50 for every Taillard instance), not per-machine
total load.

This script takes a real optimized schedule for each instance and
recomputes NEC under several ramp-time settings, comparing each to
Wei's published best NEC. It re-optimizes to get a schedule but
does NOT touch the dataset or the full experiment -- it only shows
which ramp-time reading brings NEC into Wei's range.

Run from PROJECT ROOT:

    PYTHONPATH=src ./venv/bin/python src/test_ramp_fix.py
"""

import sys
import random
import statistics
from dataclasses import replace

from wei_extended_loader import load_wei_extended_instance
from unsga3 import run_UNSGAIII
from evaluation import evaluate_population
from decoder import decode
from energy import calculate_NEC


GENERATIONS = 150
POPULATION_SIZE = 100
SEED = 1001

# (instance, Wei best NEC)
TARGETS = [
    ("ta11", 3763.92),
    ("ta31", 3883.98),
    ("ta61", 7355.44),
]


def mean_operation_time(instance):
    times = [
        op.processing_time
        for job in instance.jobs
        for op in job.operations
    ]
    return statistics.mean(times)


def with_ramp_times(machines, t_off, t_standby):
    """Return copies of the machine energy models with the given
    (size-independent) ramp times applied to every machine."""
    return [
        replace(
            m,
            ramp_off_time=t_off,
            ramp_standby_time=t_standby,
        )
        for m in machines
    ]


def best_nec_schedule(instance, due, wts, machines):
    random.seed(SEED)
    population = run_UNSGAIII(
        instance, GENERATIONS, POPULATION_SIZE, due, wts, machines
    )
    evaluated = evaluate_population(
        instance, population, due, wts, machines
    )
    best = min(evaluated, key=lambda x: float(x["NEC"]))
    return decode(instance, best["chromosome"])


def main():
    names = sys.argv[1:] or [t[0] for t in TARGETS]
    wei = dict(TARGETS)

    rows = []

    for name in names:
        instance, due, wts, machines = load_wei_extended_instance(name)

        schedule = best_nec_schedule(instance, due, wts, machines)

        p_bar = mean_operation_time(instance)

        # Ramp-time settings to compare.
        cur_off = statistics.mean(m.ramp_off_time for m in machines)
        cur_sb = statistics.mean(m.ramp_standby_time for m in machines)

        settings = {
            f"current (per-load, off~{cur_off:.0f})":
                machines,
            f"mean_op  (off={p_bar:.0f}, sb={p_bar/2:.0f})":
                with_ramp_times(machines, p_bar, p_bar / 2),
            f"half_mean(off={p_bar/2:.0f}, sb={p_bar/4:.0f})":
                with_ramp_times(machines, p_bar / 2, p_bar / 4),
        }

        print("=" * 78)
        print(f"{name}  (Wei best NEC = {wei.get(name, float('nan')):.1f}, "
              f"mean op time = {p_bar:.1f})")
        print("-" * 78)

        for label, mm in settings.items():
            nec = calculate_NEC(schedule, mm)
            w = wei.get(name)
            ratio = nec / w if w else float("nan")
            print(f"  {label:<34} NEC = {nec:9.1f}   x{ratio:.2f} vs Wei")
            rows.append((name, label, nec, ratio))
        print()

    print("=" * 78)
    print("Look for the ramp-time setting whose NEC ratio is ~1.0 across")
    print("ALL instance sizes. That reading is the correct interpretation")
    print("of Eq. 20 for reproducing Wei's NEC. NOTE: same schedule is")
    print("reused per instance, so only the ramp times differ.")
    print("=" * 78)


if __name__ == "__main__":
    main()
