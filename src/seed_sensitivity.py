#!/usr/bin/env python3
"""
Wei2021 Reproduction -- leave-one-seed-out sensitivity of the reported HV gain

Motivation
----------
The HV normalisation bounds (ideal, nadir) used for Table 4 are the
componentwise min/max over EVERY objective vector observed on an instance,
across both configs (vanilla, memetic), all seeds, and all generations
(see ``memetic_comparison.compute_bounds``). Because the pooled set includes
the proposed method's own points, a single unusually good (or bad) seed can
shift the normalisation bounds and inflate/deflate the reported HV gain. This
was noticed ad hoc for ta61 in the writeup (Section V-C draft): one memetic
seed alone set the nadir/ideal used to score all 15 seeds, and excluding it
changed the reported improvement from +78.7% to +73.6%. This script
generalises that one-off check into something run systematically over every
instance, using the ``final_hv`` values already computed (once, correctly)
under the real per-generation-pooled bounds in ``memetic_runs.csv``.

Scope of this check (documented, not hidden)
---------------------------------------------
This is a MEAN-LEVEL jackknife: it asks "how much does the reported mean HV
improvement move if any single seed's already-computed final_hv is excluded
from the average?" It holds the normalisation bounds fixed (the true bounds,
already correctly computed once from the full per-generation pool) and only
resamples which seeds are averaged -- exactly what was done by hand for ta61.

It does NOT re-derive the bounds themselves with a seed's objective vectors
removed from the pool that defined (ideal, nadir) in the first place -- that
would require rerunning the GA (the per-generation vectors are not persisted
to disk, only final Pareto fronts are), since bounds are a function of every
generation's population, not just the final front. A prior attempt at
reconstructing bounds from final-front points only produced non-representative,
exploded deltas (a much narrower point pool inflates % differences) and was
discarded; it is not a valid proxy for the real bounds. This script is
therefore a necessary, not sufficient, check: it rules out "the headline
number is driven by one seed's own score," and reports whether ta61-style
mean-level sensitivity occurs anywhere else in the primary comparison. It does
not rule out bounds-level contamination -- that would require an instrumented
rerun (see the module-level note printed at the end of this script's output).

Usage: python src/seed_sensitivity.py --result-dir results_memetic_tuned_n15
"""

import argparse

import numpy as np
import pandas as pd


def analyse_instance(runs, instance):
    di = runs[runs["instance"] == instance]
    van = di[di["config"] == "vanilla"].set_index("seed")["final_hv"]
    mem = di[di["config"] == "memetic"].set_index("seed")["final_hv"]
    seeds = sorted(set(van.index) & set(mem.index))

    hv_v_full = van.loc[seeds].mean()
    hv_m_full = mem.loc[seeds].mean()
    delta_full = (hv_m_full - hv_v_full) / hv_v_full * 100

    swings = []
    for held_out in seeds:
        remaining = [s for s in seeds if s != held_out]
        hv_v = van.loc[remaining].mean()
        hv_m = mem.loc[remaining].mean()
        delta = (hv_m - hv_v) / hv_v * 100
        swings.append((held_out, delta, delta - delta_full))

    worst = max(swings, key=lambda t: abs(t[2]))
    return {
        "instance": instance,
        "n_seeds": len(seeds),
        "delta_pct_full": delta_full,
        "worst_held_out_seed": worst[0],
        "delta_pct_without_worst_seed": worst[1],
        "max_swing_pct_points": worst[2],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--result-dir", default="results_memetic_tuned_n15")
    args = ap.parse_args()

    runs = pd.read_csv(f"{args.result_dir}/memetic_runs.csv")

    rows = [analyse_instance(runs, inst)
            for inst in sorted(runs["instance"].unique())]
    df = pd.DataFrame(rows)

    out_csv = f"{args.result_dir}/seed_sensitivity.csv"
    df.to_csv(out_csv, index=False)

    with pd.option_context("display.float_format", "{:+.2f}".format):
        print(df.to_string(index=False))
    print(f"\nWritten: {out_csv}")
    print("\nNote: this jackknifes the SEED AVERAGE under the real, already-"
          "computed bounds; it does not re-derive the bounds themselves with "
          "a seed excluded (that needs an instrumented rerun -- the raw "
          "per-generation vectors used for bounds are not persisted).")


if __name__ == "__main__":
    main()
