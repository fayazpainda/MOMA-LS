"""
Mutation-operator ablation: default (chromosome-level move-to-end) vs. the
literal per-position reading of Wei et al., Section IV-E.

WHY THIS SCRIPT EXISTS
-----------------------
`mutation.py`'s default operator (used by every algorithm in this repo) is a
disclosed deviation from Wei's literal per-position mutation text -- see that
module's docstring and REPRODUCTION_NOTES.md Correction #4. The deviation was
adopted on 2026-09-03 based on an informal observation ("inflated Cmax
11-16%"), not a controlled, reported experiment. This script runs BOTH
operators, same instances, same paired seeds, same everything else (U-NSGA-III
only, baseline reference_mode="uniform", no local search, no right-shift), and
reports the actual quantitative difference plus a paired Wilcoxon test so the
paper can cite a number instead of an anecdote.

Usage (from PROJECT ROOT):

    ./venv/bin/python src/mutation_ablation.py --smoke
    ./venv/bin/python src/mutation_ablation.py --instances ta11 ta31 --seeds 5
    ./venv/bin/python src/mutation_ablation.py            # full: 10 instances x 10 seeds

Writes results_mutation_ablation/mutation_ablation_runs.csv (one row per
instance x operator x seed; resumable) and prints a best/median/worst summary
plus a paired one-sided Wilcoxon signed-rank test per objective (H1: literal
is worse i.e. larger Cmax/TWET/NEC than default) at the end.
"""

import argparse
import csv
import random
import statistics
from pathlib import Path

from wei_extended_loader import load_wei_extended_instance
from unsga3 import run_UNSGAIII
from evaluation import evaluate_population
from nsga3_selection import fast_non_dominated_sort
from mutation import move_to_end_mutation, move_to_end_mutation_literal

try:
    from scipy.stats import wilcoxon
except ImportError:  # pragma: no cover
    wilcoxon = None


INSTANCES = ["ta11", "ta12", "ta21", "ta22", "ta31", "ta32", "ta41", "ta42", "ta61", "ta62"]
RUN_SEEDS = [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010]

GENERATIONS = 500
POPULATION_SIZE = 100

OPERATORS = {
    "default": move_to_end_mutation,
    "literal": move_to_end_mutation_literal,
}

RESULT_DIR = Path("results_mutation_ablation")
RUNS_CSV = RESULT_DIR / "mutation_ablation_runs.csv"


def unique_first_front(evaluated):
    fronts = fast_non_dominated_sort(evaluated)
    if not fronts:
        return []
    seen = set()
    unique = []
    for solution in fronts[0]:
        key = (float(solution["Cmax"]), float(solution["TWET"]), float(solution["NEC"]))
        if key in seen:
            continue
        seen.add(key)
        unique.append(solution)
    return unique


def run_one(instance, due_dates, weights, machines, mutation_fn, seed):
    random.seed(seed)
    population = run_UNSGAIII(
        instance,
        GENERATIONS,
        POPULATION_SIZE,
        due_dates,
        weights,
        machines,
        mutation_fn=mutation_fn,
        verbose=False,
    )
    evaluated = evaluate_population(instance, population, due_dates, weights, machines)
    front = unique_first_front(evaluated)
    if not front:
        raise RuntimeError("Empty Pareto front.")
    return {
        "Cmax": min(float(s["Cmax"]) for s in front),
        "TWET": min(float(s["TWET"]) for s in front),
        "NEC": min(float(s["NEC"]) for s in front),
    }


def save_run_row(row):
    RESULT_DIR.mkdir(exist_ok=True)
    exists = RUNS_CSV.exists()
    with open(RUNS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["instance", "operator", "seed", "best_Cmax", "best_TWET", "best_NEC"]
        )
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def load_completed():
    if not RUNS_CSV.exists():
        return set()
    done = set()
    with open(RUNS_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            done.add((row["instance"], row["operator"], int(row["seed"])))
    return done


def load_all_rows():
    if not RUNS_CSV.exists():
        return []
    with open(RUNS_CSV, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def analyze(instances):
    rows = load_all_rows()
    by_instance = {}
    for row in rows:
        if row["instance"] not in instances:
            continue
        by_instance.setdefault(row["instance"], {"default": {}, "literal": {}})
        by_instance[row["instance"]][row["operator"]][int(row["seed"])] = row

    print("\n" + "=" * 100)
    print("MUTATION ABLATION -- default (chromosome-level) vs literal per-position "
          "(Wei Section IV-E)")
    print("=" * 100)

    pooled_default = {"Cmax": [], "TWET": [], "NEC": []}
    pooled_literal = {"Cmax": [], "TWET": [], "NEC": []}

    for instance_name in instances:
        seeds_data = by_instance.get(instance_name)
        if not seeds_data:
            continue

        paired_seeds = sorted(set(seeds_data["default"]) & set(seeds_data["literal"]))
        if not paired_seeds:
            continue

        print(f"\n{instance_name}  (paired seeds: {len(paired_seeds)})")

        for obj in ("Cmax", "TWET", "NEC"):
            default_vals = [float(seeds_data["default"][s][f"best_{obj}"]) for s in paired_seeds]
            literal_vals = [float(seeds_data["literal"][s][f"best_{obj}"]) for s in paired_seeds]
            pooled_default[obj].extend(default_vals)
            pooled_literal[obj].extend(literal_vals)

            d_med = statistics.median(default_vals)
            l_med = statistics.median(literal_vals)
            pct_shift = 100.0 * (l_med - d_med) / d_med if d_med else float("nan")

            print(
                f"  {obj:5s} default b/m/w = "
                f"{min(default_vals):11.1f}/{d_med:11.1f}/{max(default_vals):11.1f}"
                f"   literal b/m/w = "
                f"{min(literal_vals):11.1f}/{l_med:11.1f}/{max(literal_vals):11.1f}"
                f"   median shift = {pct_shift:+.1f}%"
            )

    print("\n" + "-" * 100)
    print("POOLED paired one-sided Wilcoxon signed-rank (H1: literal > default, i.e. literal "
          "is worse)")
    print("-" * 100)

    for obj in ("Cmax", "TWET", "NEC"):
        d = pooled_default[obj]
        l = pooled_literal[obj]
        if len(d) < 6 or wilcoxon is None:
            print(f"  {obj:5s} n={len(d)} -- too few paired samples (or scipy unavailable) "
                  f"for a Wilcoxon test")
            continue
        stat, p = wilcoxon(l, d, alternative="greater")
        d_med, l_med = statistics.median(d), statistics.median(l)
        pct_shift = 100.0 * (l_med - d_med) / d_med if d_med else float("nan")
        print(f"  {obj:5s} n={len(d):3d}  median shift {pct_shift:+.1f}%  p={p:.3g}")


def main():
    global GENERATIONS  # noqa: PLW0603 (script-local override, smoke mode only)

    parser = argparse.ArgumentParser()
    parser.add_argument("--instances", nargs="+", default=INSTANCES)
    parser.add_argument("--seeds", type=int, default=len(RUN_SEEDS))
    parser.add_argument("--smoke", action="store_true",
                         help="tiny run: ta11 only, 2 seeds, 30 generations")
    parser.add_argument("--analyze-only", action="store_true",
                         help="skip running, just re-print the summary from existing CSV")
    args = parser.parse_args()

    instances = args.instances
    seeds = RUN_SEEDS[:args.seeds]

    if args.smoke:
        instances = ["ta11"]
        seeds = RUN_SEEDS[:2]
        GENERATIONS = 30

    if not args.analyze_only:
        completed = load_completed()

        for instance_name in instances:
            instance, due_dates, weights, machines = load_wei_extended_instance(instance_name)

            for seed in seeds:
                for operator_name, mutation_fn in OPERATORS.items():
                    key = (instance_name, operator_name, seed)
                    if key in completed:
                        continue

                    print(f"{instance_name} | {operator_name} | seed={seed}")

                    best = run_one(instance, due_dates, weights, machines, mutation_fn, seed)

                    save_run_row({
                        "instance": instance_name,
                        "operator": operator_name,
                        "seed": seed,
                        "best_Cmax": best["Cmax"],
                        "best_TWET": best["TWET"],
                        "best_NEC": best["NEC"],
                    })

    analyze(instances)


if __name__ == "__main__":
    main()
