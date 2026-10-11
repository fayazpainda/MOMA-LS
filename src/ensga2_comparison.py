"""
Wei2021 Reproduction -- ENSGA-II baseline comparison

Runs the ENSGA-II baseline (Luan et al. 2023: NSGA-II host + sparsity-driven
local search, `sparsity_local_search.SparsityLocalSearch` with the N1/N2
neighborhoods of `ls_ensga2.py`) across the main 10-instance study, at the
same equal-decode-budget protocol as `algo_comparison.py` / `memetic_comparison.py`.

This is a BMW-table-only literature baseline (see
`project_overleaf2_new_baselines` memory): ENSGA-II is NOT part of the
5-host robustness study (which only ever pairs a host with MOMA-LS's OWN
critical-block local search, to test whether OUR local search generalizes
across hosts). ENSGA-II instead stands on its own as "NSGA-II + the
published ENSGA-II local search," reported once, for the all-baselines BMW
table alongside U-NSGA-III (vanilla), SPEA2 (vanilla), SMS-EMOA (vanilla)
and MOMA-LS (proposed).

Equal-decode budget
--------------------
Identical principle to `algo_comparison.py`: the shared budget B for each
(instance, seed) is read from an existing `algo_runs.csv`/converted anchor
containing `unsga3_memetic` decode totals (see `--anchor-csv`), so this
script never needs to re-run U-NSGA-III. ENSGA-II always runs at
`MEMETIC_GENERATIONS` (its own decode total differs from unsga3_memetic's
since the sparsity-LS has a different cost profile, but that total is what
gets reported -- no budget truncation is applied here, matching how every
other memetic-mode method in `algo_comparison.py` is already reported).

Outputs (RESULT_DIR = results_ensga2/)
    ensga2_config.json       run configuration
    ensga2_bounds.json       per-instance ideal/nadir/ref_factor (OWN method only;
                              NOT merged with other methods' bounds -- combine
                              post-hoc with the other CSVs' own vectors if a
                              shared-bounds HV comparison is needed across all
                              BMW-table methods at once)
    ensga2_runs.csv          one row per (instance, seed): algo_runs.csv-compatible
                              schema (method="ensga2"), so it concatenates
                              directly with algo_runs.csv/memetic_runs.csv-derived
                              rows for the BMW table
    ensga2_final_pareto.csv  final front objective vectors per run
    done/<instance>.done     instance-level resume marker
"""

import argparse
import csv
import json
import random
import time
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:  # graceful fallback if tqdm is unavailable
    class _NullBar:
        def update(self, *_): pass
        def set_postfix(self, *_, **__): pass
        def close(self): pass

    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else _NullBar()

    tqdm.write = print

from claude_dataset import load_claude_extended_instance
from nsga2 import run_NSGAII
from evaluation import evaluate_population
from hypervolume import normalized_hypervolume
from sparsity_local_search import SparsityLocalSearch
from ls_ensga2 import ensga2_n1_neighbors, ensga2_n2_neighbors

from memetic_comparison import (
    unique_first_front, front_objectives, best_values,
    loop_decodes, compute_bounds, _append_rows,
    POPULATION_SIZE, MEMETIC_GENERATIONS, INSTANCES, RUN_SEEDS, REF_FACTOR,
)


METHOD_NAME = "ensga2"

# ENSGA-II's own tuned judgment threshold (Luan et al. 2023, Table 4).
JUDGMENT_THRESHOLD = 0.2

RUNS_FIELDS = [
    "instance", "method", "algo", "mode", "run", "seed", "generations",
    "decodes", "ls_evals", "best_Cmax", "best_TWET", "best_NEC",
    "pareto_size", "final_hv", "runtime_seconds",
]

PARETO_FIELDS = ["instance", "method", "run", "seed", "Cmax", "TWET", "NEC"]


# ============================================================
# Anchor budget (reused from an existing unsga3_memetic run)
# ============================================================

def _load_anchor_lookup(anchor_csv_path):
    """Read {(instance, seed): decodes} for unsga3_memetic rows from an
    existing algo_runs.csv-compatible CSV (see `make_anchor_csv.py` to build
    one from `memetic_runs.csv`)."""

    lookup = {}
    with open(anchor_csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if row["method"] == "unsga3_memetic":
                lookup[(row["instance"], int(row["seed"]))] = int(row["decodes"])

    if not lookup:
        raise ValueError(
            f"No unsga3_memetic rows found in {anchor_csv_path!r}."
        )

    return lookup


# ============================================================
# ONE RUN
# ============================================================

def run_one_ensga2(instance_bundle, seed, generations, subset_size, max_moves,
                    judgment_threshold, bar=None):

    instance, due_dates, weights, machines = instance_bundle

    local_search = SparsityLocalSearch(
        judgment_threshold=judgment_threshold,
        neighborhoods=[ensga2_n1_neighbors, ensga2_n2_neighbors],
        subset_size=subset_size,
        max_moves=max_moves,
        inject=True,
    )

    per_gen = []

    def generation_callback(generation, parent_results):
        decodes = loop_decodes(generation) + local_search.evaluations
        objs = front_objectives(unique_first_front(parent_results))
        per_gen.append((generation, decodes, objs))
        if bar is not None:
            post = {"seed": seed, "gen": f"{generation + 1}/{generations}"}
            if objs:
                post["Cmax"] = int(min(o[0] for o in objs))
                post["TWET"] = int(min(o[1] for o in objs))
                post["NEC"] = int(min(o[2] for o in objs))
            bar.set_postfix(post, refresh=True)

    random.seed(seed)
    start = time.time()
    final_population = run_NSGAII(
        instance, generations, POPULATION_SIZE,
        due_dates, weights, machines,
        generation_callback=generation_callback,
        local_search=local_search,
        right_shift=False,
        verbose=False,
    )
    runtime = time.time() - start

    evaluated = evaluate_population(
        instance, final_population, due_dates, weights, machines)
    final_front = unique_first_front(evaluated)
    final_objs = front_objectives(final_front)
    if not final_objs:
        raise RuntimeError(f"Empty final front for ensga2, seed={seed}")

    total_decodes = loop_decodes(generations) + local_search.evaluations
    per_gen.append((generations, total_decodes, final_objs))

    return {
        "algo": METHOD_NAME, "mode": "memetic", "method": METHOD_NAME,
        "seed": seed, "per_gen": per_gen, "final_objs": final_objs,
        "runtime": runtime, "total_decodes": total_decodes,
        "ls_evals": local_search.evaluations, "generations": generations,
    }


# ============================================================
# INSTANCE DRIVER
# ============================================================

def process_instance(instance_name, seeds, generations, subset_size,
                      max_moves, judgment_threshold, anchor_lookup, result_dir):

    runs_csv = result_dir / "ensga2_runs.csv"
    pareto_csv = result_dir / "ensga2_final_pareto.csv"

    bundle = load_claude_extended_instance(instance_name)

    all_vectors = []
    records = []

    bar = tqdm(total=len(seeds), desc=instance_name, unit="run", leave=False)

    for run_number, seed in enumerate(seeds, start=1):

        B = anchor_lookup[(instance_name, seed)]

        rec = run_one_ensga2(
            bundle, seed, generations, subset_size, max_moves,
            judgment_threshold, bar=bar,
        )
        bar.update(1)
        tqdm.write(
            f"  {instance_name} seed={seed}: anchor B={B} decodes "
            f"(ensga2 actual={rec['total_decodes']})"
        )

        for _g, _d, objs in rec["per_gen"]:
            all_vectors.extend(objs)
        records.append((run_number, rec))

    bar.close()

    ideal, nadir = compute_bounds(all_vectors)
    bounds_path = result_dir / "ensga2_bounds.json"
    data = json.loads(bounds_path.read_text()) if bounds_path.exists() else {}
    data[instance_name] = {"ideal": ideal, "nadir": nadir, "ref_factor": REF_FACTOR}
    bounds_path.write_text(json.dumps(data, indent=2))

    def hv(objs):
        return normalized_hypervolume(objs, ideal, nadir, ref_factor=REF_FACTOR)

    runs_rows, pareto_rows = [], []
    for run_number, rec in records:
        best = best_values(rec["final_objs"])
        runs_rows.append({
            "instance": instance_name, "method": rec["method"],
            "algo": rec["algo"], "mode": rec["mode"],
            "run": run_number, "seed": rec["seed"],
            "generations": rec["generations"], "decodes": rec["total_decodes"],
            "ls_evals": rec["ls_evals"], **best,
            "pareto_size": len(rec["final_objs"]),
            "final_hv": hv(rec["final_objs"]),
            "runtime_seconds": rec["runtime"],
        })
        for c, t, n in rec["final_objs"]:
            pareto_rows.append({
                "instance": instance_name, "method": rec["method"],
                "run": run_number, "seed": rec["seed"],
                "Cmax": c, "TWET": t, "NEC": n,
            })

    _append_rows(runs_csv, RUNS_FIELDS, runs_rows)
    _append_rows(pareto_csv, PARETO_FIELDS, pareto_rows)

    hvs = [r["final_hv"] for r in runs_rows]
    print(f"  {instance_name}: ensga2 mean HV = {sum(hvs)/len(hvs):.4f} "
          f"(n={len(hvs)})")
    print(f"  wrote {len(runs_rows)} run rows for {instance_name}")


# ============================================================
# MAIN
# ============================================================

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--instances", nargs="*", default=None)
    p.add_argument("--seeds", type=int, default=None,
                   help="Use only the first N seeds (default: all).")
    p.add_argument("--memetic-generations", type=int, default=MEMETIC_GENERATIONS)
    p.add_argument("--result-dir", default="results_ensga2")
    p.add_argument("--anchor-csv", required=True,
                   help="Path to an algo_runs.csv-compatible CSV containing "
                        "unsga3_memetic rows for the requested instances/seeds "
                        "(use make_anchor_csv.py to build one from an existing "
                        "memetic_runs.csv).")
    p.add_argument("--subset-size", type=int, default=10)
    p.add_argument("--max-moves", type=int, default=50)
    p.add_argument("--judgment-threshold", type=float, default=JUDGMENT_THRESHOLD,
                   help="Sparsity judgment threshold k, 0<k<1 "
                        f"(default {JUDGMENT_THRESHOLD}, Luan et al. 2023's own "
                        "tuned value).")
    p.add_argument("--smoke", action="store_true",
                   help="ta01, 1 seed, 8 gens -- fast end-to-end check.")
    args = p.parse_args()

    if args.smoke:
        instances = ["ta11"]
        seeds = RUN_SEEDS[:1]
        generations = 8
        result_dir = Path("results_ensga2_smoke")
    else:
        instances = args.instances or INSTANCES
        seeds = RUN_SEEDS[: args.seeds] if args.seeds else RUN_SEEDS
        generations = args.memetic_generations
        result_dir = Path(args.result_dir)

    result_dir.mkdir(exist_ok=True)
    (result_dir / "done").mkdir(exist_ok=True)

    anchor_lookup = _load_anchor_lookup(args.anchor_csv)

    print("=" * 80)
    print("ENSGA-II BASELINE  --  NSGA-II + sparsity-driven local search")
    print("=" * 80)
    print("Instances:", ", ".join(instances))
    print("Seeds:", seeds)
    print("Generations:", generations)
    print(f"Local search: neighborhoods=N1+N2 | subset_size={args.subset_size} "
          f"max_moves={args.max_moves} judgment_threshold={args.judgment_threshold}")
    print("Anchor source:", args.anchor_csv)
    print("Result dir:", result_dir)

    config = {
        "population_size": POPULATION_SIZE,
        "generations": generations,
        "method": METHOD_NAME,
        "local_search": {
            "neighborhoods": ["ensga2_n1", "ensga2_n2"],
            "subset_size": args.subset_size,
            "max_moves": args.max_moves,
            "judgment_threshold": args.judgment_threshold,
        },
        "seeds": seeds, "instances": instances,
        "anchor_csv": args.anchor_csv, "ref_factor": REF_FACTOR,
    }
    (result_dir / "ensga2_config.json").write_text(json.dumps(config, indent=2))

    for instance_name in instances:
        marker = result_dir / "done" / f"{instance_name}.done"
        if marker.exists():
            print(f"Skipping {instance_name}: already complete "
                  f"(remove {marker} to rerun).")
            continue

        process_instance(
            instance_name, seeds, generations, args.subset_size,
            args.max_moves, args.judgment_threshold, anchor_lookup, result_dir,
        )

        marker.write_text("done\n", encoding="utf-8")

    print("\n" + "=" * 80)
    print("ENSGA-II BASELINE RUN FINISHED")
    print("Per-seed rows in:", result_dir / "ensga2_runs.csv")
    print("=" * 80)


if __name__ == "__main__":
    main()
