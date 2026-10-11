"""
Limitation #4 extension -- a 2nd instance per size class for the component
ablation (OverLeaf 2/sections/results.tex sub:ablation currently uses one
instance per size class: ta12, ta21, ta34, ta41, ta62).

New instances (confirmed untouched by any tuning/test/robustness/ablation
run so far, and distinct from the Limitation #2 robustness-extension picks
ta15/ta24/ta35/ta44/ta64): ta16 (20x15), ta25 (20x20), ta36 (30x15),
ta45 (30x20), ta65 (50x20).

Reconstructs the paper's exact 3-config methodology on each new instance:
    config 1 "naive"           -- accept=cmax_greedy, no-inject
    config 2 "pareto_noinject" -- accept=dominance,   no-inject
    config 3 "primary"         -- accept=dominance,   inject (= full MOMA-LS,
                                   identical to the headline method)

PIPELINE (3 phases, each gated on the previous; see memetic_comparison.py's
--match-decodes-csv docstring for why this is needed, not just a straight
--subset-size=10 run for every config):

  Phase 1 (anchor): run config 3 for 15 seeds on each new instance. This is
      both a real result (config 3 = MOMA-LS) and the decode-budget anchor
      configs 1/2 get truncated to.

  Phase 2 (K-probe): configs 1/2 consume decodes at a different RATE than
      config 3 at the same subset_size (naive's more permissive acceptance
      and noinject's missing archive overhead both change how many hill-
      climb decodes happen per generation). A single-seed probe over a
      small grid of subset_size values around the K=10 default finds the
      K whose 150-generation decode trajectory lands closest to config 3's
      own decode total for that instance -- this keeps the per-generation
      search INTENSITY comparable (same ballpark K) rather than just
      truncating an arbitrarily-different-K run down to size, which was
      the actual reason the original ablation (ta12/21/34/41/62) needed
      this step instead of relying on --match-decodes-csv truncation alone.

  Phase 3 (exact): full 15-seed run of configs 1/2 at their chosen K, with
      --match-decodes-csv pointing at Phase 1's anchor CSV, so every seed's
      reported decode total is truncated to not exceed config 3's own
      decode total for that exact seed (the "_exact" naming convention
      matches the existing results_ablation_{naive,pareto_noinject}_*_exact
      directories from the original 5-instance ablation).

15 seeds (not 20) throughout -- matches every other instance already in
Table 11, per the project's seed-count-uniformity decision.

Usage (from the project root):
    ./venv/bin/python src/run_ablation_extension.py

Resumable: each phase/job checks for its own already-written output
(done marker or non-empty runs CSV) before running, so Ctrl+C + rerun only
redoes the interrupted job, not everything before it.
"""

import csv
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PYTHON = "./venv/bin/python"
DRIVER = "src/memetic_comparison.py"

INSTANCES = ["ta16", "ta25", "ta36", "ta45", "ta65"]
SEEDS = 15
PROBE_SEED_COUNT = 1          # single-seed probe (RUN_SEEDS[:1] == [1001])
K_DEFAULT = 10
K_GRID = [7, 8, 9, 10, 11, 12, 13]
MAX_MOVES = 50
MAX_WORKERS = 8

CONFIGS = {
    "naive": {"accept": "cmax_greedy", "no_inject": True},
    "pareto_noinject": {"accept": "dominance", "no_inject": True},
}


def run(cmd, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True)


def runs_csv_decodes(result_dir, instance, seed=None):
    """Read {seed: decodes} (or a single seed's decodes) for the 'memetic'
    rows of result_dir/memetic_runs.csv. Returns {} if the file doesn't
    exist yet (job not done)."""
    path = Path(result_dir) / "memetic_runs.csv"
    if not path.exists():
        return {}
    out = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row["instance"] == instance and row["config"] == "memetic":
                out[int(row["seed"])] = int(row["decodes"])
    if seed is not None:
        return {seed: out[seed]} if seed in out else {}
    return out


def phase1_anchor(instance):
    result_dir = f"results_ablation_{instance}_primary"
    if len(runs_csv_decodes(result_dir, instance)) >= SEEDS:
        print(f"[skip] phase1 {instance}: already has {SEEDS} seeds")
        return result_dir
    cmd = [
        PYTHON, DRIVER,
        "--instances", instance, "--seeds", str(SEEDS),
        "--subset-size", str(K_DEFAULT), "--max-moves", str(MAX_MOVES),
        "--accept", "dominance",
        "--result-dir", result_dir,
    ]
    print(f"[run] phase1 anchor {instance} -> {result_dir}")
    run(cmd, Path(result_dir) / "run.log")
    return result_dir


def phase2_probe_one(instance, config_name, accept, no_inject, k, anchor_seed1):
    result_dir = f"results_ablation_probe_{config_name}_{instance}_K{k}"
    cached = runs_csv_decodes(result_dir, instance, seed=1001)
    if cached:
        decodes = cached[1001]
    else:
        cmd = [
            PYTHON, DRIVER,
            "--instances", instance, "--seeds", str(PROBE_SEED_COUNT),
            "--subset-size", str(k), "--max-moves", str(MAX_MOVES),
            "--accept", accept,
            "--result-dir", result_dir,
        ]
        if no_inject:
            cmd.append("--no-inject")
        print(f"[run] phase2 probe {config_name} {instance} K={k}")
        run(cmd, Path(result_dir) / "run.log")
        decodes = runs_csv_decodes(result_dir, instance, seed=1001)[1001]
    gap = abs(decodes - anchor_seed1)
    return k, decodes, gap


def phase2_choose_k(instance, config_name, accept, no_inject, anchor_seed1):
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futs = [
            pool.submit(phase2_probe_one, instance, config_name, accept,
                        no_inject, k, anchor_seed1)
            for k in K_GRID
        ]
        for fut in futs:
            results.append(fut.result())
    best_k, best_decodes, best_gap = min(results, key=lambda r: r[2])
    print(f"  {instance}/{config_name}: K-probe results "
          f"{[(k, d) for k, d, _ in results]}, anchor={anchor_seed1} "
          f"-> chosen K={best_k} (decodes={best_decodes}, gap={best_gap})")
    return best_k


def phase3_exact(instance, config_name, accept, no_inject, k, anchor_csv):
    result_dir = f"results_ablation_{config_name}_{instance}_exact"
    if len(runs_csv_decodes(result_dir, instance)) >= SEEDS:
        print(f"[skip] phase3 {config_name} {instance}: already has {SEEDS} seeds")
        return
    cmd = [
        PYTHON, DRIVER,
        "--instances", instance, "--seeds", str(SEEDS),
        "--subset-size", str(k), "--max-moves", str(MAX_MOVES),
        "--accept", accept,
        "--match-decodes-csv", anchor_csv,
        "--result-dir", result_dir,
    ]
    if no_inject:
        cmd.append("--no-inject")
    print(f"[run] phase3 exact {config_name} {instance} K={k} -> {result_dir}")
    run(cmd, Path(result_dir) / "run.log")


def main():
    # ---- Phase 1: anchor, all 5 instances in parallel ----
    print("=" * 80)
    print("PHASE 1: config-3 (primary/anchor) runs,", len(INSTANCES), "instances")
    print("=" * 80)
    anchor_dirs = {}
    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(INSTANCES))) as pool:
        futs = {inst: pool.submit(phase1_anchor, inst) for inst in INSTANCES}
        for inst, fut in futs.items():
            anchor_dirs[inst] = fut.result()

    # ---- Phase 2: K-probes, all (instance, config) pairs in parallel ----
    print("=" * 80)
    print("PHASE 2: single-seed K-probes")
    print("=" * 80)
    chosen_k = {}
    for inst in INSTANCES:
        anchor_seed1 = runs_csv_decodes(anchor_dirs[inst], inst, seed=1001)[1001]
        for config_name, params in CONFIGS.items():
            k = phase2_choose_k(inst, config_name, params["accept"],
                                 params["no_inject"], anchor_seed1)
            chosen_k[(inst, config_name)] = k

    # ---- Phase 3: full 15-seed exact-matched runs ----
    print("=" * 80)
    print("PHASE 3: 15-seed exact decode-matched runs")
    print("=" * 80)
    jobs = []
    for inst in INSTANCES:
        anchor_csv = f"{anchor_dirs[inst]}/memetic_runs.csv"
        for config_name, params in CONFIGS.items():
            k = chosen_k[(inst, config_name)]
            jobs.append((inst, config_name, params["accept"],
                         params["no_inject"], k, anchor_csv))
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futs = [pool.submit(phase3_exact, *job) for job in jobs]
        for fut in futs:
            fut.result()

    print("=" * 80)
    print("ALL ABLATION-EXTENSION PHASES FINISHED")
    print("=" * 80)
    print("Result dirs: results_ablation_{ta16,ta25,ta36,ta45,ta65}_primary/")
    print("             results_ablation_{naive,pareto_noinject}_{ta16,ta25,ta36,ta45,ta65}_exact/")
    print("Chosen K per (instance, config):", chosen_k)


if __name__ == "__main__":
    sys.exit(main())
