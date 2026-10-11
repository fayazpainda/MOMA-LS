"""
Task 2 -- hyperparameter ablation with held-out tuning/test split.

WHY A SEPARATE TUNING SET
--------------------------
Every reported memetic result (results_memetic_n15, results_algo, etc.) used
the SAME local-search hyperparameters (subset_size=20, max_moves=20,
inject=True) directly on ta11-ta62 -- the paper's test set. These values were
never tuned; they are round-number defaults picked when ParetoLocalSearch was
first written (see `pareto_local_search.py`) and never revisited. There is no
evidence they were adjusted after seeing ta11-ta62 results (no post-hoc
fitting occurred), but there is also no evidence they are good choices: they
were never compared against alternatives on ANY data. A reviewer who asks "why
these numbers" has no defensible answer today.

TUNING SET
----------
ta13, ta23, ta33, ta43, ta63 -- one instance per size class (20x15, 20x20,
30x15, 30x20, 50x20), the SAME five size classes Wei's Table 9 and this
repo's test set (ta11/ta12, ta21/ta22, ..., ta61/ta62) already use, but
specific instances that appear in NEITHER the paper's Table 9 NOR any
reported memetic/AHP/robustness result. This avoids introducing a new size
class (which would confound instance-size generalization with hyperparameter
generalization) while keeping the tuning set genuinely disjoint from the test
set. All five already have Wei-extended data in data/wei_extended_taillard/ (generated
by the same generate_wei_dataset.py run that produced ta11-ta62), so no new
data generation is needed. Mirrors the existing stratification convention
used for the Task-3 robustness study (one instance per size class).

GRID (subset_size x max_moves x inject)
----------------------------------------
subset_size in {10, 20, 40}   -- 20 is the untested current default
max_moves   in {5, 20, 50}    -- 20 is the untested current default
inject      in {True, False}  -- the archive/injection policy toggle

= 18 configs. Run at a REDUCED budget (fewer generations/seeds than the
final n=15 test-set sweep) since this is a ranking/selection exercise, not
the reported result -- the reported result is the ONE chosen config,
re-run at full budget on the test set separately (memetic_comparison.py
default seeds/generations).

METRIC (optimize on the tuning set ONLY)
------------------------------------------
Mean equal-decode-budget normalized-HV gain (%), pooled across the 5 tuning
instances (memetic_analysis.py's `hv_delta_pct`, one row per instance,
averaged). This is the same fairness-controlled metric used for every
headline claim in this repo, just computed on tuning instances instead of
test instances.

KNOWN CAVEAT: TWET BASELINE FIDELITY IS WEAKER THAN Cmax/NEC (flag, not a
blocker)
-----------------------------------------------------------------------------
REPRODUCTION_NOTES.md's Task 1 closure (2026-09-12 audit) found that on the
ta11-ta62 TEST set, two instances (ta21, ta22) show our median TWET diverging
3-6x from Wei's median even though best-of-N still beats Wei -- i.e. TWET has
higher seed-to-seed variance / less centered behavior than Cmax or NEC on at
least some instances. This sweep's HV metric is purely self-referential
(memetic vs vanilla on the SAME tuning instance; it never compares against
Wei's Table 9, so the Wei-fidelity gap itself does not threaten this sweep's
validity). What DOES transfer is the underlying risk: if any of the tuning
instances (ta13/ta23/ta33/ta43/ta63) share that high-TWET-variance property,
the reduced-budget (5-seed) config ranking could be noisier specifically
because of TWET's contribution to HV, not because of a bad hyperparameter
choice. If the sensitivity table (hyperparam_analysis.py) shows unstable or
counterintuitive TWET-driven rankings, check this before concluding a config
is genuinely unstable -- it is a documented candidate explanation, not
something to rediscover from scratch.

PRE-FLIGHT CHECK RESULT (2026-09-12, see HYPERPARAM_SWEEP_PREFLIGHT.md): none
of the 5 tuning instances reaches ta21/ta22's TWET seed-to-seed CV (0.34/0.50)
-- all five sit at 0.047-0.134, in line with the tight comparison instances.
Cleared to run at the reduced budget below with no adjustment.

Usage (from PROJECT ROOT):

    # cheap sanity pass first (tiny budget, 2 configs) before committing to the
    # full grid:
    ./venv/bin/python src/hyperparam_sweep.py --dry-run

    # full grid, reduced tuning budget (default: 5 seeds, 50 generations)
    ./venv/bin/python src/hyperparam_sweep.py

    # then pick the winner and print the sensitivity table:
    ./venv/bin/python src/hyperparam_analysis.py
"""

import argparse
import itertools
import subprocess
import sys
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    class _NullBar:
        def update(self, *_): pass
        def set_postfix(self, *_, **__): pass
        def set_postfix_str(self, *_, **__): pass
        def close(self): pass
        def __enter__(self): return self
        def __exit__(self, *a): pass

    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else _NullBar()

TUNING_INSTANCES = ["ta13", "ta23", "ta33", "ta43", "ta63"]

SUBSET_SIZES = [10, 20, 40]
MAX_MOVES = [5, 20, 50]
INJECT = [True, False]

RESULT_ROOT = Path("results_hparam_tuning")


def config_name(subset_size, max_moves, inject):
    inject_tag = "inject" if inject else "noinject"
    return f"s{subset_size}_m{max_moves}_{inject_tag}"


def all_configs():
    for subset_size, max_moves, inject in itertools.product(
        SUBSET_SIZES, MAX_MOVES, INJECT
    ):
        yield subset_size, max_moves, inject


def run_config(subset_size, max_moves, inject, seeds, generations, instances, bar=None):
    name = config_name(subset_size, max_moves, inject)
    result_dir = RESULT_ROOT / name

    if bar is not None:
        bar.set_postfix_str(name)

    done_markers = [result_dir / "done" / f"{inst}.done" for inst in instances]
    if all(marker.exists() for marker in done_markers):
        print(f"[skip] {name}: already complete")
        return

    print(f"\n{'=' * 80}\n[run] {name}  (subset_size={subset_size}, "
          f"max_moves={max_moves}, inject={inject})\n{'=' * 80}")

    cmd = [
        sys.executable, "src/memetic_comparison.py",
        "--instances", *instances,
        "--seeds", str(seeds),
        "--memetic-generations", str(generations),
        "--subset-size", str(subset_size),
        "--max-moves", str(max_moves),
        "--result-dir", str(result_dir),
        "--analyze",
    ]
    if not inject:
        cmd.append("--no-inject")

    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", nargs="*", default=TUNING_INSTANCES)
    parser.add_argument("--seeds", type=int, default=5,
                        help="Tuning-phase seeds per config (default 5; the "
                             "final test-set sweep uses 10-15).")
    parser.add_argument("--generations", type=int, default=50,
                        help="Tuning-phase memetic generation budget (default "
                             "50; the final test-set sweep uses 150).")
    parser.add_argument("--dry-run", action="store_true",
                         help="Sanity pass: 1 instance (ta13), 2 seeds, 15 "
                              "generations, only 2 configs (the current "
                              "default and one alternative) -- confirms the "
                              "sweep plumbing works before committing hours.")
    args = parser.parse_args()

    if args.dry_run:
        instances = ["ta13"]
        seeds = 2
        generations = 15
        configs = [(20, 20, True), (10, 5, False)]
    else:
        instances = args.instances
        seeds = args.seeds
        generations = args.generations
        configs = list(all_configs())

    RESULT_ROOT.mkdir(exist_ok=True)

    print(f"Tuning instances: {', '.join(instances)}")
    print(f"Seeds per config: {seeds}  |  Generations per config: {generations}")
    print(f"Configs to run: {len(configs)}")

    with tqdm(total=len(configs), desc="hyperparam sweep", unit="config") as bar:
        for subset_size, max_moves, inject in configs:
            run_config(subset_size, max_moves, inject, seeds, generations, instances, bar=bar)
            bar.update(1)

    print("\nAll configs finished. Run: ./venv/bin/python src/hyperparam_analysis.py")


if __name__ == "__main__":
    main()
