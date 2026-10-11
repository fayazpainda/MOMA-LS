"""
Task 2(c)/(d) -- read every config's memetic_summary.csv (written by
memetic_comparison.py --analyze / memetic_analysis.py) out of
results_hparam_tuning/<config>/, score each config by its mean equal-budget
normalized-HV gain (hv_delta_pct) pooled across the tuning instances, and
report:

  1. A ranked table of all configs (the sensitivity table Task 2(c) asks for
     -- this is the honesty check: if the gain is roughly flat across the
     grid, the headline result is NOT hyperparameter-fragile; if it swings
     wildly, that must be disclosed).
  2. The winning config, to be re-run at FULL budget on the TEST set
     (ta11-ta62) as the paper's reported hyperparameters.
  3. An explicit flag comparing the winner to the untested current default
     (subset_size=20, max_moves=20, inject=True) -- Task 2(d)'s post-hoc
     fitting check: since the default was never tuned before this sweep, if
     it happens to win, that's a lucky coincidence worth stating plainly, not
     evidence of prior tuning.

Usage (from PROJECT ROOT):

    ./venv/bin/python src/hyperparam_analysis.py
"""

import argparse
import csv
from pathlib import Path

from hyperparam_sweep import RESULT_ROOT, SUBSET_SIZES, MAX_MOVES, INJECT, config_name

CURRENT_DEFAULT = (20, 20, True)


def load_config_score(result_dir):
    summary_path = result_dir / "memetic_summary.csv"
    if not summary_path.exists():
        return None

    with open(summary_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        return None

    # memetic_summary.csv has one row per (instance, objective) -- hv_delta_pct
    # is per-instance and therefore duplicated across each instance's 3
    # objective rows. Deduplicate by instance before averaging.
    per_instance = {}
    for r in rows:
        if r.get("hv_delta_pct"):
            per_instance[r["instance"]] = float(r["hv_delta_pct"])

    deltas = list(per_instance.values())
    if not deltas:
        return None

    return {
        "n_instances": len(deltas),
        "mean_hv_delta_pct": sum(deltas) / len(deltas),
        "min_hv_delta_pct": min(deltas),
        "max_hv_delta_pct": max(deltas),
        "per_instance": deltas,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", default=str(RESULT_ROOT))
    args = parser.parse_args()

    result_root = Path(args.result_root)

    scored = []
    for subset_size in SUBSET_SIZES:
        for max_moves in MAX_MOVES:
            for inject in INJECT:
                name = config_name(subset_size, max_moves, inject)
                result_dir = result_root / name
                score = load_config_score(result_dir)
                if score is None:
                    continue
                scored.append({
                    "subset_size": subset_size,
                    "max_moves": max_moves,
                    "inject": inject,
                    "name": name,
                    **score,
                })

    if not scored:
        print(f"No completed configs found under {result_root}/. "
              "Run hyperparam_sweep.py first.")
        return

    scored.sort(key=lambda c: c["mean_hv_delta_pct"], reverse=True)

    print("=" * 100)
    print("TUNING-SET SENSITIVITY TABLE (mean equal-budget normalized-HV gain %, "
          f"pooled over {scored[0]['n_instances']} tuning instances)")
    print("=" * 100)
    print(f"{'subset_size':>11} {'max_moves':>10} {'inject':>7} {'mean Δ%':>10} "
          f"{'min Δ%':>8} {'max Δ%':>8}")

    for c in scored:
        marker = "  <- current default (untested before this sweep)" if (
            (c["subset_size"], c["max_moves"], c["inject"]) == CURRENT_DEFAULT
        ) else ""
        print(f"{c['subset_size']:>11} {c['max_moves']:>10} {str(c['inject']):>7} "
              f"{c['mean_hv_delta_pct']:>9.1f}% {c['min_hv_delta_pct']:>7.1f}% "
              f"{c['max_hv_delta_pct']:>7.1f}%{marker}")

    best = scored[0]
    worst = scored[-1]
    spread = best["mean_hv_delta_pct"] - worst["mean_hv_delta_pct"]

    print("\n" + "-" * 100)
    print(f"WINNER: subset_size={best['subset_size']}, max_moves={best['max_moves']}, "
          f"inject={best['inject']}  (mean ΔHV = {best['mean_hv_delta_pct']:+.1f}%)")
    print(f"Grid spread (best - worst mean ΔHV): {spread:.1f} percentage points "
          f"across {len(scored)} configs.")

    if len(scored) < len(SUBSET_SIZES) * len(MAX_MOVES) * len(INJECT):
        print(f"NOTE: only {len(scored)}/"
              f"{len(SUBSET_SIZES) * len(MAX_MOVES) * len(INJECT)} configs have "
              "completed results -- this table is partial.")

    is_default_winner = (
        (best["subset_size"], best["max_moves"], best["inject"]) == CURRENT_DEFAULT
    )
    print(f"\nCurrent default (20, 20, inject=True) is the winner: {is_default_winner}.")
    if is_default_winner:
        print("This is a coincidence, not evidence of prior tuning -- the default was")
        print("never compared against alternatives before this sweep (see module docstring).")

    print("\nNEXT STEP: re-run memetic_comparison.py on the TEST set (ta11-ta62) with the")
    print(f"winning hyperparameters (--subset-size {best['subset_size']} --max-moves "
          f"{best['max_moves']}" + (" --no-inject" if not best["inject"] else "") +
          "), at the FULL protocol (10-15 seeds, 150 generations), and report that as the")
    print("paper's headline result. Report this sensitivity table alongside it so reviewers")
    print("can judge how fragile the gain is to the exact hyperparameter choice.")


if __name__ == "__main__":
    main()
