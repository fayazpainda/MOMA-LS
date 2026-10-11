"""
Head-to-head MOMA-LS (unsga3_memetic) vs SPEA2+LS (spea2_memetic) across
all 10 main instances (not just the 5 robustness instances), to check
whether SPEA2 is a genuinely better host once given the full instance set.

Bounds (ideal/nadir) are recomputed per instance from the union of ONLY
these two methods' final fronts (a fair head-to-head bounds, same
approach as src/algo_merge.py), then normalised HV is recomputed for
every run under those bounds. A paired one-sided Wilcoxon signed-rank
test (MOMA-LS > SPEA2, matching the direction tested in Table 10) is run
on the resulting 150 (10 instances x 15 seeds) paired HV values.
"""

import pandas as pd
from scipy.stats import wilcoxon

from hypervolume import normalized_hypervolume

REF_FACTOR = 1.1
MAIN_INSTANCES = ["ta11", "ta12", "ta21", "ta22", "ta31", "ta32", "ta41", "ta42", "ta61", "ta62"]
BMW_SEVEN = ["ta11", "ta12", "ta22", "ta31", "ta32", "ta42", "ta62"]


def load_spea2_memetic_fronts():
    base = pd.read_csv("results_algo_optionB_tuned/algo_final_pareto.csv")
    base = base[(base.method == "spea2_memetic") & (base.instance.isin(["ta21", "ta41", "ta61"]))]
    parts = [base[["instance", "run", "seed", "Cmax", "TWET", "NEC"]]]
    for inst in BMW_SEVEN:
        d = pd.read_csv(f"results_algo_bmw_spea2_{inst}/algo_final_pareto.csv")
        d = d[d.method == "spea2_memetic"]
        parts.append(d[["instance", "run", "seed", "Cmax", "TWET", "NEC"]])
    out = pd.concat(parts, ignore_index=True)
    out["method"] = "spea2_memetic"
    return out


def load_moma_ls_fronts():
    df = pd.read_csv("results_memetic_tuned_n15/memetic_final_pareto.csv")
    df = df[df.config == "memetic"]
    out = df[["instance", "run", "seed", "Cmax", "TWET", "NEC"]].copy()
    out["method"] = "unsga3_memetic"
    return out


def main():
    fronts = pd.concat([load_spea2_memetic_fronts(), load_moma_ls_fronts()], ignore_index=True)

    hv_rows = []
    for inst, g in fronts.groupby("instance"):
        objs = g[["Cmax", "TWET", "NEC"]].to_numpy()
        ideal = objs.min(axis=0).tolist()
        nadir = objs.max(axis=0).tolist()
        for (method, run, seed), front in g.groupby(["method", "run", "seed"]):
            vectors = list(front[["Cmax", "TWET", "NEC"]].itertuples(index=False, name=None))
            hv = normalized_hypervolume(vectors, ideal, nadir, ref_factor=REF_FACTOR)
            hv_rows.append({"instance": inst, "method": method, "seed": seed, "hv": hv})

    hv_df = pd.DataFrame(hv_rows)
    hv_df.to_csv("results_spea2_vs_moma_10instances_hv.csv", index=False)

    pivot = hv_df.pivot_table(index=["instance", "seed"], columns="method", values="hv")
    pivot = pivot.dropna()
    assert len(pivot) == 150, f"expected 150 paired rows, got {len(pivot)}"

    print("Mean HV per instance:")
    mean_by_inst = hv_df.groupby(["instance", "method"]).hv.mean().unstack()
    mean_by_inst["winner"] = mean_by_inst.idxmax(axis=1)
    print(mean_by_inst.to_string())
    print()
    wins = (mean_by_inst["winner"] == "unsga3_memetic").sum()
    print(f"MOMA-LS wins {wins}/10 instances on mean HV; SPEA2 wins {10 - wins}/10.")
    print()

    diffs = pivot["unsga3_memetic"] - pivot["spea2_memetic"]
    stat, p_greater = wilcoxon(diffs, alternative="greater")
    stat2, p_two = wilcoxon(diffs, alternative="two-sided")
    print(f"Paired one-sided Wilcoxon (MOMA-LS > SPEA2+LS), n=150 pairs: "
          f"statistic={stat:.1f}, p={p_greater:.4f}")
    print(f"Two-sided Wilcoxon: p={p_two:.4f}")
    print(f"Mean HV: MOMA-LS={pivot['unsga3_memetic'].mean():.4f}, "
          f"SPEA2+LS={pivot['spea2_memetic'].mean():.4f}")
    print(f"MOMA-LS wins {int((diffs > 0).sum())}/150 seed-level pairs, "
          f"SPEA2+LS wins {int((diffs < 0).sum())}/150, ties {int((diffs == 0).sum())}")


if __name__ == "__main__":
    main()
