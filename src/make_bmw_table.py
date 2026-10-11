"""
Build the BMW (Best/Median/Worst) table data for the OverLeaf 2 paper,
matching Wei2021's Table 9 format.

Gathers each algorithm's NATIVE-form rows (no cross-pollinated local search)
across the 10 main instances (ta11, ta12, ta21, ta22, ta31, ta32, ta41,
ta42, ta61, ta62), from wherever that algorithm's run for that instance
happens to already exist on disk, then reports Best/Median/Worst of
Cmax/TWET/NEC over the 15 seeds.

Sources:
- unsga3_vanilla, unsga3_memetic (= MOMA-LS): results_memetic_tuned_n15/memetic_runs.csv
- spea2_vanilla: results_algo_optionB_tuned (ta21/ta41/ta61) +
  results_algo_bmw_spea2_ta{11,12,22,31,32,42,62} (the other 7)
- sms_emoa_vanilla: results_algo_sms_emoa_robustness_ta{21,41,61} +
  results_algo_bmw_sms_emoa_ta{11,12,22,31,32,42,62} (the other 7)
- ensga2 (NSGA-II host + ENSGA-II's own sparsity LS): results_ensga2_ta*/ensga2_runs.csv
"""

import pandas as pd

MAIN_INSTANCES = ["ta11", "ta12", "ta21", "ta22", "ta31", "ta32", "ta41", "ta42", "ta61", "ta62"]
BMW_SEVEN = ["ta11", "ta12", "ta22", "ta31", "ta32", "ta42", "ta62"]
OBJ_COLS = ["Cmax", "TWET", "NEC"]


def _long(df, instance_col, method_label, obj_cols=("best_Cmax", "best_TWET", "best_NEC")):
    out = df[[instance_col, "seed", *obj_cols]].copy()
    out.columns = ["instance", "seed", "Cmax", "TWET", "NEC"]
    out["algorithm"] = method_label
    return out


def load_unsga3():
    df = pd.read_csv("results_memetic_tuned_n15/memetic_runs.csv")
    vanilla = df[df.config == "vanilla"]
    memetic = df[df.config == "memetic"]
    return pd.concat([
        _long(vanilla, "instance", "unsga3_vanilla"),
        _long(memetic, "instance", "moma_ls"),
    ], ignore_index=True)


def load_spea2():
    base = pd.read_csv("results_algo_optionB_tuned/algo_runs.csv")
    base = base[(base.method == "spea2_vanilla") & (base.instance.isin(["ta21", "ta41", "ta61"]))]
    parts = [_long(base, "instance", "spea2_vanilla")]
    for inst in BMW_SEVEN:
        d = pd.read_csv(f"results_algo_bmw_spea2_{inst}/algo_runs.csv")
        d = d[d.method == "spea2_vanilla"]
        parts.append(_long(d, "instance", "spea2_vanilla"))
    return pd.concat(parts, ignore_index=True)


def load_sms_emoa():
    parts = []
    for inst in ["ta21", "ta41", "ta61"]:
        d = pd.read_csv(f"results_algo_sms_emoa_robustness_{inst}/algo_runs.csv")
        d = d[d.method == "sms_emoa_vanilla"]
        parts.append(_long(d, "instance", "sms_emoa_vanilla"))
    for inst in BMW_SEVEN:
        d = pd.read_csv(f"results_algo_bmw_sms_emoa_{inst}/algo_runs.csv")
        d = d[d.method == "sms_emoa_vanilla"]
        parts.append(_long(d, "instance", "sms_emoa_vanilla"))
    return pd.concat(parts, ignore_index=True)


def load_ensga2():
    parts = []
    for inst in MAIN_INSTANCES:
        d = pd.read_csv(f"results_ensga2_{inst}/ensga2_runs.csv")
        parts.append(_long(d, "instance", "ensga2"))
    return pd.concat(parts, ignore_index=True)


def main():
    long_df = pd.concat(
        [load_unsga3(), load_spea2(), load_sms_emoa(), load_ensga2()],
        ignore_index=True,
    )

    # sanity: every (algorithm, instance) pair should have exactly 15 seeds
    counts = long_df.groupby(["algorithm", "instance"]).size()
    bad = counts[counts != 15]
    if len(bad):
        print("WARNING: unexpected seed counts:")
        print(bad)

    long_df.to_csv("results_bmw_merged_long.csv", index=False)

    rows = []
    for (algo, inst), g in long_df.groupby(["algorithm", "instance"]):
        row = {"algorithm": algo, "instance": inst, "n_seeds": len(g)}
        for obj in OBJ_COLS:
            row[f"{obj}_best"] = g[obj].min()
            row[f"{obj}_median"] = g[obj].median()
            row[f"{obj}_worst"] = g[obj].max()
        rows.append(row)
    bmw = pd.DataFrame(rows).sort_values(["instance", "algorithm"])
    bmw.to_csv("results_bmw_table.csv", index=False)

    print(f"Wrote results_bmw_merged_long.csv ({len(long_df)} rows) "
          f"and results_bmw_table.csv ({len(bmw)} rows).")
    print("\nAlgorithms:", sorted(long_df.algorithm.unique()))
    print("Instances:", sorted(long_df.instance.unique()))
    print(f"\nPreview:\n{bmw.to_string(index=False)}")


if __name__ == "__main__":
    main()
