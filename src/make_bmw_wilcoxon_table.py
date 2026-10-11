"""
Build the significance data for table14_bmw_wilcoxon.tex: for each of the
four additional baselines (U-NSGA-III vanilla, SPEA2, SMS-EMOA, ENSGA-II)
against MOMA-LS, a paired one-sided Wilcoxon signed-rank test (n=15 seeds)
per instance and objective, corrected for the 30 tests (10 instances x 3
objectives) within each competitor's own family via Holm-Bonferroni.

Reads results_bmw_merged_long.csv (written by make_bmw_table.py).
"""

import pandas as pd
from scipy.stats import wilcoxon

MAIN_INSTANCES = ["ta11", "ta12", "ta21", "ta22", "ta31", "ta32", "ta41", "ta42", "ta61", "ta62"]
OBJS = ["Cmax", "TWET", "NEC"]
COMPETITORS = ["unsga3_vanilla", "spea2_vanilla", "sms_emoa_vanilla", "ensga2"]


def main():
    long_df = pd.read_csv("results_bmw_merged_long.csv")
    moma = long_df[long_df.algorithm == "moma_ls"].set_index(["instance", "seed"])

    rows = []
    for competitor in COMPETITORS:
        comp = long_df[long_df.algorithm == competitor].set_index(["instance", "seed"])
        for instance in MAIN_INSTANCES:
            for obj in OBJS:
                m = moma.loc[instance, obj]
                c = comp.loc[instance, obj]
                diffs = (c - m).values  # positive = MOMA-LS better (lower)
                stat, p = wilcoxon(diffs, alternative="greater")
                rows.append({"competitor": competitor, "instance": instance,
                             "objective": obj, "p_raw": p})

    results = pd.DataFrame(rows)

    # Holm-Bonferroni within each competitor's own family of 30 tests.
    results["significant"] = False
    for competitor, g in results.groupby("competitor"):
        g = g.sort_values("p_raw").reset_index()
        m = len(g)
        thresholds = [0.05 / (m - i) for i in range(m)]
        sig = []
        rejecting = True
        for p, thr in zip(g["p_raw"], thresholds):
            rejecting = rejecting and (p < thr)
            sig.append(rejecting)
        results.loc[g["index"], "significant"] = sig

    results.to_csv("results_bmw_wilcoxon.csv", index=False)

    print("Significant tests per competitor x objective (of 10 instances):")
    summary = results.groupby(["competitor", "objective"])["significant"].sum().unstack()
    summary["Total"] = summary.sum(axis=1)
    print(summary.to_string())

    print("\nAny non-significant tests:")
    print(results[~results.significant].to_string(index=False))


if __name__ == "__main__":
    main()
