"""
Render results_bmw_table.csv into the LaTeX table13_bmw_newbaselines.tex
figure, matching the formatting conventions of the existing
figures/table7_full_bmw.tex (OverLeaf 2): scriptsize, adjustbox,
multirow instance blocks with size labels, comma thousands separators,
bold = best (lowest) value per column within each instance block.
"""

import pandas as pd

SIZE_LABELS = {
    "ta11": "20$\\times$15", "ta12": "20$\\times$15",
    "ta21": "20$\\times$20", "ta22": "20$\\times$20",
    "ta31": "30$\\times$15", "ta32": "30$\\times$15",
    "ta41": "30$\\times$20", "ta42": "30$\\times$20",
    "ta61": "50$\\times$20", "ta62": "50$\\times$20",
}

ALGO_ORDER = ["unsga3_vanilla", "spea2_vanilla", "sms_emoa_vanilla", "ensga2", "moma_ls"]
ALGO_LABEL = {
    "unsga3_vanilla": "U-NSGA-III",
    "spea2_vanilla": "SPEA2",
    "sms_emoa_vanilla": "SMS-EMOA",
    "ensga2": "ENSGA-II",
    "moma_ls": "\\textbf{MOMA-LS}",
}
OBJS = ["Cmax", "TWET", "NEC"]
STATS = ["best", "median", "worst"]

INSTANCES = ["ta11", "ta12", "ta21", "ta22", "ta31", "ta32", "ta41", "ta42", "ta61", "ta62"]


def fmt(v):
    return f"{round(v):,}"


def main():
    df = pd.read_csv("results_bmw_table.csv").set_index(["instance", "algorithm"])

    lines = []
    lines.append(r"\begin{table*}[!t]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(r"\caption{Best/median/worst objective values across ten instances: "
                 r"U-NSGA-III, SPEA2, SMS-EMOA, ENSGA-II, and MOMA-LS, each in its native form.}")
    lines.append(r"\label{tab:bmw-newbaselines}")
    lines.append(r"\begin{adjustbox}{max width=\textwidth}")
    lines.append(r"\begin{tabular}{@{}cl ccc ccc ccc@{}}")
    lines.append(r"\toprule")
    lines.append(r"\multirow{2}{*}{Instance (size)} & \multirow{2}{*}{Algorithm} & "
                 r"\multicolumn{3}{c}{Makespan ($C_{\max}$)} & \multicolumn{3}{c}{TWET} & \multicolumn{3}{c}{NEC} \\")
    lines.append(r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}")
    lines.append(r" & & Best & Median & Worst & Best & Median & Worst & Best & Median & Worst \\")
    lines.append(r"\midrule")

    for i, inst in enumerate(INSTANCES):
        # determine per-column minimum across the 5 algorithms for bolding
        mins = {}
        for obj in OBJS:
            for stat in STATS:
                col = f"{obj}_{stat}"
                mins[col] = min(df.loc[(inst, a), col] for a in ALGO_ORDER)

        lines.append(
            r"\multirow{5}{*}{\makecell{" + inst + r"\\(" + SIZE_LABELS[inst] + r")}}"
            + f" & {ALGO_LABEL[ALGO_ORDER[0]]}"
            + _row_cells(df, inst, ALGO_ORDER[0], mins)
            + r" \\"
        )
        for algo in ALGO_ORDER[1:]:
            lines.append(f" & {ALGO_LABEL[algo]}" + _row_cells(df, inst, algo, mins) + r" \\")
        if i < len(INSTANCES) - 1:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"\end{adjustbox}")
    lines.append(r"\end{table*}")

    out = "\n".join(lines) + "\n"
    with open("OverLeaf 2/figures/table13_bmw_newbaselines.tex", "w") as f:
        f.write(out)
    print(out)


def _row_cells(df, inst, algo, mins):
    cells = []
    for obj in OBJS:
        for stat in STATS:
            col = f"{obj}_{stat}"
            v = df.loc[(inst, algo), col]
            s = fmt(v)
            if round(v) == round(mins[col]):
                s = r"\textbf{" + s + "}"
            cells.append(s)
    return " & " + " & ".join(cells)


if __name__ == "__main__":
    main()
