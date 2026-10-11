# MOMA-LS

Code and result data for *MOMA-LS: A Memetic U-NSGA-III with Critical-Block Local Search for Energy-Efficient Job Shop Scheduling*.

## Structure

- `src/` — core algorithm (chromosome, decoder, critical-path N5 local search, U-NSGA-III), baselines (SPEA2, SMS-EMOA, ENSGA-II), and the scripts used to generate the paper's tables and the final component ablation.
- `data/taillard/` and `data/wei_extended_taillard/` — benchmark instances used in the experiments.
- `results/` — result data backing the paper's tables and figures: main hypervolume comparison (`results_memetic_tuned_n15`), held-out validation (`results_newgen_ta*`), ENSGA-II baseline runs (`results_ensga2_ta*`), and the final component ablation (`results_ablation_*_exact`, `results_ablation_ta*_primary`), plus top-level BMW baseline comparison CSVs.

## Setup

```
pip install -r requirements.txt
```
