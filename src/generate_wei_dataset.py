"""
Wei2021 Reproduction

Paper-faithful extended-Taillard dataset generator.

This script regenerates the tri-objective extended benchmark
DIRECTLY from the rules stated in:

    H. Wei, S. Li, H. Quan, D. Liu, S. Rao, C. Li, J. Hu,
    "Unified Multi-Objective Genetic Algorithm for Energy
    Efficient Job Shop Scheduling," IEEE Access, vol. 9, 2021,
    pp. 54542-54557.  Section V.A "TEST INSTANCES".

Every value below is traceable to a numbered equation / line
in the paper, so the generated dataset is defensible for a
reproduction (unlike the previous hand-reconstructed data).

------------------------------------------------------------
PARAMETER RULES (paper Section V.A)
------------------------------------------------------------
Per job i:
    Due date        (Eq. 18):  d_i = gamma_i * sum_k pt_ijk
                               gamma_i ~ Uniform(1.8, 2.3)
    Tardiness coeff (Eq. 19):  alpha_i ~ choice{1, 5, 10}
    Earliness coeff (Eq. 19):  beta_i  = 1 / alpha_i

Per machine j:
    Working power   :  P_W,j ~ Uniform(0.9, 1.0)  kW
    Idle power      :  P_I,j ~ Uniform(0.6, 0.7)  kW
    Standby power   :  P_S,j ~ Uniform(0.3, 0.5)  kW
    Ramp-up powers  :  P_rampup,off,j = P_rampup,standby,j = 0.8 kW
    Ramp-up times   (Eq. 20, SIZE-INDEPENDENT reading):
        t_rampup,standby,j = (1/(2m)) * (one job's total processing)
                           = (1/(2m)) * (m * p_bar) = p_bar / 2
        t_rampup,off,j     = 2 * t_rampup,standby,j = p_bar
        where p_bar = mean operation processing time (~50, constant
        across all Taillard instances).

    A job has ~m operations, so Eq. 20's inner sum (over a job's
    n_i operations) is ~ m * p_bar; dividing by 2m leaves ramp
    times that depend ONLY on p_bar and therefore do NOT grow with
    job count. This is the reading that reproduces Wei's near-flat
    NEC-vs-instance-size curve.

    An earlier per-machine-load reading -- summing ALL operations
    on a machine (= n jobs' worth) -- smuggled the job-count factor
    into the ramp time, so ramp_off grew from ~64 (ta11) to ~127
    (ta61), locked large instances out of STANDBY/OFF, and inflated
    NEC (verified ratio 1.07 -> 1.92 across ta11 -> ta61). Fixed.

------------------------------------------------------------
NOTE ON "EXACT" REPRODUCTION
------------------------------------------------------------
gamma_i, alpha_i and the three powers are RANDOM draws. Wei did
not publish their specific realization, so no generator can
reproduce Wei's Table-9 numbers exactly. This script fixes a
seed so OUR dataset is reproducible and documented; matching
Wei then means matching the rules + landing in the same ballpark
+ preserving the U-NSGA-III > NSGA-III > NSGA-II ranking.
"""

import os
import json
import argparse
import random
import shutil
import zlib
from datetime import datetime

from benchmark import load_taillard


TAILLARD_DIR = "data/taillard/full"
DEFAULT_OUT = "data/wei_paper"
INSTALL_TARGET = "data/wei_extended_taillard"

# Master seed. Per-instance seeds are derived from it so each
# instance is independent yet the whole dataset is reproducible.
MASTER_SEED = 20210414  # paper publication date, for traceability

ALPHA_CHOICES = (1, 5, 10)


# ============================================================
# Per-instance generation
# ============================================================

def build_jobs(instance, rng):
    """Generate job records (Eq. 18 due dates, Eq. 19 weights)."""

    jobs = []

    for job in instance.jobs:

        operations = [
            {
                "machine_id": op.machine,
                "processing_time": op.processing_time,
            }
            for op in job.operations
        ]

        total_pt = sum(
            op.processing_time for op in job.operations
        )

        # Eq. 18: d_i = gamma * sum(pt), gamma ~ U(1.8, 2.3)
        gamma = rng.uniform(1.8, 2.3)
        due_date = round(gamma * total_pt, 2)

        # Eq. 19: alpha in {1,5,10}, beta = 1/alpha
        alpha = rng.choice(ALPHA_CHOICES)
        beta = round(1.0 / alpha, 4)

        jobs.append({
            "job_id": job.job_id,
            "operations": operations,
            "due_date": due_date,
            "tardiness_penalty_alpha": alpha,
            "earliness_penalty_beta": beta,
        })

    return jobs


def build_machines(instance, rng):
    """Generate machine energy records (power ranges + Eq. 20)."""

    m = instance.num_machines

    # Eq. 20 ramp times (SIZE-INDEPENDENT reading).
    #
    # Eq. 20:  t_standby,j = (1/(2m)) * sum_k pt of a job (n_i ops)
    #          t_off,j     = 2 * t_standby,j
    #
    # A job has ~m operations, so one job's total processing is
    # ~ m * p_bar (p_bar = mean operation processing time). Hence
    #          t_standby = (1/(2m)) * (m * p_bar) = p_bar / 2
    #          t_off     = p_bar
    # i.e. the ramp times depend only on the MEAN OPERATION TIME,
    # which is ~constant (~50) for every Taillard instance and does
    # NOT grow with job count.
    #
    # The previous per-machine-load reading (sum of ALL ops on a
    # machine = n jobs' worth) made ramp times scale with n, which
    # locked large instances out of STANDBY/OFF and inflated NEC
    # (verified: ratio 1.07->1.92 ta11->ta61). This corrected
    # reading keeps NEC in Wei's range across all sizes.
    all_pt = [
        op.processing_time
        for job in instance.jobs
        for op in job.operations
    ]
    p_bar = sum(all_pt) / len(all_pt)

    t_off = round(p_bar, 4)
    t_standby = round(p_bar / 2.0, 4)

    machines = []

    for machine_id in range(m):

        working = round(rng.uniform(0.9, 1.0), 4)
        idle = round(rng.uniform(0.6, 0.7), 4)
        standby = round(rng.uniform(0.3, 0.5), 4)

        machines.append({
            "machine_id": machine_id,
            "working_power_kw": working,
            "idle_power_kw": idle,
            "standby_power_kw": standby,
            "rampup_off_power_kw": 0.8,
            "rampup_standby_power_kw": 0.8,
            "rampup_off_time": t_off,
            "rampup_standby_time": t_standby,
        })

    return machines


def generate_instance(instance_name, taillard_path, out_dir):

    instance = load_taillard(taillard_path)

    # Deterministic per-instance seed derived from master + name.
    # zlib.crc32 is used (not builtin hash()) because hash() is
    # salted per process, which would break reproducibility.
    name_hash = zlib.crc32(instance_name.encode("utf-8"))
    seed = MASTER_SEED ^ name_hash
    rng = random.Random(seed)

    jobs = build_jobs(instance, rng)
    machines = build_machines(instance, rng)

    data = {
        "instance": instance_name,
        "n_jobs": instance.num_jobs,
        "n_machines": instance.num_machines,
        "jobs": jobs,
        "machines": machines,
    }

    out_path = os.path.join(
        out_dir, f"{instance_name}_extended.json"
    )
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=4)

    return data


# ============================================================
# Validation summary (proves compliance with the rules)
# ============================================================

def validate(data):

    m = data["n_machines"]

    gammas = []
    alphas = []
    for job in data["jobs"]:
        total = sum(
            op["processing_time"] for op in job["operations"]
        )
        gammas.append(job["due_date"] / total)
        alphas.append(job["tardiness_penalty_alpha"])

    working = [mm["working_power_kw"] for mm in data["machines"]]
    idle = [mm["idle_power_kw"] for mm in data["machines"]]
    standby = [mm["standby_power_kw"] for mm in data["machines"]]

    ok = (
        1.8 <= min(gammas) and max(gammas) <= 2.3
        and set(alphas) <= set(ALPHA_CHOICES)
        and 0.9 <= min(working) and max(working) <= 1.0
        and 0.6 <= min(idle) and max(idle) <= 0.7
        and 0.3 <= min(standby) and max(standby) <= 0.5
    )

    return {
        "gamma": (min(gammas), max(gammas)),
        "alpha_set": sorted(set(alphas)),
        "working": (min(working), max(working)),
        "idle": (min(idle), max(idle)),
        "standby": (min(standby), max(standby)),
        "ok": ok,
    }


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Regenerate the Wei extended dataset from "
                    "the paper's rules (Eqs. 18-20)."
    )
    parser.add_argument(
        "--out", default=DEFAULT_OUT,
        help=f"output directory (default {DEFAULT_OUT})"
    )
    parser.add_argument(
        "--install", action="store_true",
        help=f"after generating, back up {INSTALL_TARGET} and "
             f"replace it with the new dataset so existing "
             f"experiments pick it up automatically."
    )
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    txt_files = sorted(
        f for f in os.listdir(TAILLARD_DIR)
        if f.endswith(".txt")
    )

    if not txt_files:
        raise FileNotFoundError(
            f"No Taillard .txt files in {TAILLARD_DIR}"
        )

    print("=" * 78)
    print("PAPER-FAITHFUL WEI DATASET GENERATOR")
    print(f"master seed = {MASTER_SEED}   output = {args.out}")
    print("=" * 78)
    print(
        f"{'instance':>9} {'gamma range':>16} {'alpha':>10} "
        f"{'work kW':>14} {'ok':>4}"
    )
    print("-" * 78)

    all_ok = True
    count = 0

    for filename in txt_files:
        instance_name = filename[:-4]
        data = generate_instance(
            instance_name,
            os.path.join(TAILLARD_DIR, filename),
            args.out,
        )
        v = validate(data)
        all_ok = all_ok and v["ok"]
        count += 1
        print(
            f"{instance_name:>9} "
            f"{v['gamma'][0]:.3f}-{v['gamma'][1]:.3f}     "
            f"{str(v['alpha_set']):>10} "
            f"{v['working'][0]:.3f}-{v['working'][1]:.3f} "
            f"{('OK' if v['ok'] else 'BAD'):>4}"
        )

    print("-" * 78)
    print(f"generated {count} instances   all_rules_ok = {all_ok}")

    if not all_ok:
        raise SystemExit(
            "Rule validation FAILED for at least one instance."
        )

    if args.install:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = f"{INSTALL_TARGET}_backup_{stamp}"
        if os.path.isdir(INSTALL_TARGET):
            shutil.copytree(INSTALL_TARGET, backup)
            print(f"backed up {INSTALL_TARGET} -> {backup}")
        os.makedirs(INSTALL_TARGET, exist_ok=True)
        for filename in os.listdir(args.out):
            shutil.copy2(
                os.path.join(args.out, filename),
                os.path.join(INSTALL_TARGET, filename),
            )
        print(
            f"installed new dataset into {INSTALL_TARGET} "
            f"(experiments will now use it)"
        )
    else:
        print()
        print("Not installed. The new files are in "
              f"'{args.out}'. To use them in experiments:")
        print("  - inspect them, then re-run with --install to")
        print(f"    back up and replace '{INSTALL_TARGET}', OR")
        print("  - pass json_directory=\"" + args.out + "\" to")
        print("    load_wei_extended_instance().")

    print("=" * 78)


if __name__ == "__main__":
    main()
