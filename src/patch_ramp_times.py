"""
Wei2021 Reproduction

Surgical ramp-time patch for data/wei_extended_taillard.

Fixes ONLY the NEC bug: rewrites each machine's rampup_off_time
and rampup_standby_time to the SIZE-INDEPENDENT Eq. 20 reading

    t_rampup,off,j     = p_bar          (mean operation time)
    t_rampup,standby,j = p_bar / 2

while leaving due dates, alpha/beta, and all power values exactly
as they are. This preserves the current (already good) Cmax/TWET
reproduction and only corrects the ramp times that were inflating
NEC on large instances.

The original data/wei_extended_taillard is backed up first.

Validated by test_ramp_fix.py: this reading holds NEC at ~1.0x
Wei across ta11/ta31/ta61 (0.89 / 1.07 / 1.16), versus the old
per-machine-load reading (1.07 / 1.60 / 1.92).

Run from PROJECT ROOT:

    ./venv/bin/python src/patch_ramp_times.py
"""

import os
import json
import shutil
from datetime import datetime


DATA_DIR = "data/wei_extended_taillard"


def mean_op_time(data):
    times = [
        op["processing_time"]
        for job in data["jobs"]
        for op in job["operations"]
    ]
    return sum(times) / len(times)


def main():

    if not os.path.isdir(DATA_DIR):
        raise FileNotFoundError(f"{DATA_DIR} not found")

    # One-time backup.
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = f"{DATA_DIR}_backup_oldramp_{stamp}"
    shutil.copytree(DATA_DIR, backup)
    print(f"backed up {DATA_DIR} -> {backup}")
    print("=" * 74)

    files = sorted(
        f for f in os.listdir(DATA_DIR)
        if f.endswith("_extended.json")
    )

    print(f"{'file':>26} {'p_bar':>8} {'old off (avg)':>14} "
          f"{'new off':>9} {'new sb':>8}")
    print("-" * 74)

    for filename in files:
        path = os.path.join(DATA_DIR, filename)
        with open(path, encoding="utf-8") as h:
            data = json.load(h)

        p_bar = mean_op_time(data)
        t_off = round(p_bar, 4)
        t_standby = round(p_bar / 2.0, 4)

        old_off_avg = sum(
            m["rampup_off_time"] for m in data["machines"]
        ) / len(data["machines"])

        for m in data["machines"]:
            m["rampup_off_time"] = t_off
            m["rampup_standby_time"] = t_standby

        with open(path, "w", encoding="utf-8") as h:
            json.dump(data, h, indent=4)

        print(f"{filename:>26} {p_bar:>8.2f} {old_off_avg:>14.2f} "
              f"{t_off:>9.2f} {t_standby:>8.2f}")

    print("-" * 74)
    print(f"patched {len(files)} files in {DATA_DIR}")
    print("Due dates, alpha/beta, and powers are UNCHANGED.")
    print("=" * 74)


if __name__ == "__main__":
    main()
