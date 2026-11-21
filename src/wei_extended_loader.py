import json

from benchmark import load_taillard
from energy import MachineEnergy


def load_wei_extended_instance(
        instance_name,
        json_directory="data/wei_extended_taillard",
        taillard_directory="data/taillard/full"
):
    """
    Load one paper-faithful extended Wei/Taillard instance
    (see generate_wei_dataset.py for how json_directory's files
    are generated from Wei et al. 2021, Section V.A, Eqs. 18-20).

    Original scheduling routes and processing times are loaded
    through our verified Taillard loader.

    Due dates, alpha/beta penalties, and energy parameters are
    loaded directly from the generated JSON file.

    Nothing is regenerated randomly here.
    """

    # ========================================================
    # Original Taillard instance
    # ========================================================

    taillard_path = (
        f"{taillard_directory}/"
        f"{instance_name}.txt"
    )

    instance = load_taillard(
        taillard_path
    )

    # ========================================================
    # Extended JSON
    # ========================================================

    json_path = (
        f"{json_directory}/"
        f"{instance_name}_extended.json"
    )

    with open(
        json_path,
        "r",
        encoding="utf-8"
    ) as handle:

        data = json.load(
            handle
        )

    # ========================================================
    # Validation
    # ========================================================

    if data["instance"] != instance_name:

        raise ValueError(
            f"JSON instance mismatch: "
            f"expected {instance_name}, "
            f"found {data['instance']}"
        )

    if data["n_jobs"] != instance.num_jobs:

        raise ValueError(
            "Job-count mismatch between "
            "Taillard and extended dataset."
        )

    if data["n_machines"] != instance.num_machines:

        raise ValueError(
            "Machine-count mismatch between "
            "Taillard and extended dataset."
        )

    # ========================================================
    # Due Dates
    # ========================================================

    due_dates = {}

    # ========================================================
    # Alpha / Beta
    # ========================================================

    weights = {}

    for job in data["jobs"]:

        job_id = int(
            job["job_id"]
        )

        due_dates[job_id] = float(
            job["due_date"]
        )

        weights[job_id] = {
            "alpha": float(
                job[
                    "tardiness_penalty_alpha"
                ]
            ),

            "beta": float(
                job[
                    "earliness_penalty_beta"
                ]
            )
        }

    # ========================================================
    # Machine Energy Parameters
    # ========================================================

    machines = []

    machine_records = sorted(
        data["machines"],
        key=lambda x: x["machine_id"]
    )

    for machine in machine_records:

        machines.append(

            MachineEnergy(

                processing_power=float(
                    machine[
                        "working_power_kw"
                    ]
                ),

                idle_power=float(
                    machine[
                        "idle_power_kw"
                    ]
                ),

                standby_power=float(
                    machine[
                        "standby_power_kw"
                    ]
                ),

                ramp_standby_power=float(
                    machine[
                        "rampup_standby_power_kw"
                    ]
                ),

                ramp_standby_time=float(
                    machine[
                        "rampup_standby_time"
                    ]
                ),

                ramp_off_power=float(
                    machine[
                        "rampup_off_power_kw"
                    ]
                ),

                ramp_off_time=float(
                    machine[
                        "rampup_off_time"
                    ]
                )

            )

        )

    return (
        instance,
        due_dates,
        weights,
        machines
    )
