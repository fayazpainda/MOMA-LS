"""
Wei et al. (2021) Reproduction
Objective Functions  (TWET BUG FIX)

Implementation of the objective equations reported by
Wei et al., "Unified Multi-Objective Genetic Algorithm for
Energy Efficient Job Shop Scheduling."

Objectives:

    Eq. (9)
        Cmax = max(C_i,k)

    Eq. (10)
        TWET = sum_i(
            alpha_i * T_i
            +
            beta_i * E_i
        )

    Eq. (16) (as printed)
        T_i = max(Cmax - d_i, 0)

    Eq. (17) (as printed)
        E_i = max(d_i - Cmax, 0)

------------------------------------------------------------
WHY THIS VERSION DIFFERS FROM THE PRINTED EQUATIONS
------------------------------------------------------------
The previous implementation (see objectives_before_twet_fix.py
and the original objectives.py) took Eq. (16)-(17) literally and
used the GLOBAL makespan Cmax for every job's tardiness/earliness.

That reading is provably inconsistent with Wei et al.'s own
reported results. In their Table 8 (instance 15x15-1):

    Scenario  Cmax    TWET
    S1        1486    352.96
    S2        1499    736.16

Cmax differs by <1% between S1 and S2, but TWET differs by
~2.1x. Under a Cmax-only formula, TWET would be an affine
function of Cmax alone, so two schedules with nearly identical
Cmax would necessarily have nearly identical TWET. They do not.
This proves the printed equations contain a typesetting error
(almost certainly "C_i,k" mis-rendered as "Cmax"), and that the
correct, intended formulas are:

    T_i = max(C_i - d_i, 0)
    E_i = max(d_i - C_i, 0)

where C_i is the completion time of job i's OWN last operation
(i.e. the finish time of operation O_i,n_i), not the shared
schedule-wide makespan.

This file implements the corrected, per-job formulation.
The makespan objective (Eq. 9) and NEC objective are unaffected
by this change and are not touched here.
"""


# ============================================================
# OBJECTIVE 1
# Makespan
# Wei Eq. (9)
# ============================================================

def calculate_makespan(schedule):
    """
    Calculate the makespan Cmax.

    Wei Eq. (9):

        Cmax = max(C_i,k)

    where C_i,k is the completion time of operation k
    of job i.

    The decoder already calculates and stores this value as:

        schedule.makespan
    """

    if schedule is None:
        raise ValueError(
            "schedule cannot be None."
        )

    return float(
        schedule.makespan
    )


# ============================================================
# JOB COMPLETION TIMES
# Required for the corrected TWET formula
# ============================================================

def job_completion_times(schedule):
    """
    Compute each job's own completion time C_i.

    C_i is defined as the finish time of that job's LAST
    scheduled operation (i.e. the largest `finish` value among
    all ScheduledOperation records with matching job_id).

    This is NOT the same as the global makespan Cmax, except
    for whichever job happens to determine Cmax.

    Parameters
    ----------
    schedule : Schedule
        Decoded schedule containing a list of
        ScheduledOperation records (job_id, finish, ...).

    Returns
    -------
    completions : dict
        {job_id: C_i}
    """

    if schedule is None:
        raise ValueError(
            "schedule cannot be None."
        )

    if not hasattr(schedule, "operations"):
        raise AttributeError(
            "schedule must have an 'operations' attribute."
        )

    completions = {}

    for op in schedule.operations:

        job_id = op.job_id
        finish = op.finish

        if (
            job_id not in completions
            or finish > completions[job_id]
        ):
            completions[job_id] = finish

    return completions


# ============================================================
# EXTENDED BENCHMARK PARAMETERS
# ============================================================

def extract_job_parameters(extended_data):
    """
    Extract the job parameters required by Wei's TWET model.

    Expected Wei-style structure:

        {
            "jobs": [
                {
                    "job_id": 0,
                    "due_date": ...,
                    "tardiness_penalty_alpha": ...,
                    "earliness_penalty_beta": ...
                },
                ...
            ]
        }

    Returns
    -------
    due_dates : dict
        {
            job_id: due_date
        }

    weights : dict
        {
            job_id: {
                "alpha": alpha_i,
                "beta": beta_i
            }
        }

    Notes
    -----
    The JSON field names used by the current Wei extended dataset
    are:

        tardiness_penalty_alpha
        earliness_penalty_beta

    No alternative "weight" interpretation is used here because
    this function is intended for the strict Wei reproduction.
    """

    if not isinstance(
        extended_data,
        dict
    ):
        raise TypeError(
            "extended_data must be a dictionary."
        )

    if "jobs" not in extended_data:
        raise KeyError(
            "Extended data does not contain 'jobs'."
        )

    if not isinstance(
        extended_data["jobs"],
        list
    ):
        raise TypeError(
            "extended_data['jobs'] must be a list."
        )

    due_dates = {}
    weights = {}

    for job in extended_data["jobs"]:

        if not isinstance(
            job,
            dict
        ):
            raise TypeError(
                "Each job record must be a dictionary."
            )

        # ----------------------------------------------------
        # Job ID
        # ----------------------------------------------------

        if "job_id" not in job:
            raise KeyError(
                "Job record is missing 'job_id'."
            )

        job_id = int(
            job["job_id"]
        )

        # ----------------------------------------------------
        # Due date
        # ----------------------------------------------------

        if "due_date" not in job:
            raise KeyError(
                f"Job {job_id} is missing 'due_date'."
            )

        due_date = float(
            job["due_date"]
        )

        # ----------------------------------------------------
        # Wei alpha
        # ----------------------------------------------------

        if (
            "tardiness_penalty_alpha"
            not in job
        ):
            raise KeyError(
                f"Job {job_id} is missing "
                "'tardiness_penalty_alpha'."
            )

        alpha = float(
            job[
                "tardiness_penalty_alpha"
            ]
        )

        # ----------------------------------------------------
        # Wei beta
        # ----------------------------------------------------

        if (
            "earliness_penalty_beta"
            not in job
        ):
            raise KeyError(
                f"Job {job_id} is missing "
                "'earliness_penalty_beta'."
            )

        beta = float(
            job[
                "earliness_penalty_beta"
            ]
        )

        # ----------------------------------------------------
        # Basic validation
        # ----------------------------------------------------

        if alpha <= 0:
            raise ValueError(
                f"Job {job_id}: alpha must be > 0, "
                f"got {alpha}."
            )

        if beta < 0:
            raise ValueError(
                f"Job {job_id}: beta must be >= 0, "
                f"got {beta}."
            )

        due_dates[job_id] = due_date

        weights[job_id] = {
            "alpha": alpha,
            "beta": beta
        }

    # --------------------------------------------------------
    # Validate unique job IDs
    # --------------------------------------------------------

    if len(due_dates) != len(
        extended_data["jobs"]
    ):
        raise ValueError(
            "Duplicate job_id detected in extended data."
        )

    return (
        due_dates,
        weights
    )


# ============================================================
# OBJECTIVE 2
# TWET  (FIXED: per-job completion time C_i)
# ============================================================

def calculate_twet(
        schedule,
        due_dates,
        weights,
        completion_mode="per_job"
):
    """
    Calculate TWET (Wei Eq. 10 with Eq. 16/17).

        TWET = sum_i( alpha_i * T_i + beta_i * E_i )

    completion_mode controls which completion time is used
    for T_i and E_i:

    "per_job" (DEFAULT — corrected formulation):
        Uses each job's own completion time C_i (the finish
        time of that job's LAST operation):

            T_i = max(C_i - d_i, 0)
            E_i = max(d_i - C_i, 0)

        This is the intended, internally consistent objective
        and is the ONLY mode used by the optimization pipeline
        (see evaluation.evaluate_individual). It is the default
        so that every caller — including the smoke test — runs
        the exact formula the optimizer uses. See the module
        docstring for why the literal printed equations are a
        typesetting error.

    "makespan" (strict Wei paper — diagnostic only):
        Literal Eq. (16)/(17) as printed:

            T_i = max(Cmax - d_i, 0)
            E_i = max(d_i - Cmax, 0)

        where Cmax is the schedule-wide makespan (identical
        for every job).

        WARNING: under this literal reading, TWET becomes a
        pure function of Cmax alone (an affine transform of a
        single scalar), so it carries almost no information
        independent of the makespan objective. Retained only
        for the Table 8 internal-consistency check; do NOT use
        it as an optimization objective.
    """

    if schedule is None:
        raise ValueError(
            "schedule cannot be None."
        )

    if not isinstance(
        due_dates,
        dict
    ):
        raise TypeError(
            "due_dates must be a dictionary."
        )

    if not isinstance(
        weights,
        dict
    ):
        raise TypeError(
            "weights must be a dictionary."
        )

    if completion_mode not in ("makespan", "per_job"):
        raise ValueError(
            "completion_mode must be 'makespan' "
            "(strict Wei paper) or 'per_job'."
        )

    # --------------------------------------------------------
    # Completion time per job.
    #
    # "makespan": every job uses the global Cmax (strict Wei
    #             Eq. 16/17 as printed).
    # "per_job" : each job uses its own last-operation finish.
    # --------------------------------------------------------

    if completion_mode == "makespan":
        cmax = float(schedule.makespan)
        completions = {
            job_id: cmax
            for job_id in due_dates.keys()
        }
    else:
        completions = job_completion_times(
            schedule
        )

    total_twet = 0.0

    # --------------------------------------------------------
    # Calculate Eq. (10) with corrected Eq. (16)/(17)
    # --------------------------------------------------------

    for job_id in sorted(
        due_dates.keys()
    ):

        # ----------------------------------------------------
        # Verify weights exist
        # ----------------------------------------------------

        if job_id not in weights:
            raise KeyError(
                f"Missing alpha/beta weights "
                f"for job {job_id}."
            )

        if job_id not in completions:
            raise KeyError(
                f"Schedule does not contain a completion "
                f"time for job {job_id}."
            )

        # ----------------------------------------------------
        # Parameters
        # ----------------------------------------------------

        d_i = float(
            due_dates[job_id]
        )

        alpha_i = float(
            weights[job_id]["alpha"]
        )

        beta_i = float(
            weights[job_id]["beta"]
        )

        C_i = float(
            completions[job_id]
        )

        # ----------------------------------------------------
        # Corrected Eq. (16)
        #
        # T_i = max(C_i - d_i, 0)
        # ----------------------------------------------------

        T_i = max(
            C_i - d_i,
            0.0
        )

        # ----------------------------------------------------
        # Corrected Eq. (17)
        #
        # E_i = max(d_i - C_i, 0)
        # ----------------------------------------------------

        E_i = max(
            d_i - C_i,
            0.0
        )

        # ----------------------------------------------------
        # Wei Eq. (10)
        #
        # alpha_i*T_i + beta_i*E_i
        # ----------------------------------------------------

        total_twet += (
            alpha_i * T_i
            +
            beta_i * E_i
        )

    return float(
        total_twet
    )


# ============================================================
# OPTIONAL DIAGNOSTIC
# ============================================================

def calculate_twet_breakdown(
        schedule,
        due_dates,
        weights
):
    """
    Diagnostic version of the corrected Eq. (10).

    Returns one record per job showing:

        job_id
        C_i           (that job's own completion time)
        due_date
        alpha
        beta
        tardiness
        earliness
        contribution

    This function does NOT change the optimization objective.
    It is only for verification/debugging.
    """

    completions = job_completion_times(
        schedule
    )

    breakdown = []

    total = 0.0

    for job_id in sorted(
        due_dates.keys()
    ):

        if job_id not in weights:
            raise KeyError(
                f"Missing weights for job {job_id}."
            )

        if job_id not in completions:
            raise KeyError(
                f"Schedule does not contain a completion "
                f"time for job {job_id}."
            )

        d_i = float(
            due_dates[job_id]
        )

        alpha_i = float(
            weights[job_id]["alpha"]
        )

        beta_i = float(
            weights[job_id]["beta"]
        )

        C_i = float(
            completions[job_id]
        )

        T_i = max(
            C_i - d_i,
            0.0
        )

        E_i = max(
            d_i - C_i,
            0.0
        )

        contribution = (
            alpha_i * T_i
            +
            beta_i * E_i
        )

        total += contribution

        breakdown.append({

            "job_id":
                job_id,

            "C_i":
                C_i,

            "due_date":
                d_i,

            "alpha":
                alpha_i,

            "beta":
                beta_i,

            "tardiness":
                T_i,

            "earliness":
                E_i,

            "contribution":
                contribution

        })

    return (
        breakdown,
        float(total)
    )
