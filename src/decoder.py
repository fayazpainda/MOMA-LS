"""
Wei2021 Reproduction

Classical JSSP Schedule Decoder

Operation-based chromosome decoding.

For every appearance of job i in the OS chromosome:

    1st appearance -> operation O_i1
    2nd appearance -> operation O_i2
    ...
    k-th appearance -> operation O_ik

The machine assigned to each operation is fixed by
the Taillard benchmark.

For operation O_ik processed on machine j:

    S_ik =
        max(
            completion time of previous operation of job i,
            availability time of machine j
        )

    C_ik =
        S_ik + pt_ik

This guarantees:

1. technological/job precedence;
2. one operation at a time on each machine;
3. non-preemptive processing.

The parameter require_complete is used because the
NEH heuristic evaluates partial job sequences during
its constructive insertion procedure.

Normal optimization decoding:
    require_complete = True

NEH partial decoding:
    require_complete = False
"""


from dataclasses import dataclass



# ============================================================
# Scheduled Operation
# ============================================================

@dataclass
class ScheduledOperation:

    job_id: int

    operation_id: int

    machine: int

    start: float

    finish: float



# ============================================================
# Complete Schedule
# ============================================================

@dataclass
class Schedule:

    operations: list

    makespan: float



# ============================================================
# Decode Operation-Based Chromosome
# ============================================================

def decode(
        instance,
        chromosome,
        require_complete=True
):

    """
    Decode an operation-based JSP chromosome.

    Parameters
    ----------
    instance:
        JSP benchmark instance.

    chromosome:
        Operation-based chromosome.

    require_complete:
        True:
            every job must appear exactly the required
            number of times.

        False:
            partial chromosomes are permitted.
            This is required during NEH construction.

    Returns
    -------
    Schedule
    """


    # ========================================================
    # Machine Availability Times
    # ========================================================

    machine_time = [

        0.0

        for _ in range(
            instance.num_machines
        )

    ]



    # ========================================================
    # Job Availability Times
    # ========================================================

    job_time = [

        0.0

        for _ in range(
            instance.num_jobs
        )

    ]



    # ========================================================
    # Next Operation Index for Each Job
    # ========================================================

    job_operation_index = [

        0

        for _ in range(
            instance.num_jobs
        )

    ]



    scheduled_operations = []



    # ========================================================
    # Decode OS From Left to Right
    # ========================================================

    for job_id in chromosome.operation_sequence:


        # ----------------------------------------------------
        # Validate Job ID
        # ----------------------------------------------------

        if (

            job_id < 0

            or

            job_id >= instance.num_jobs

        ):

            raise ValueError(

                f"Invalid job ID in chromosome: {job_id}"

            )



        # ----------------------------------------------------
        # k-th occurrence of job i corresponds
        # to k-th operation of that job.
        # ----------------------------------------------------

        operation_index = (

            job_operation_index[
                job_id
            ]

        )



        # ----------------------------------------------------
        # Job appears too many times
        # ----------------------------------------------------

        if operation_index >= len(

            instance.jobs[
                job_id
            ].operations

        ):

            raise ValueError(

                f"Job {job_id} appears too many times "
                "in the operation sequence."

            )



        # ====================================================
        # Get Operation Data
        # ====================================================

        operation = (

            instance
            .jobs[
                job_id
            ]
            .operations[
                operation_index
            ]

        )



        machine_id = (

            operation.machine

        )



        processing_time = (

            operation.processing_time

        )



        # ----------------------------------------------------
        # Validate Machine ID
        # ----------------------------------------------------

        if (

            machine_id < 0

            or

            machine_id >= instance.num_machines

        ):

            raise ValueError(

                f"Invalid machine ID {machine_id} "
                f"for job {job_id}, "
                f"operation {operation_index}."

            )



        # ----------------------------------------------------
        # Validate Processing Time
        # ----------------------------------------------------

        if processing_time <= 0:

            raise ValueError(

                f"Invalid processing time "
                f"{processing_time} for job {job_id}, "
                f"operation {operation_index}."

            )



        # ====================================================
        # Earliest Feasible Start Time
        #
        # Job precedence:
        # previous operation of this job must finish.
        #
        # Machine capacity:
        # machine must be free.
        # ====================================================

        start_time = max(

            job_time[
                job_id
            ],

            machine_time[
                machine_id
            ]

        )



        # ====================================================
        # Completion Time
        # ====================================================

        finish_time = (

            start_time

            +

            processing_time

        )



        # ====================================================
        # Store Scheduled Operation
        # ====================================================

        scheduled_operations.append(

            ScheduledOperation(

                job_id=
                    job_id,

                operation_id=
                    operation_index,

                machine=
                    machine_id,

                start=
                    start_time,

                finish=
                    finish_time

            )

        )



        # ====================================================
        # Update Job Availability
        # ====================================================

        job_time[
            job_id
        ] = (

            finish_time

        )



        # ====================================================
        # Update Machine Availability
        # ====================================================

        machine_time[
            machine_id
        ] = (

            finish_time

        )



        # ====================================================
        # Advance to Next Operation of This Job
        # ====================================================

        job_operation_index[
            job_id
        ] += 1



    # ========================================================
    # Validate Complete Chromosome
    #
    # Normal optimization requires all jobs to contain
    # exactly the expected number of operations.
    #
    # NEH temporarily uses partial sequences, so this
    # validation is disabled when require_complete=False.
    # ========================================================

    if require_complete:


        for job in instance.jobs:


            expected_operations = len(

                job.operations

            )


            actual_operations = (

                job_operation_index[
                    job.job_id
                ]

            )


            if actual_operations != expected_operations:

                raise ValueError(

                    f"Job {job.job_id} has "
                    f"{actual_operations} decoded operations; "
                    f"expected {expected_operations}."

                )



    # ========================================================
    # Makespan
    # ========================================================

    if machine_time:

        makespan = max(

            machine_time

        )

    else:

        makespan = 0.0



    # ========================================================
    # Return Schedule
    # ========================================================

    return Schedule(

        operations=
            scheduled_operations,

        makespan=
            makespan

    )


# ============================================================
# Energy-Aware Right-Shift Decode (MEMETIC_DESIGN.md 3b / 4.3a)
# ============================================================

def decode_right_shift(
        instance,
        chromosome,
        require_complete=True,
        base_schedule=None
):

    """
    Semi-active left decode, then push every operation AS LATE AS POSSIBLE
    within its slack, subject to three constraints:

        (i)   its job-successor's (new) start,
        (ii)  its machine-successor's (new) start,
        (iii) its own job's completion time C_i must not increase
              (the job's LAST operation is pinned at its left-decode finish).

    Because every job's completion is pinned, this refinement leaves
    **Cmax and TWET exactly unchanged** and only redistributes each machine's
    idle time.

    IMPORTANT -- NEC is NOT monotonically improved by this transform alone.
    Pushing each operation to its latest independent start can *fragment* a
    machine's idle (opening a gap both before and after an intermediate pinned
    operation) as easily as it can merge it, so the right-shifted NEC may be
    higher OR lower than the semi-active NEC for a given solution. The Pareto-safe
    "decrease or no-op" guarantee of MEMETIC_DESIGN.md §4.3a is therefore enforced
    one level up, in `evaluation.evaluate_individual(right_shift=True)`, which
    keeps the right-shifted NEC only when it does not exceed the semi-active NEC.
    This function is the raw transform; the min() safeguard lives in evaluation.

    It changes start times, never the operation sequence, so it does NOT affect
    the critical-path / block structure used by the local-search neighborhoods.

    Parameters
    ----------
    base_schedule:
        Optional precomputed semi-active `decode` of the same chromosome. When
        given it is used as the starting point (its `ScheduledOperation` records
        are mutated in place); otherwise a fresh `decode` is performed. Pass this
        to avoid decoding twice when the caller already holds the left schedule.

    Returns a `Schedule` with the same makespan as `decode` but right-shifted
    start/finish times.
    """

    # Left (semi-active) schedule first -- fixes the sequence and every C_i.
    if base_schedule is None:
        schedule = decode(
            instance, chromosome, require_complete=require_complete
        )
    else:
        schedule = base_schedule

    operations = schedule.operations

    makespan = schedule.makespan

    # --- successor links -------------------------------------------------
    # Job-successor: the next operation of the same job (operation_id + 1).
    # Machine-successor: the next operation on the same machine by start time.

    # Index by (job_id, operation_id) for job-successor lookup.
    by_job_op = {
        (op.job_id, op.operation_id): op
        for op in operations
    }

    # Per-machine chronological order for machine-successor lookup.
    machine_order = {}
    for op in operations:
        machine_order.setdefault(op.machine, []).append(op)
    for machine_id in machine_order:
        machine_order[machine_id].sort(key=lambda o: o.start)

    machine_successor = {}
    for machine_id, ops in machine_order.items():
        for index in range(len(ops) - 1):
            machine_successor[id(ops[index])] = ops[index + 1]

    # --- push right in reverse start order -------------------------------
    # Processing from latest start to earliest guarantees that when an
    # operation is shifted, both of its successors already carry their final
    # (already right-shifted) start times.

    for op in sorted(operations, key=lambda o: o.start, reverse=True):

        processing_time = op.finish - op.start

        # (iii) job completion constraint: the job's last operation is pinned.
        job_successor = by_job_op.get((op.job_id, op.operation_id + 1))

        if job_successor is None:
            # Last operation of the job -> pin its finish (keep C_i fixed).
            latest_finish = op.finish
        else:
            latest_finish = job_successor.start

        # (ii) machine capacity: cannot overlap the next op on the machine.
        mach_successor = machine_successor.get(id(op))

        if mach_successor is not None:
            latest_finish = min(latest_finish, mach_successor.start)

        # A right shift can never move an operation earlier than its
        # semi-active start; guard against numerical drift.
        new_finish = max(op.finish, latest_finish)

        op.start = new_finish - processing_time

        op.finish = new_finish

    return Schedule(
        operations=operations,
        makespan=makespan,
    )