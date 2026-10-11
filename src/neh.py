"""
Wei2021 Reproduction

NEH Initialization Component

IMPORTANT
---------
Wei et al. state that MME combines:

    MinMax (MM)
    +
    Nawaz-Enscore-Ham (NEH)

However, the Wei paper does not give the exact
job-shop adaptation of NEH used inside MME.

Therefore this file implements a standard NEH-style
job insertion heuristic adapted to the operation-based
JSP encoding.

It MUST be reported as an implementation approximation,
not as Wei's exact original source code.
"""


from chromosome import Chromosome
from decoder import decode



# ============================================================
# Total Processing Requirement of Each Job
# ============================================================

def calculate_job_times(instance):

    job_times = []


    for job in instance.jobs:

        total_processing_time = sum(

            operation.processing_time

            for operation
            in job.operations

        )


        job_times.append(

            (
                job.job_id,
                total_processing_time
            )

        )


    return job_times



# ============================================================
# Convert Job Order to Operation-Based Chromosome
# ============================================================

def build_operation_sequence(
        instance,
        job_order
):

    """
    Convert a job-level NEH order into a valid
    operation-based JSP chromosome.

    Each job ID appears once for every operation
    belonging to that job.
    """


    sequence = []


    for job_id in job_order:


        number_of_operations = len(

            instance.jobs[
                job_id
            ].operations

        )


        sequence.extend(

            [job_id]

            *

            number_of_operations

        )


    return sequence



# ============================================================
# Fixed Machine Sequence
# ============================================================

def create_machine_sequence(instance):

    """
    Classical JSP has a fixed machine route.
    """


    machine_sequence = []


    for job in instance.jobs:


        for operation in job.operations:


            machine_sequence.append(

                operation.machine

            )


    return machine_sequence



# ============================================================
# Evaluate One Candidate Job Order
# ============================================================

def evaluate_job_order(
        instance,
        job_order
):

    """
    Evaluate a candidate NEH job order by makespan.
    """


    operation_sequence = build_operation_sequence(

        instance,

        job_order

    )


    machine_sequence = create_machine_sequence(

        instance

    )


    chromosome = Chromosome(

        operation_sequence=
            operation_sequence,

        machine_sequence=
            machine_sequence

    )


    schedule = decode(

        instance,

        chromosome,

        require_complete=False

    )


    return schedule.makespan



# ============================================================
# NEH-Style Insertion Procedure
# ============================================================

def neh_job_order(instance):

    """
    Standard NEH-style procedure:

    1. Sort jobs by decreasing total processing time.
    2. Start with the first job.
    3. Insert each remaining job into every possible position.
    4. Keep the position producing the smallest makespan.
    """


    job_times = calculate_job_times(

        instance

    )


    # --------------------------------------------------------
    # Initial NEH ordering:
    # decreasing total processing requirement
    # --------------------------------------------------------

    job_times.sort(

        key=lambda item:
            (
                -item[1],
                item[0]
            )

    )


    sorted_jobs = [

        job_id

        for job_id, _
        in job_times

    ]


    if not sorted_jobs:

        return []



    current_order = [

        sorted_jobs[0]

    ]



    # ========================================================
    # Insert Remaining Jobs
    # ========================================================

    for job_id in sorted_jobs[1:]:


        best_order = None

        best_makespan = None



        for position in range(

            len(current_order) + 1

        ):


            candidate_order = (

                current_order[:position]

                +

                [job_id]

                +

                current_order[position:]

            )


            makespan = evaluate_job_order(

                instance,

                candidate_order

            )



            if (

                best_makespan is None

                or

                makespan < best_makespan

            ):


                best_makespan = makespan

                best_order = candidate_order



        current_order = best_order



    return current_order



# ============================================================
# Create NEH Chromosome
# ============================================================

def create_NEH_chromosome(instance):

    job_order = neh_job_order(

        instance

    )


    operation_sequence = build_operation_sequence(

        instance,

        job_order

    )


    machine_sequence = create_machine_sequence(

        instance

    )


    return Chromosome(

        operation_sequence=
            operation_sequence,

        machine_sequence=
            machine_sequence

    )