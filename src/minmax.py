"""
Wei2021 Reproduction

MinMax Initialization Component

IMPORTANT
---------
Wei et al. state that the MME initialization heuristic
combines MinMax (MM) and NEH.

However, the Wei paper does not provide the internal
MinMax pseudocode or enough information to reproduce
the exact MM procedure.

Therefore this file contains a conservative deterministic
MM-style seed only.

It MUST be reported as an implementation approximation
until the original MinMax/MME source procedure is recovered.
"""


from chromosome import Chromosome



# ============================================================
# Conservative MM-Style Seed
# ============================================================

def minmax_OS(instance):

    """
    Construct one deterministic valid operation-based
    chromosome.

    Current approximation:

        jobs are ordered by decreasing total
        processing requirement.

    IMPORTANT:
        This ordering is NOT claimed to be Wei's
        exact MinMax algorithm.
    """


    job_loads = []


    for job in instance.jobs:


        total_processing_time = sum(

            operation.processing_time

            for operation
            in job.operations

        )


        job_loads.append(

            (
                job.job_id,

                total_processing_time
            )

        )



    # --------------------------------------------------------
    # Largest processing requirement first
    # --------------------------------------------------------

    job_loads.sort(

        key=lambda item:
            (
                -item[1],
                item[0]
            )

    )



    operation_sequence = []


    for job_id, _ in job_loads:


        number_of_operations = len(

            instance.jobs[
                job_id
            ].operations

        )


        operation_sequence.extend(

            [
                job_id
            ]

            *

            number_of_operations

        )



    return operation_sequence



# ============================================================
# Create MM Chromosome
# ============================================================

def create_minmax_chromosome(
        instance
):


    operation_sequence = minmax_OS(

        instance

    )



    machine_sequence = []


    for job in instance.jobs:


        for operation in job.operations:


            machine_sequence.append(

                operation.machine

            )



    return Chromosome(

        operation_sequence=
            operation_sequence,

        machine_sequence=
            machine_sequence

    )