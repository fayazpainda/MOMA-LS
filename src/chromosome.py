"""
Wei2021 Reproduction

Chromosome Representation

Wei uses operation-based encoding for the classical
Job Shop Scheduling Problem.

OS:
    Operation Sequence chromosome.

Each job ID appears once for every operation belonging
to that job.

Example for 3 jobs x 3 machines:

    [0, 0, 0,
     1, 1, 1,
     2, 2, 2]

After random permutation:

    [0, 1, 0, 2, 1, 2, 0, 1, 2]

The k-th appearance of job i represents the k-th
operation of that job.

Machine assignments are NOT decision variables in the
classical JSSP. They are fixed by the Taillard instance.

The machine_sequence field is retained only for
compatibility with the existing decoder.
"""


import random

from dataclasses import dataclass



# ============================================================
# Chromosome
# ============================================================

@dataclass
class Chromosome:

    # Operation-based chromosome
    operation_sequence: list

    # Fixed machine route from benchmark data.
    # This is NOT a machine-selection decision variable.
    machine_sequence: list



# ============================================================
# Random Operation Sequence
# ============================================================

def create_operation_sequence(
        instance
):

    """
    Create a valid random operation-based chromosome.

    For every job i:

        job i appears n_i times,

    where n_i is the number of operations of job i.

    Random permutation preserves the required number
    of occurrences of every job.
    """


    sequence = []


    for job in instance.jobs:


        number_of_operations = len(

            job.operations

        )


        sequence.extend(

            [job.job_id]

            *

            number_of_operations

        )



    random.shuffle(

        sequence

    )


    return sequence



# ============================================================
# Fixed Machine Route
# ============================================================

def create_machine_sequence(
        instance
):

    """
    Store the fixed machine route from the JSP instance.

    IMPORTANT:
    This is not a machine-selection chromosome.

    In classical JSSP, every operation already has
    a predetermined machine.
    """


    machine_sequence = []


    for job in instance.jobs:


        for operation in job.operations:


            machine_sequence.append(

                operation.machine

            )


    return machine_sequence



# ============================================================
# Generate Random Chromosome
# ============================================================

def generate_chromosome(
        instance
):

    """
    Generate one valid random JSP chromosome.
    """


    operation_sequence = create_operation_sequence(

        instance

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



# ============================================================
# Module Test
# ============================================================

if __name__ == "__main__":

    print(
        "Chromosome module loaded"
    )