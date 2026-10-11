"""
Wei2021 Reproduction

Genetic Operators

Wei two-point crossover:

1. Choose two crossover points randomly.
2. Exchange the genes between the two intersections.
3. Repair excessive / missing job occurrences.
4. Return TWO feasible offspring chromosomes.

Important
---------
Wei's published crossover example shows that the
exchanged segment itself is preserved during repair.

Example from the paper:

    P1 = 213112323
    P2 = 131233122

    crossover positions = 3 and 6

After segment exchange:

    211233323
    133112122

After repair:

    211233123
    133112322

Therefore repair is performed outside the exchanged
segment, beginning after the second crossover point.
"""


import random

from chromosome import Chromosome



# ============================================================
# Chromosome Repair
# ============================================================

def repair_operation_sequence(
        sequence,
        required_counts,
        crossover_start,
        crossover_end
):

    """
    Repair an operation-based chromosome after Wei's
    two-point segment exchange.

    The crossover segment:

        [crossover_start, crossover_end]

    is preserved.

    Excessive job occurrences outside that segment are
    replaced by missing jobs.

    Repair priority follows the behavior demonstrated
    in Wei's published example:

        1. positions after the exchanged segment;
        2. positions before the exchanged segment.
    """


    repaired = sequence.copy()



    # ========================================================
    # Current Occurrence Counts
    # ========================================================

    current_counts = {}


    for job in repaired:

        current_counts[job] = (

            current_counts.get(
                job,
                0
            )

            +

            1

        )



    # ========================================================
    # Missing Job Occurrences
    # ========================================================

    missing = []


    for job, required in required_counts.items():


        current = current_counts.get(

            job,

            0

        )


        shortage = (

            required

            -

            current

        )


        if shortage > 0:

            missing.extend(

                [job]

                *

                shortage

            )



    # ========================================================
    # Nothing to Repair
    # ========================================================

    if not missing:

        return repaired



    # ========================================================
    # Repair Search Order
    #
    # Wei example:
    #
    # first search after crossover segment,
    # then search before crossover segment.
    #
    # Do NOT alter exchanged segment.
    # ========================================================

    repair_positions = (

        list(

            range(

                crossover_end + 1,

                len(repaired)

            )

        )

        +

        list(

            range(

                0,

                crossover_start

            )

        )

    )



    missing_index = 0



    # ========================================================
    # Replace Excessive Occurrences
    # ========================================================

    for position in repair_positions:


        if missing_index >= len(missing):

            break



        job = repaired[position]


        required = required_counts.get(

            job,

            0

        )


        current = current_counts.get(

            job,

            0

        )



        # ----------------------------------------------------
        # Replace only an excessive occurrence
        # ----------------------------------------------------

        if current > required:


            replacement = missing[

                missing_index

            ]



            # Reduce excessive job count

            current_counts[job] -= 1



            # Replace gene

            repaired[position] = (

                replacement

            )



            # Increase replacement-job count

            current_counts[replacement] = (

                current_counts.get(

                    replacement,

                    0

                )

                +

                1

            )



            missing_index += 1



    # ========================================================
    # Repair Must Be Complete
    # ========================================================

    if missing_index != len(missing):

        raise ValueError(

            "Crossover repair could not restore "
            "all required job occurrences without "
            "altering the exchanged segment."

        )



    # ========================================================
    # Final Validation
    # ========================================================

    for job, required in required_counts.items():


        actual = repaired.count(

            job

        )


        if actual != required:

            raise ValueError(

                f"Crossover repair failed for job {job}: "
                f"expected {required}, got {actual}"

            )



    return repaired



# ============================================================
# Wei Two-Point Crossover
# ============================================================

def crossover_OS(
        parent1,
        parent2
):

    """
    Wei two-point crossover.

    Steps:

    1. Randomly choose two crossover positions.
    2. Exchange the segment between them.
    3. Preserve the exchanged segment.
    4. Repair excessive/missing occurrences outside it.
    5. Return two feasible offspring.
    """


    p1 = (

        parent1.operation_sequence

    )


    p2 = (

        parent2.operation_sequence

    )



    # ========================================================
    # Parent Compatibility
    # ========================================================

    if len(p1) != len(p2):

        raise ValueError(

            "Parents must have operation sequences "
            "of equal length."

        )



    size = len(p1)



    # ========================================================
    # Tiny Chromosome Safety
    # ========================================================

    if size < 2:


        child1 = Chromosome(

            operation_sequence=
                p1.copy(),

            machine_sequence=
                parent1.machine_sequence.copy()

        )


        child2 = Chromosome(

            operation_sequence=
                p2.copy(),

            machine_sequence=
                parent2.machine_sequence.copy()

        )


        return (

            child1,

            child2

        )



    # ========================================================
    # Step 1:
    # Choose Two Intersections Randomly
    # ========================================================

    point1, point2 = sorted(

        random.sample(

            range(size),

            2

        )

    )



    # ========================================================
    # Required Job Counts
    #
    # Operation-based encoding requires each job ID
    # to occur exactly once per operation of that job.
    # ========================================================

    required_counts = {}


    for job in p1:

        required_counts[job] = (

            required_counts.get(

                job,

                0

            )

            +

            1

        )



    # ========================================================
    # Step 2:
    # Exchange Segment
    #
    # Both crossover positions are included.
    # ========================================================

    child1_sequence = (

        p1.copy()

    )


    child2_sequence = (

        p2.copy()

    )



    child1_sequence[

        point1:
        point2 + 1

    ] = p2[

        point1:
        point2 + 1

    ]



    child2_sequence[

        point1:
        point2 + 1

    ] = p1[

        point1:
        point2 + 1

    ]



    # ========================================================
    # Step 3:
    # Repair Child 1
    # ========================================================

    child1_sequence = repair_operation_sequence(

        child1_sequence,

        required_counts,

        crossover_start=
            point1,

        crossover_end=
            point2

    )



    # ========================================================
    # Step 3:
    # Repair Child 2
    # ========================================================

    child2_sequence = repair_operation_sequence(

        child2_sequence,

        required_counts,

        crossover_start=
            point1,

        crossover_end=
            point2

    )



    # ========================================================
    # Step 4:
    # Create TWO Feasible Offspring
    # ========================================================

    child1 = Chromosome(

        operation_sequence=
            child1_sequence,

        machine_sequence=
            parent1.machine_sequence.copy()

    )


    child2 = Chromosome(

        operation_sequence=
            child2_sequence,

        machine_sequence=
            parent2.machine_sequence.copy()

    )



    return (

        child1,

        child2

    )