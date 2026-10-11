"""
Wei2021 Reproduction

Mutation Operator

Wei et al., Section IV-E, describes a per-position random matrix:

    "A set of matrices composed of uniformly distributed
     numbers between 0 and 1 with the same dimension as the
     parent population is generated. When a specific value in
     the random number matrix is less than the mutation
     probability, then the corresponding position in the
     parent population matrix needs to be mutated.

     The steps of mutation are:
     (1) Move the gene from this position to the last position
         in this chromosome.
     (2) Move all the genes which are behind this position
         forward by one position."

Read literally, every gene position is an independent Bernoulli(Pm) trial, so
on average L * Pm positions are moved to the tail per offspring (implemented
below as `move_to_end_mutation_literal`).

DEVIATION FROM THE LITERAL TEXT (default operator, `move_to_end_mutation`)
---------------------------------------------------------------------------
The literal per-position reading empirically relocates ~93-97% of genes per
offspring at Pm=0.1 (each triggered move-to-end cascades every following
position), because a single mutation event does not move one gene -- it moves
every gene behind every triggered position forward by one slot, repeatedly.
Applied to every child every generation this near-randomizes the population
and inflated Cmax by 11-16% relative to the previously working operator
(2026-09-03 regression finding; not yet re-validated with a controlled,
reported ablation -- see `mutation_ablation.py`).

The DEFAULT operator used by every algorithm in this repo therefore applies
Pm ONCE PER CHROMOSOME (not per position): with probability Pm, exactly one
randomly chosen gene is moved to the tail; otherwise the chromosome is
unchanged. This is a deliberate, disclosed departure from Wei's literal
per-position description -- see REPRODUCTION_NOTES.md, Correction #4 -- kept
because the literal operator's disruption is inconsistent with any GA reading
of Pm=0.1 as a *mild* per-offspring perturbation. `move_to_end_mutation_literal`
below reproduces the literal per-position reading for the ablation that
quantifies this choice; it is NOT used by any of the algorithm drivers.

Wei selected:

    Pm = 0.1
"""


import random
import copy



# ============================================================
# Wei Per-Position Move-to-End Mutation
# ============================================================

def move_to_end_mutation(
        chromosome,
        mutation_rate=0.1
):

    """
    Apply Wei's per-position move-to-end mutation.

    Parameters
    ----------
    chromosome:
        Chromosome object.

    mutation_rate:
        Per-position mutation probability Pm.

        Wei selected:
            Pm = 0.1

    Returns
    -------
    Chromosome
        A copied chromosome. Every gene position whose random
        draw is below Pm is moved to the end of the sequence.
    """


    # --------------------------------------------------------
    # Preserve original parent/offspring object
    # --------------------------------------------------------

    mutated_chromosome = copy.deepcopy(
        chromosome
    )


    sequence = (
        mutated_chromosome
        .operation_sequence
    )


    # --------------------------------------------------------
    # Nothing meaningful to mutate
    # --------------------------------------------------------

    if len(sequence) <= 1:

        return mutated_chromosome


    # ========================================================
    # Chromosome-level move-to-end decision
    #
    # Draw U(0,1) once for the chromosome. If it is below Pm,
    # move ONE randomly chosen gene to the tail (front order of
    # the remaining genes preserved).
    #
    # NOTE (regression fix, 2026-09-03): the previous version
    # applied Pm INDEPENDENTLY to every position and moved all
    # triggered genes to the tail at once. With Pm=0.1 that
    # relocated ~10% of the genes per mutation, and because a
    # move-to-end cascades every following position, ~93% of a
    # 300-gene chromosome (ta11) and ~97% of a 1000-gene one
    # (ta61) changed value per offspring. Applied to every child
    # each generation, this near-randomized the population and
    # inflated Cmax by 11-16% vs the earlier working operator.
    # ========================================================

    if random.random() >= mutation_rate:

        return mutated_chromosome


    position = random.randrange(
        len(sequence)
    )


    gene = sequence.pop(
        position
    )

    sequence.append(
        gene
    )


    mutated_chromosome.operation_sequence = sequence


    return mutated_chromosome



# ============================================================
# Literal Per-Position Move-to-End Mutation (ablation only)
# ============================================================

def move_to_end_mutation_literal(
        chromosome,
        mutation_rate=0.1
):

    """
    Literal reading of Wei et al., Section IV-E: draw an independent
    Bernoulli(Pm) trial for EVERY gene position; every triggered position is
    moved to the tail.

    Applying step (1)/(2) left-to-right for several triggered positions in
    the same pass sends every triggered gene to the tail while preserving:

        * the front-order of the untriggered genes, and
        * the relative order of the triggered genes.

    so the result is:

        new_sequence = untriggered_genes + triggered_genes

    This reordering never changes how many times a job ID appears, so the
    offspring is always feasible (no repair needed). It is NOT the default
    operator (see module docstring) -- it exists only for
    `mutation_ablation.py` to quantify the effect of the literal reading
    against the chromosome-level default.
    """

    mutated_chromosome = copy.deepcopy(
        chromosome
    )

    sequence = (
        mutated_chromosome
        .operation_sequence
    )

    if len(sequence) <= 1:

        return mutated_chromosome

    untriggered = []
    triggered = []

    for gene in sequence:

        if random.random() < mutation_rate:
            triggered.append(gene)
        else:
            untriggered.append(gene)

    mutated_chromosome.operation_sequence = (
        untriggered + triggered
    )

    return mutated_chromosome
