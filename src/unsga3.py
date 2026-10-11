"""
Wei2021 Reproduction

U-NSGA-III Framework

Paper-confirmed Wei parameters:

    Population size N = 100
    Crossover probability Pc = 0.9
    Mutation probability Pm = 0.1

Wei Algorithm 1:

    1. P'_t = NichingBasedSelection(P_t)
    2. Q_t  = Crossover + Mutation(P'_t)
    3. R_t  = P_t union Q_t
    4. Non-dominated sort R_t
    5. Construct S_t
    6. Normalize objectives
    7. Associate solutions with reference directions
    8. Niching environmental selection
    9. Construct P_(t+1)

Important reproduction notes
----------------------------

1. Wei confirms the niching-based tournament rule.

2. The complete mating pool is constructed before
   crossover.

3. Wei reports:

       Pc = 0.9
       Pm = 0.1
       N  = 100

4. The exact structured-reference division parameter H
   is not clearly reported by Wei.

   We currently use divisions = 12:

       H = 91 reference directions

   which satisfies the U-NSGA-III requirement:

       N > H

   because:

       100 > 91

   This remains a reproduction setting,
   not a Wei-confirmed numerical parameter.

5. The number of generations is supplied externally.
   Wei does not clearly report generations = 200,
   so 200 must remain documented as our
   experimental reproduction setting.
"""


import copy
import random


from mating import build_chromosome_mating_pool

from mme import initialize_MME_population

from evaluation import evaluate_population

from genetic import crossover_OS

from mutation import move_to_end_mutation

from nsga3_selection import (
    nsga3_environmental_selection,
    prepare_reference_metadata
)

from reference_points import generate_reference_points



# ============================================================
# Wei Paper Parameters
# ============================================================

CROSSOVER_PROBABILITY = 0.9

MUTATION_PROBABILITY = 0.1



# ============================================================
# Structured Reference Points
# ============================================================

def generate_reference_points_3d(
        divisions
):

    """
    Generate Das-Dennis structured reference directions
    for the three objectives:

        Cmax
        TWET
        NEC

    Every reference direction satisfies:

        z1 + z2 + z3 = 1

        z1 >= 0
        z2 >= 0
        z3 >= 0

    For M = 3 objectives and divisions = 12:

        H = C(12 + 3 - 1, 3 - 1)
          = C(14, 2)
          = 91

    Wei states that structured reference points are used,
    but the exact division parameter is not clearly
    reported.

    Therefore divisions = 12 remains an explicit
    reproduction setting.
    """


    if divisions <= 0:

        raise ValueError(

            "divisions must be greater than zero"

        )



    reference_points = []



    for i in range(

        divisions + 1

    ):


        for j in range(

            divisions - i + 1

        ):


            k = (

                divisions

                -

                i

                -

                j

            )



            reference_points.append(

                [

                    i / divisions,

                    j / divisions,

                    k / divisions

                ]

            )



    return reference_points



# ============================================================
# Validate Mating Pool
# ============================================================

def validate_mating_pool(
        mating_pool,
        population_size
):

    """
    Defensive validation.

    U-NSGA-III requires P'_t to contain N chromosomes.
    """


    if len(mating_pool) != population_size:

        raise RuntimeError(

            "Mating pool size mismatch: "
            f"expected {population_size}, "
            f"obtained {len(mating_pool)}."

        )



# ============================================================
# Generate Offspring Population Q_t
# ============================================================

def generate_offspring(
        mating_pool,
        population_size,
        mutation_fn=None
):

    """
    Generate Q_t from P'_t.

    For each parent pair:

        1. Apply two-point crossover with Pc = 0.9.
        2. Otherwise copy both parents.
        3. Apply Wei mutation operator with Pm = 0.1.
        4. Add both children to Q_t.

    For Wei's N = 100, this produces exactly
    100 offspring.

    mutation_fn:
        Optional callable(chromosome, mutation_rate) -> chromosome. Defaults
        to None, which resolves to `mutation.move_to_end_mutation` (the
        validated baseline). See `run_UNSGAIII`'s `mutation_fn` parameter.
    """


    if mutation_fn is None:

        mutation_fn = move_to_end_mutation


    validate_mating_pool(

        mating_pool,

        population_size

    )



    offspring = []



    # ========================================================
    # Parent Pairs
    # ========================================================

    for index in range(

        0,

        population_size,

        2

    ):


        parent1 = (

            mating_pool[
                index
            ]

        )


        # ----------------------------------------------------
        # Wei uses N = 100, so the final parent always
        # has a partner.
        #
        # This defensive branch supports arbitrary odd N.
        # ----------------------------------------------------

        if index + 1 < population_size:

            parent2 = (

                mating_pool[
                    index + 1
                ]

            )

        else:

            parent2 = (

                mating_pool[0]

            )



        # ====================================================
        # Crossover
        #
        # Wei:
        #
        # Pc = 0.9
        #
        # Two-point crossover produces two offspring.
        # ====================================================

        if (

            random.random()

            <

            CROSSOVER_PROBABILITY

        ):


            child1, child2 = crossover_OS(

                parent1,

                parent2

            )



        else:


            # ------------------------------------------------
            # No crossover
            #
            # Keep two independent chromosome copies.
            # ------------------------------------------------

            child1 = copy.deepcopy(

                parent1

            )


            child2 = copy.deepcopy(

                parent2

            )



        # ====================================================
        # Mutation
        #
        # Wei: Pm = 0.1. Default operator is the chromosome-level
        # move-to-end mutation (mutation.move_to_end_mutation), a
        # disclosed deviation from Wei's literal per-position
        # description -- see mutation.py module docstring and
        # REPRODUCTION_NOTES.md Correction #4.
        # ====================================================

        child1 = mutation_fn(

            child1,

            MUTATION_PROBABILITY

        )


        child2 = mutation_fn(

            child2,

            MUTATION_PROBABILITY

        )



        # ====================================================
        # Add Children
        # ====================================================

        if len(offspring) < population_size:

            offspring.append(

                child1

            )


        if len(offspring) < population_size:

            offspring.append(

                child2

            )



    # ========================================================
    # Validation
    # ========================================================

    if len(offspring) != population_size:

        raise RuntimeError(

            "Offspring population size mismatch: "
            f"expected {population_size}, "
            f"obtained {len(offspring)}."

        )



    return offspring



# ============================================================
# U-NSGA-III
# ============================================================

def run_UNSGAIII(
        instance,
        generations,
        population_size,
        due_dates,
        weights,
        machines,
        reference_mode="uniform",
        weight_vector=None,
        radius=None,
        divisions=12,
        generation_callback=None,
        local_search=None,
        right_shift=False,
        mutation_fn=None,
        verbose=True
):

    """
    Execute the Wei U-NSGA-III reproduction.

    Parameters
    ----------
    instance:
        Classical JSP instance.

    generations:
        Evolutionary stopping setting.

        IMPORTANT:
        Wei does not clearly report generations = 200.
        Therefore this value comes from the calling
        experiment and must remain documented as a
        reproduction setting.

    population_size:
        Wei selected N = 100.

    due_dates:
        Extended-Taillard due dates.

    weights:
        alpha_i and beta_i.

    machines:
        Machine energy parameters.

    reference_mode:
        "uniform" (default) reproduces the validated Das-Dennis baseline.
        "ahp_guided" concentrates reference directions near weight_vector.
        This is the ONLY thing that differs between baseline and AHP runs.

    weight_vector:
        Length-3 AHP weight vector (Cmax, TWET, NEC). Required when
        reference_mode="ahp_guided"; ignored otherwise.

    radius:
        Filtering radius for the AHP-guided reference set. If None, the
        reference_points module default is used.

    divisions:
        Das-Dennis division number p (H = C(p+2, 2) for M = 3). Default 12
        gives H = 91, matching the validated baseline.

    generation_callback:
        Optional callable(generation_index, parent_results) invoked once per
        generation with the evaluated current population P_t. Used to log
        per-generation convergence (e.g. hypervolume) without changing the
        algorithm. Defaults to None (no logging).

    local_search:
        Optional memetic local-search object exposing
        `improve(instance, population, due_dates, weights, machines,
        reference_points=...) -> population`. When provided (e.g. a
        `pareto_local_search.ParetoLocalSearch`), it refines the offspring Q_t
        each generation before evaluation. Defaults to None, which reproduces the
        validated baseline exactly (no local search).

    right_shift:
        When True, every objective evaluation applies the Pareto-safe
        energy-aware right-shift NEC refinement (MEMETIC_DESIGN.md §3b/§4.3a) via
        `evaluation.evaluate_individual`. Cmax and TWET are unchanged; NEC is
        refined downward or left as-is. Must be set identically for every method
        being compared. Defaults to False (validated semi-active scoring).

    mutation_fn:
        Optional callable(chromosome, mutation_rate) -> chromosome, used in
        place of the default `mutation.move_to_end_mutation`. Exists so
        `mutation_ablation.py` can substitute
        `mutation.move_to_end_mutation_literal` without touching this
        function's control flow. Defaults to None, which reproduces the
        validated baseline exactly (chromosome-level move-to-end mutation).

    Returns
    -------
    population:
        Final chromosome population P_t.
    """



    if population_size <= 0:

        raise ValueError(

            "population_size must be greater than zero."

        )


    if mutation_fn is None:

        mutation_fn = move_to_end_mutation


    if generations < 0:

        raise ValueError(

            "generations cannot be negative."

        )



    # ========================================================
    # Initial Population P_0
    #
    # Wei:
    #
    # MME + random generation.
    #
    # Exact MME/random ratio is not published.
    # ========================================================

    population = initialize_MME_population(

        instance,

        population_size

    )



    if len(population) != population_size:

        raise RuntimeError(

            "Initialization produced "
            f"{len(population)} individuals; "
            f"expected {population_size}."

        )



    # ========================================================
    # Structured Reference Points Z^s
    #
    # Current reproduction:
    #
    # divisions = 12
    # H = 91
    #
    # N = 100 > H = 91
    # ========================================================

    reference_points = generate_reference_points(

        mode=reference_mode,

        weight_vector=weight_vector,

        p=divisions,

        radius=radius

    )



    if population_size <= len(reference_points):

        raise ValueError(

            "U-NSGA-III requires population size N "
            "to be larger than the number of "
            "reference directions H. "
            f"N={population_size}, "
            f"H={len(reference_points)}."

        )



    # ========================================================
    # Evolutionary Generations
    # ========================================================

    for generation in range(

        generations

    ):


        if verbose and (generation == 1 or (generation > 1 and generation % 10 == 0)):
         print(
        f"U-NSGA-III Generation {generation}/{generations}"
    )



        # ====================================================
        # Evaluate Parent Population P_t
        # ====================================================

        parent_results = evaluate_population(

            instance,

            population,

            due_dates,

            weights,

            machines,

            right_shift=right_shift

        )



        # ====================================================
        # Per-generation convergence hook
        #
        # Reports the evaluated current population P_t so the
        # caller can log a convergence indicator (e.g. HV)
        # without altering the algorithm. No-op when None.
        # ====================================================

        if generation_callback is not None:

            generation_callback(

                generation,

                parent_results

            )



        # ====================================================
        # Prepare U-NSGA-III Tournament Information
        #
        # Required metadata:
        #
        # rank
        # reference_id
        # perpendicular_distance
        # ====================================================

        prepare_reference_metadata(

            parent_results,

            reference_points

        )



        # ====================================================
        # Algorithm 1 Step 2
        #
        # P'_t = NichingBasedSelection(P_t)
        # ====================================================

        mating_pool = build_chromosome_mating_pool(

            parent_results

        )



        validate_mating_pool(

            mating_pool,

            population_size

        )



        # ====================================================
        # Algorithm 1 Step 3
        #
        # Q_t =
        # Crossover + Mutation(P'_t)
        # ====================================================

        offspring = generate_offspring(

            mating_pool,

            population_size,

            mutation_fn=mutation_fn

        )



        # ====================================================
        # Memetic Local Search (optional)
        #
        # Refine Q_t before evaluation. When local_search is None
        # (default) this block is skipped and the algorithm is the
        # validated baseline byte-for-byte. See MEMETIC_DESIGN.md.
        # ====================================================

        if local_search is not None:

            offspring = local_search.improve(

                instance,

                offspring,

                due_dates,

                weights,

                machines,

                reference_points=reference_points

            )



            if len(offspring) != population_size:

                raise RuntimeError(

                    "Local search changed the offspring population size: "
                    f"expected {population_size}, "
                    f"obtained {len(offspring)}."

                )



        # ====================================================
        # Evaluate Offspring Population Q_t
        # ====================================================

        offspring_results = evaluate_population(

            instance,

            offspring,

            due_dates,

            weights,

            machines,

            right_shift=right_shift

        )



        # ====================================================
        # Algorithm 1 Step 4
        #
        # R_t = P_t union Q_t
        # ====================================================

        combined = (

            parent_results

            +

            offspring_results

        )



        expected_combined_size = (

            2

            *

            population_size

        )


        if len(combined) != expected_combined_size:

            raise RuntimeError(

                "Combined population R_t has "
                f"{len(combined)} individuals; "
                f"expected {expected_combined_size}."

            )



        # ====================================================
        # Algorithm 1 Steps 5-18
        #
        # Non-dominated sorting
        # normalization
        # association
        # niche counting
        # environmental selection
        # ====================================================

        selected = nsga3_environmental_selection(

            combined,

            population_size,

            reference_points

        )



        # ====================================================
        # Construct P_(t+1)
        # ====================================================

        population = [

            solution[
                "chromosome"
            ]

            for solution
            in selected

        ]



        if len(population) != population_size:

            raise RuntimeError(

                "Next population P_(t+1) has "
                f"{len(population)} individuals; "
                f"expected {population_size}."

            )



    # ========================================================
    # Final Population
    # ========================================================

    return population