"""
Wei2021 Reproduction

Standard NSGA-III Comparator

Purpose
-------
Implement NSGA-III on the SAME reconstructed extended
Taillard benchmark used for U-NSGA-III.

Objectives:

    min(Cmax, TWET, NEC)

Common experimental settings:

    Population size N = 100
    Crossover probability Pc = 0.9
    Mutation probability Pm = 0.1

Main comparison difference
--------------------------

NSGA-III:
    binary tournament based on non-dominated rank

U-NSGA-III:
    Wei niching-based tournament using:
        reference direction
        non-dominated rank
        perpendicular distance

Everything else is kept as consistent as possible:

    same benchmark
    same initialization
    same chromosome representation
    same crossover
    same mutation
    same objectives
    same environmental selection
    same reference directions
    same population size
    same generations

Important
---------
Wei does not publish every implementation detail of the
NSGA-III comparator.

Therefore the binary tournament used here is a documented
reproduction choice:

    lower rank wins
    equal rank -> random winner

This avoids introducing NSGA-II crowding distance into
the NSGA-III comparator.
"""


import copy
import random


from mme import initialize_MME_population

from evaluation import evaluate_population

from genetic import crossover_OS

from mutation import move_to_end_mutation

from nondominated_sort import fast_non_dominated_sort

from nsga3_selection import (
    nsga3_environmental_selection
)



# ============================================================
# Common Experimental Parameters
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
    for three objectives:

        Cmax
        TWET
        NEC

    For:

        divisions = 12

    number of reference directions:

        H = C(14, 2) = 91
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
# Assign Non-Dominated Rank
# ============================================================

def assign_parent_ranks(
        population_results
):
    """
    Perform non-dominated sorting on P_t.

    Assign:

        rank = 0 for F1
        rank = 1 for F2
        rank = 2 for F3
        ...

    These ranks are used by the NSGA-III binary
    tournament mating selection.
    """


    fronts = fast_non_dominated_sort(

        population_results

    )



    for rank, front in enumerate(

        fronts

    ):


        for solution in front:


            solution["rank"] = (

                rank

            )



    return fronts



# ============================================================
# Standard Binary Tournament
# ============================================================

def binary_tournament(
        solution1,
        solution2
):
    """
    NSGA-III comparator binary tournament.

    Rule:

        lower non-dominated rank wins

    If both solutions have the same rank:

        choose one randomly

    We intentionally do NOT use crowding distance because
    crowding distance belongs to NSGA-II and is not part of
    the reference-point environmental selection of NSGA-III.
    """


    if "rank" not in solution1:

        raise KeyError(

            "solution1 missing 'rank'"

        )


    if "rank" not in solution2:

        raise KeyError(

            "solution2 missing 'rank'"

        )



    rank1 = (

        solution1[
            "rank"
        ]

    )


    rank2 = (

        solution2[
            "rank"
        ]

    )



    # ========================================================
    # Better Rank Wins
    # ========================================================

    if rank1 < rank2:

        return solution1



    if rank2 < rank1:

        return solution2



    # ========================================================
    # Same Rank
    #
    # Random tie-break.
    # ========================================================

    return random.choice(

        [

            solution1,

            solution2

        ]

    )



# ============================================================
# Build Standard NSGA-III Mating Pool
# ============================================================

def build_standard_mating_pool(
        population_results,
        population_size
):
    """
    Construct a mating pool of size N using independent
    binary tournaments.

    Each tournament:

        1. randomly selects two different candidates
        2. compares their non-dominated rank
        3. lower rank wins
        4. same rank -> random winner

    Returns only chromosomes.
    """


    if not population_results:

        raise ValueError(

            "Cannot construct mating pool "
            "from an empty population."

        )



    if population_size <= 0:

        raise ValueError(

            "population_size must be greater than zero."

        )



    if len(population_results) == 1:


        return [

            copy.deepcopy(
                population_results[0]["chromosome"]
            )

            for _ in range(
                population_size
            )

        ]



    mating_pool = []



    # ========================================================
    # Select N Parents
    # ========================================================

    while len(mating_pool) < population_size:


        candidate1, candidate2 = random.sample(

            population_results,

            2

        )



        winner = binary_tournament(

            candidate1,

            candidate2

        )



        mating_pool.append(

            winner[
                "chromosome"
            ]

        )



    return mating_pool



# ============================================================
# Generate Offspring Q_t
# ============================================================

def generate_offspring(
        mating_pool,
        population_size
):
    """
    Generate offspring population Q_t.

    For every parent pair:

        crossover with Pc = 0.9

    otherwise:

        copy both parents

    Then apply mutation with:

        Pm = 0.1

    using the same mutation operator as U-NSGA-III.
    """


    if len(mating_pool) != population_size:

        raise RuntimeError(

            "Mating pool size mismatch: "
            f"expected {population_size}, "
            f"obtained {len(mating_pool)}."

        )



    offspring = []



    # ========================================================
    # Pair Selected Parents
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
        # Wei experiments use N = 100.
        #
        # Therefore this fallback is not used in the
        # final experiment.
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


            child1 = copy.deepcopy(

                parent1

            )


            child2 = copy.deepcopy(

                parent2

            )



        # ====================================================
        # Mutation
        # ====================================================

        child1 = move_to_end_mutation(

            child1,

            MUTATION_PROBABILITY

        )


        child2 = move_to_end_mutation(

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
# Standard NSGA-III
# ============================================================

def run_NSGAIII(
        instance,
        generations,
        population_size,
        due_dates,
        weights,
        machines,
        generation_callback=None,
        local_search=None,
        right_shift=False,
        verbose=True
):
    """
    Run the NSGA-III comparator.

    Parameters
    ----------
    instance:
        Taillard JSP instance.

    generations:
        Number of evolutionary generations.

        200 is currently our reproduction setting.

    population_size:
        Use N = 100 for comparison with Wei.

    due_dates:
        Reconstructed extended Taillard due dates.

    weights:
        alpha_i / beta_i penalty coefficients.

    machines:
        Machine energy parameters.

    Returns
    -------
    list
        Final chromosome population.
    """



    # ========================================================
    # Input Validation
    # ========================================================

    if population_size <= 0:

        raise ValueError(

            "population_size must be greater than zero."

        )



    if generations < 0:

        raise ValueError(

            "generations cannot be negative."

        )



    # ========================================================
    # Initial Population P_0
    #
    # Keep the SAME initialization routine as U-NSGA-III.
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
    # Structured Reference Directions
    # ========================================================

    reference_points = generate_reference_points_3d(

        divisions=12

    )



    if not reference_points:

        raise RuntimeError(

            "No NSGA-III reference directions "
            "were generated."

        )



    # ========================================================
    # Evolutionary Loop
    # ========================================================

    for generation in range(

        generations

    ):


        if verbose and (generation == 1 or (generation > 1 and generation % 10 == 0)):
            print(
                f"NSGA-III Generation {generation}/{generations}"
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



        if generation_callback is not None:
            generation_callback(generation, parent_results)



        # ====================================================
        # Non-Dominated Rank for Parent Selection
        # ====================================================

        assign_parent_ranks(

            parent_results

        )



        # ====================================================
        # Build NSGA-III Mating Pool
        #
        # Standard binary tournament:
        #
        # lower rank wins
        # same rank -> random
        # ====================================================

        mating_pool = build_standard_mating_pool(

            parent_results,

            population_size

        )



        if len(mating_pool) != population_size:

            raise RuntimeError(

                "NSGA-III mating pool has "
                f"{len(mating_pool)} individuals; "
                f"expected {population_size}."

            )



        # ====================================================
        # Generate Q_t
        # ====================================================

        offspring = generate_offspring(

            mating_pool,

            population_size

        )



        # ====================================================
        # Memetic Local Search on Q_t (optional; None = baseline)
        # ====================================================

        if local_search is not None:

            offspring = local_search.improve(
                instance, offspring, due_dates, weights, machines
            )

            if len(offspring) != population_size:
                raise RuntimeError(
                    "Local search changed the NSGA-III offspring size: "
                    f"expected {population_size}, got {len(offspring)}."
                )



        # ====================================================
        # Evaluate Q_t
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
        # NSGA-III Environmental Selection
        #
        # Non-dominated sorting
        # normalization
        # reference association
        # niche counting
        # splitting-front selection
        # ====================================================

        selected = nsga3_environmental_selection(

            combined,

            population_size,

            reference_points

        )



        # ====================================================
        # P_(t+1)
        # ====================================================

        population = [

            solution[
                "chromosome"
            ]

            for solution
            in selected

        ]



        # ====================================================
        # Validation
        # ====================================================

        if len(population) != population_size:

            raise RuntimeError(

                "Next NSGA-III population has "
                f"{len(population)} individuals; "
                f"expected {population_size}."

            )



    # ========================================================
    # Final Population
    # ========================================================

    return population