import random


# ============================================================
# Niching-Based Tournament
# ============================================================

def niching_tournament(
        solution1,
        solution2
):

    """
    U-NSGA-III niching-based tournament.

    Rule:
    1. Different reference directions:
       random selection.

    2. Same reference direction:
       smaller rank wins.

    3. Same rank:
       smaller perpendicular distance wins.
    """


    required_fields = (
        "rank",
        "reference_id",
        "perpendicular_distance"
    )


    for field in required_fields:

        if field not in solution1:
            raise KeyError(
                f"solution1 missing '{field}'"
            )

        if field not in solution2:
            raise KeyError(
                f"solution2 missing '{field}'"
            )


    reference1 = solution1["reference_id"]
    reference2 = solution2["reference_id"]


    # ========================================================
    # Different reference directions
    # ========================================================

    if reference1 != reference2:

        return random.choice(
            [
                solution1,
                solution2
            ]
        )


    # ========================================================
    # Same reference direction
    # Better rank wins
    # ========================================================

    rank1 = solution1["rank"]
    rank2 = solution2["rank"]


    if rank1 < rank2:

        return solution1


    if rank2 < rank1:

        return solution2


    # ========================================================
    # Same rank
    # Smaller distance wins
    # ========================================================

    distance1 = solution1[
        "perpendicular_distance"
    ]

    distance2 = solution2[
        "perpendicular_distance"
    ]


    if distance1 < distance2:

        return solution1


    if distance2 < distance1:

        return solution2


    # Exact tie

    return random.choice(
        [
            solution1,
            solution2
        ]
    )



# ============================================================
# One tournament pass
# ============================================================

def tournament_pass(
        population_results
):

    population_size = len(
        population_results
    )


    if population_size == 0:

        return []


    selected = []


    pair_limit = (
        population_size
        -
        (
            population_size % 2
        )
    )


    for index in range(
        0,
        pair_limit,
        2
    ):

        winner = niching_tournament(
            population_results[index],
            population_results[index + 1]
        )

        selected.append(
            winner
        )


    if population_size % 2 == 1:

        selected.append(
            population_results[-1]
        )


    return selected



# ============================================================
# Complete mating pool
# ============================================================

def build_mating_pool(
        population_results
):


    population_size = len(
        population_results
    )


    if population_size == 0:

        raise ValueError(
            "Cannot create mating pool from empty population."
        )


    if population_size == 1:

        return [
            population_results[0]
        ]



    # First pass

    first_population = (
        population_results.copy()
    )


    first_selected = tournament_pass(
        first_population
    )


    # Second pass

    second_population = (
        population_results.copy()
    )


    random.shuffle(
        second_population
    )


    second_selected = tournament_pass(
        second_population
    )


    mating_results = (
        first_selected
        +
        second_selected
    )


    mating_results = (
        mating_results[:population_size]
    )


    if len(mating_results) != population_size:

        raise RuntimeError(
            "Mating pool size mismatch"
        )


    return mating_results



# ============================================================
# Chromosome mating pool
# ============================================================

def build_chromosome_mating_pool(
        population_results
):


    mating_results = build_mating_pool(
        population_results
    )


    return [
        solution["chromosome"]
        for solution in mating_results
    ]