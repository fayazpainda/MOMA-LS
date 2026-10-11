"""
Wei2021 Reproduction

Fast Non-Dominated Sorting

This implementation does NOT rely on solution["id"]
being globally unique.

Parent and offspring populations may both contain
IDs starting from zero, so Python list indices are
used internally during sorting.
"""


from pareto import dominates



def fast_non_dominated_sort(
        population
):

    """
    Fast non-dominated sorting.

    Parameters
    ----------
    population:
        List of evaluated solution dictionaries.

    Returns
    -------
    fronts:
        List of Pareto fronts.

        fronts[0] = best non-dominated front
        fronts[1] = second front
        ...

    Every solution receives:

        solution["rank"]

    where rank 0 is the best front.
    """


    population_size = len(
        population
    )


    if population_size == 0:

        return []


    # ========================================================
    # Internal structures
    #
    # IMPORTANT:
    # Use list indices instead of solution["id"].
    # ========================================================

    domination_count = [

        0
        for _ in range(
            population_size
        )

    ]


    dominated_solutions = [

        []
        for _ in range(
            population_size
        )

    ]


    first_front_indices = []



    # ========================================================
    # Compare every pair of solutions
    # ========================================================

    for p_index in range(
        population_size
    ):


        p = population[
            p_index
        ]


        for q_index in range(
            population_size
        ):


            # Same actual solution

            if p_index == q_index:

                continue


            q = population[
                q_index
            ]


            # p dominates q

            if dominates(
                p,
                q
            ):

                dominated_solutions[
                    p_index
                ].append(
                    q_index
                )


            # q dominates p

            elif dominates(
                q,
                p
            ):

                domination_count[
                    p_index
                ] += 1



        # ----------------------------------------------------
        # First Pareto front
        # ----------------------------------------------------

        if domination_count[
            p_index
        ] == 0:


            p["rank"] = 0


            first_front_indices.append(

                p_index

            )



    # ========================================================
    # Build all subsequent fronts
    # ========================================================

    front_indices = [

        first_front_indices

    ]


    current_front = 0



    while (

        current_front
        <
        len(front_indices)

        and

        len(
            front_indices[
                current_front
            ]
        )
        >
        0

    ):


        next_front = []


        for p_index in front_indices[
            current_front
        ]:


            for q_index in dominated_solutions[
                p_index
            ]:


                domination_count[
                    q_index
                ] -= 1


                if domination_count[
                    q_index
                ] == 0:


                    population[
                        q_index
                    ]["rank"] = (

                        current_front
                        +
                        1

                    )


                    next_front.append(

                        q_index

                    )



        if next_front:

            front_indices.append(

                next_front

            )


        current_front += 1



    # ========================================================
    # Convert indices back to solution dictionaries
    # ========================================================

    fronts = []


    for indices in front_indices:


        front = [

            population[index]

            for index in indices

        ]


        if front:

            fronts.append(

                front

            )



    return fronts