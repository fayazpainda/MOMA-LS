"""
Wei2021 Reproduction

Pareto Dominance Functions

"""


def dominates(a, b):

    """
    Return True if solution a dominates b
    """

    objectives_a = [
        a["Cmax"],
        a["TWET"],
        a["NEC"]
    ]

    objectives_b = [
        b["Cmax"],
        b["TWET"],
        b["NEC"]
    ]


    better_or_equal = all(
        x <= y
        for x, y in zip(
            objectives_a,
            objectives_b
        )
    )


    strictly_better = any(
        x < y
        for x, y in zip(
            objectives_a,
            objectives_b
        )
    )


    return (
        better_or_equal
        and
        strictly_better
    )



def get_pareto_front(population):

    pareto = []


    for solution in population:

        dominated = False


        for other in population:

            if dominates(
                other,
                solution
            ):

                dominated = True

                break


        if not dominated:

            pareto.append(solution)


    return pareto