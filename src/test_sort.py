"""Fast non-dominated sort partitions the whole population into fronts."""

from nondominated_sort import fast_non_dominated_sort


def test_fast_non_dominated_sort_partitions_population():
    population = [
        {"id": 0, "Cmax": 10, "TWET": 20, "NEC": 200},
        {"id": 1, "Cmax": 15, "TWET": 10, "NEC": 150},
        {"id": 2, "Cmax": 30, "TWET": 40, "NEC": 300},  # dominated by all others
        {"id": 3, "Cmax": 12, "TWET": 25, "NEC": 250},
    ]

    fronts = fast_non_dominated_sort(population)

    assert len(fronts) >= 1
    # Every individual appears in exactly one front.
    assert sum(len(front) for front in fronts) == len(population)
    # The strictly-dominated solution cannot be in the first front.
    assert not any(sol["id"] == 2 for sol in fronts[0])
