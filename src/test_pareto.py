"""Pareto front keeps only non-dominated solutions (minimization)."""

from pareto import get_pareto_front


def test_get_pareto_front_drops_dominated():
    solutions = [
        {"Cmax": 10, "TWET": 20, "NEC": 200},
        {"Cmax": 15, "TWET": 10, "NEC": 150},
        {"Cmax": 20, "TWET": 30, "NEC": 300},  # dominated by the first
        {"Cmax": 12, "TWET": 18, "NEC": 190},
    ]

    front = get_pareto_front(solutions)

    dominated = {"Cmax": 20, "TWET": 30, "NEC": 300}
    assert dominated not in front
    assert len(front) == 3
    assert all(sol in solutions for sol in front)
