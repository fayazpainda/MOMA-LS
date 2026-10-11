"""
Wei2021 Reproduction

SPEA2 Comparator

Purpose
-------
Implement SPEA2 (Zitzler, Laumanns & Thiele, "SPEA2: Improving the Strength
Pareto Evolutionary Algorithm," TIK-Report 103, ETH Zurich, 2001) on the same
reconstructed extended Taillard benchmark used for NSGA-II / NSGA-III /
U-NSGA-III, so it can be added as another baseline in the multi-algorithm
robustness study (`algo_comparison.py`).

Objectives:

    min(Cmax, TWET, NEC)

Same experimental settings as the other comparators (population size,
crossover/mutation probabilities and operators, MME initialization,
objectives). The only thing that changes is fitness assignment and
environmental/mating selection, which is SPEA2's contribution:

    * fitness(i) = raggedness(i) + density(i)
        raggedness(i) = sum of raw dominance strengths of everyone that
                         dominates i (0 for any non-dominated individual)
        density(i)    = 1 / (sigma_k + 2), sigma_k = distance to the k-th
                         nearest neighbour in (min-max normalized) objective
                         space, k = round(sqrt(n))
      lower fitness is better; fitness < 1 means non-dominated.
    * a fixed-size external archive holds the best `population_size`
      individuals every generation (all non-dominated members if they fit,
      else filled by the lowest-fitness dominated members, else truncated by
      iteratively removing the individual closest to another archive member).
    * mating selection is binary tournament on fitness, performed on the
      archive only (not on the raw population), which is what makes SPEA2's
      selection pressure elitist.

Archive size
------------
Fixed to `population_size` (Zitzler et al.'s usual N == N-bar setting). This
keeps SPEA2 on the same single population-size knob as every other comparator
in the study.

Decode-budget note
-------------------
Canonically SPEA2 only decodes the ~N offspring each generation (archive
members already carry their objective values from when they were selected).
The equal-decode-budget harness in `algo_comparison.py` / `memetic_comparison`
assumes every method costs exactly `2 * population_size` decodes per
generation (`loop_decodes`), matching NSGA-II/III's habit of re-evaluating
both the surviving population and the offspring from scratch every
generation. To keep the shared budget arithmetic correct without touching
that already-validated harness, this implementation deliberately re-decodes
the incoming archive at the top of every generation too. `decode()` is a
pure function of the chromosome, so this changes no objective value -- it
only restores the 2x-per-generation decode cost the harness assumes.
"""


import copy
import random

from mme import initialize_MME_population
from evaluation import evaluate_population
from genetic import crossover_OS
from mutation import move_to_end_mutation
from pareto import dominates


# ============================================================
# Common Experimental Parameters
# ============================================================

CROSSOVER_PROBABILITY = 0.9

MUTATION_PROBABILITY = 0.1

OBJECTIVE_NAMES = ["Cmax", "TWET", "NEC"]


# ============================================================
# SPEA2 Fitness Assignment
# ============================================================

def assign_spea2_fitness(results):
    """
    Assign SPEA2 fitness (raggedness + density) to every solution in
    `results`, in place, as `solution["fitness"]` (lower is better;
    fitness < 1.0 marks a non-dominated member of `results`).
    """

    n = len(results)

    if n == 0:
        return

    if n == 1:
        results[0]["fitness"] = 0.0
        return

    # --------------------------------------------------------
    # Raw strength S(i) and raggedness R(i)
    # --------------------------------------------------------

    dominance = [[False] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if i != j and dominates(results[i], results[j]):
                dominance[i][j] = True

    strength = [sum(row) for row in dominance]

    raggedness = [0] * n

    for i in range(n):
        raggedness[i] = sum(
            strength[j] for j in range(n) if dominance[j][i]
        )

    # --------------------------------------------------------
    # Density D(i): k-th nearest neighbour on normalized objectives
    # --------------------------------------------------------

    lo = [min(r[name] for r in results) for name in OBJECTIVE_NAMES]
    hi = [max(r[name] for r in results) for name in OBJECTIVE_NAMES]
    span = [
        (h - l) if (h - l) > 1e-9 else 1.0
        for l, h in zip(lo, hi)
    ]

    def normalized(r):
        return tuple(
            (r[name] - lo[k]) / span[k]
            for k, name in enumerate(OBJECTIVE_NAMES)
        )

    vectors = [normalized(r) for r in results]

    k = max(1, round(n ** 0.5))

    density = [0.0] * n

    for i in range(n):
        distances = sorted(
            sum((vectors[i][d] - vectors[j][d]) ** 2 for d in range(3)) ** 0.5
            for j in range(n)
            if j != i
        )
        sigma_k = distances[min(k - 1, len(distances) - 1)]
        density[i] = 1.0 / (sigma_k + 2.0)

    for i, r in enumerate(results):
        r["fitness"] = raggedness[i] + density[i]


# ============================================================
# Environmental Selection (build A_{t+1} from R_t = P_t U A_t)
# ============================================================

def _truncate(front, archive_size):
    """
    SPEA2 truncation operator: while the front is larger than the archive,
    repeatedly remove the individual whose distance to its nearest archive
    neighbour is smallest (ties broken by the next-nearest distance, etc.).
    """

    if len(front) <= archive_size:
        return front

    lo = [min(r[name] for r in front) for name in OBJECTIVE_NAMES]
    hi = [max(r[name] for r in front) for name in OBJECTIVE_NAMES]
    span = [
        (h - l) if (h - l) > 1e-9 else 1.0
        for l, h in zip(lo, hi)
    ]

    def normalized(r):
        return tuple(
            (r[name] - lo[k]) / span[k]
            for k, name in enumerate(OBJECTIVE_NAMES)
        )

    vectors = {id(r): normalized(r) for r in front}

    def distance(a, b):
        va, vb = vectors[id(a)], vectors[id(b)]
        return sum((x - y) ** 2 for x, y in zip(va, vb)) ** 0.5

    remaining = list(front)

    while len(remaining) > archive_size:

        sorted_distances = {
            id(r): sorted(
                distance(r, other) for other in remaining if other is not r
            )
            for r in remaining
        }

        worst = min(remaining, key=lambda r: sorted_distances[id(r)])
        remaining.remove(worst)

    return remaining


def environmental_selection(combined, archive_size):
    """
    Build the next archive A_{t+1} from `combined` = P_t U A_t.

    Assumes `combined` already carries `["fitness"]` (see
    `assign_spea2_fitness`).
    """

    nondominated = [r for r in combined if r["fitness"] < 1.0]

    if len(nondominated) == archive_size:
        return nondominated

    if len(nondominated) < archive_size:

        dominated_sorted = sorted(
            (r for r in combined if r["fitness"] >= 1.0),
            key=lambda r: r["fitness"],
        )

        needed = archive_size - len(nondominated)

        return nondominated + dominated_sorted[:needed]

    return _truncate(nondominated, archive_size)


# ============================================================
# Mating Selection (binary tournament on archive fitness)
# ============================================================

def _tournament_winner(a, b):
    if a["fitness"] < b["fitness"]:
        return a
    if b["fitness"] < a["fitness"]:
        return b
    return random.choice([a, b])


def build_spea2_mating_pool(archive_results, population_size):
    """
    Construct a mating pool of size `population_size` using independent
    binary tournaments on SPEA2 fitness, drawn from the archive only.
    """

    if not archive_results:
        raise ValueError(
            "Cannot construct mating pool from an empty archive."
        )

    if population_size <= 0:
        raise ValueError("population_size must be greater than zero.")

    if len(archive_results) == 1:
        return [
            copy.deepcopy(archive_results[0]["chromosome"])
            for _ in range(population_size)
        ]

    mating_pool = []

    while len(mating_pool) < population_size:

        candidate1, candidate2 = random.sample(archive_results, 2)

        winner = _tournament_winner(candidate1, candidate2)

        mating_pool.append(winner["chromosome"])

    return mating_pool


# ============================================================
# Generate Offspring Q_t
# ============================================================

def generate_offspring(mating_pool, population_size):
    """
    Generate offspring Q_t using the same crossover/mutation operators as
    the other comparators.
    """

    if len(mating_pool) != population_size:
        raise RuntimeError(
            "Mating pool size mismatch: "
            f"expected {population_size}, obtained {len(mating_pool)}."
        )

    offspring = []

    for index in range(0, population_size, 2):

        parent1 = mating_pool[index]

        if index + 1 < population_size:
            parent2 = mating_pool[index + 1]
        else:
            parent2 = mating_pool[0]

        if random.random() < CROSSOVER_PROBABILITY:
            child1, child2 = crossover_OS(parent1, parent2)
        else:
            child1 = copy.deepcopy(parent1)
            child2 = copy.deepcopy(parent2)

        child1 = move_to_end_mutation(child1, MUTATION_PROBABILITY)
        child2 = move_to_end_mutation(child2, MUTATION_PROBABILITY)

        if len(offspring) < population_size:
            offspring.append(child1)

        if len(offspring) < population_size:
            offspring.append(child2)

    if len(offspring) != population_size:
        raise RuntimeError(
            "Offspring population size mismatch: "
            f"expected {population_size}, obtained {len(offspring)}."
        )

    return offspring


# ============================================================
# Standard SPEA2
# ============================================================

def run_SPEA2(
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
    Run the SPEA2 comparator.

    Signature matches run_UNSGAIII / run_NSGAIII / run_NSGAII so it plugs
    directly into the existing comparison harness.

    generation_callback, local_search, right_shift:
        Optional hooks with identical semantics to run_UNSGAIII. Defaults
        (None/None/False) reproduce plain SPEA2; supplying a
        ParetoLocalSearch turns this into memetic SPEA2, enabling the
        cross-framework ablation.

    Returns
    -------
    list
        Final chromosome archive (size population_size).
    """

    if population_size <= 0:
        raise ValueError("population_size must be greater than zero.")

    if generations < 0:
        raise ValueError("generations cannot be negative.")

    archive_size = population_size

    # Bootstrap: the archive at generation 0 is the initial MME population
    # (an empty external archive combined with P_0, matching the canonical
    # scheme where A_0 = 0 and mating for generation 0 draws from P_0).
    archive_chromosomes = initialize_MME_population(
        instance, population_size
    )

    if len(archive_chromosomes) != population_size:
        raise RuntimeError(
            "Initialization produced "
            f"{len(archive_chromosomes)} individuals; "
            f"expected {population_size}."
        )

    for generation in range(generations):

        if verbose and (generation == 1 or (
            generation > 1 and generation % 10 == 0
        )):
            print(f"SPEA2 Generation {generation}/{generations}")

        # Re-decode the incoming archive (see module docstring: this keeps
        # the per-generation decode cost at 2*population_size, matching the
        # equal-decode-budget harness).
        archive_results = evaluate_population(
            instance, archive_chromosomes, due_dates, weights, machines,
            right_shift=right_shift
        )

        assign_spea2_fitness(archive_results)

        if generation_callback is not None:
            generation_callback(generation, archive_results)

        mating_pool = build_spea2_mating_pool(
            archive_results, population_size
        )

        offspring = generate_offspring(mating_pool, population_size)

        # Memetic local search on Q_t (optional; None = baseline).
        if local_search is not None:
            offspring = local_search.improve(
                instance, offspring, due_dates, weights, machines
            )
            if len(offspring) != population_size:
                raise RuntimeError(
                    "Local search changed the SPEA2 offspring size: "
                    f"expected {population_size}, got {len(offspring)}."
                )

        offspring_results = evaluate_population(
            instance, offspring, due_dates, weights, machines,
            right_shift=right_shift
        )

        combined = archive_results + offspring_results

        assign_spea2_fitness(combined)

        new_archive_results = environmental_selection(
            combined, archive_size
        )

        archive_chromosomes = [
            r["chromosome"] for r in new_archive_results
        ]

        if len(archive_chromosomes) != population_size:
            raise RuntimeError(
                "Next SPEA2 archive has "
                f"{len(archive_chromosomes)} individuals; "
                f"expected {population_size}."
            )

    return archive_chromosomes
