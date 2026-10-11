"""
Wei2021 Reproduction

SMS-EMOA Comparator

Purpose
-------
Implement SMS-EMOA (Beume, Naujoks & Emmerich, "SMS-EMOA: Multiobjective
selection based on dominated hypervolume," European Journal of Operational
Research 181(3), 2007) on the same reconstructed extended Taillard benchmark
used for NSGA-II / NSGA-III / U-NSGA-III / SPEA2 / MOEA-D, adding the one
selection paradigm (indicator-based) missing from that set: NSGA-II and
NSGA-III use dominance + crowding/reference-point niching, SPEA2 uses
strength-Pareto density, MOEA-D uses scalar decomposition, and this adds
hypervolume-contribution-based selection.

Objectives:

    min(Cmax, TWET, NEC)

Same experimental settings as the other comparators (population size,
crossover/mutation probabilities and operators, MME initialization,
objectives). The only thing that changes is environmental selection.

Generational adaptation
------------------------
Canonical SMS-EMOA is steady-state: one offspring is generated per
iteration, and if it falls in the last (worst) non-dominated front, exactly
one individual -- the one with the smallest hypervolume contribution to that
front -- is deleted to keep the population size fixed. That one-offspring-
per-iteration scheme is incompatible with this study's equal-decode-budget
harness, which assumes every comparator consumes `population_size` offspring
per generation. This implementation therefore uses the standard generational
adaptation of SMS-EMOA found throughout the MOEA literature: non-dominated
sorting exactly as in NSGA-II, whole fronts are added while they fit, and
the one splitting front that would overflow the population is reduced by
*repeated* least-hypervolume-contribution deletion (one individual removed
at a time, contributions recomputed after each removal) until it fits. This
is the environmental-selection analogue of SMS-EMOA's own rule, applied to a
generational (mu+lambda) loop instead of (mu+1), and only ever touches the
single splitting front (all earlier fronts survive unconditionally, exactly
as in NSGA-II).

Hypervolume contribution
-------------------------
Computed on the splitting front only, objectives min-max normalised to
[0, 1] over that front (same convention as SPEA2's density calculation and
Luan et al. 2023's sparsity normalisation, Eq. 15), with the hypervolume
reference point placed at (1.1, 1.1, 1.1) in normalised space -- the same
`ref_factor=1.1` convention already used by `hypervolume.normalized_hypervolume`
elsewhere in this codebase. Contribution of x is HV(front) - HV(front \\ {x});
the individual with the smallest contribution is removed first.
"""


import copy
import random

from mme import initialize_MME_population
from evaluation import evaluate_population
from genetic import crossover_OS
from mutation import move_to_end_mutation
from nondominated_sort import fast_non_dominated_sort
from hypervolume import hypervolume


# ============================================================
# Common Experimental Parameters
# ============================================================

CROSSOVER_PROBABILITY = 0.9

MUTATION_PROBABILITY = 0.1

OBJECTIVE_NAMES = ["Cmax", "TWET", "NEC"]

# Reference point for hypervolume contribution, in normalised [0, 1] space.
# Matches the ref_factor=1.1 convention in hypervolume.normalized_hypervolume.
_REF_FACTOR = 1.1


# ============================================================
# Hypervolume-contribution front reduction
# ============================================================

def _normalize_front(front):
    """Min-max normalise a front's three objectives to [0, 1]. A degenerate
    (zero-span) objective is treated as already normalised (0 for every
    member), matching `hypervolume.normalized_hypervolume`'s convention."""

    lo = [min(r[name] for r in front) for name in OBJECTIVE_NAMES]
    hi = [max(r[name] for r in front) for name in OBJECTIVE_NAMES]

    span = [
        (h - l) if (h - l) > 1e-9 else 1.0
        for l, h in zip(lo, hi)
    ]

    return {
        id(r): tuple(
            (r[name] - lo[k]) / span[k]
            for k, name in enumerate(OBJECTIVE_NAMES)
        )
        for r in front
    }


# Exact hypervolume-contribution removal is cubic-plus in front size (each
# removal recomputes `hypervolume()` -- itself O(m^2 log m) in the shared
# `hypervolume.py` primitive -- once per surviving candidate), which is the
# well-documented scalability limit of SMS-EMOA's exact indicator and is
# fine for small fronts but becomes impractically slow once a splitting
# front approaches the full combined-population size (2 * population_size):
# measured directly during development, a single generation with a ~150-
# individual splitting front took several minutes under the naive
# all-exact approach. The two-stage reduction below is the standard,
# literature-accepted compromise for this limit: a cheap nearest-neighbour
# crowding pre-cut (no hypervolume geometry, just pairwise distances) does
# the bulk of the reduction, and the expensive exact hypervolume
# contribution is reserved for only the final `_EXACT_HV_PHASE_BUDGET`
# removals, where its precision actually matters (deciding the last few
# borderline survivors). On fronts already within that budget of the
# target size -- the common case once the population has converged --
# this reduces to the original all-exact behaviour unchanged.
_EXACT_HV_PHASE_BUDGET = 20


def _nearest_neighbor_distance(points, index):
    """Euclidean distance from `points[index]` to its nearest other point."""

    pi = points[index]
    best = None

    for j, pj in enumerate(points):
        if j == index:
            continue
        d = sum((pi[k] - pj[k]) ** 2 for k in range(len(pi))) ** 0.5
        if best is None or d < best:
            best = d

    return best


def _crowding_precut(front, target_size):
    """Cheaply reduce `front` to `target_size` by repeatedly removing the
    individual with the smallest nearest-neighbour distance in normalised
    objective space (no hypervolume geometry -- O(1) comparisons per pair
    instead of a full `hypervolume()` call)."""

    remaining = list(front)

    while len(remaining) > target_size:

        normalized = _normalize_front(remaining)
        points = [normalized[id(r)] for r in remaining]

        distances = [
            _nearest_neighbor_distance(points, i) for i in range(len(points))
        ]

        worst_index = min(range(len(remaining)), key=lambda i: distances[i])

        remaining.pop(worst_index)

    return remaining


def reduce_front_by_hv_contribution(front, target_size):
    """
    Repeatedly remove the individual with the smallest hypervolume
    contribution from `front` until only `target_size` remain.

    `front` must already be mutually non-dominated (SMS-EMOA only ever
    reduces the single splitting front, which already satisfies this by
    construction from `fast_non_dominated_sort`). See `_EXACT_HV_PHASE_BUDGET`
    above for the cheap-precut-then-exact two-stage strategy used to keep
    this tractable on large fronts.
    """

    remaining = list(front)

    if len(remaining) - target_size > _EXACT_HV_PHASE_BUDGET:
        remaining = _crowding_precut(
            remaining, target_size + _EXACT_HV_PHASE_BUDGET
        )

    reference = (_REF_FACTOR, _REF_FACTOR, _REF_FACTOR)

    while len(remaining) > target_size:

        normalized = _normalize_front(remaining)
        points = [normalized[id(r)] for r in remaining]

        total_hv = hypervolume(points, reference)

        worst = None
        worst_contribution = None

        for i, r in enumerate(remaining):
            rest = points[:i] + points[i + 1:]
            hv_without = hypervolume(rest, reference)
            contribution = total_hv - hv_without

            if worst_contribution is None or contribution < worst_contribution:
                worst_contribution = contribution
                worst = r

        remaining.remove(worst)

    return remaining


# ============================================================
# SMS-EMOA Environmental Selection
# ============================================================

def sms_emoa_environmental_selection(population, population_size):
    """
    Select the next population from Rt = Pt U Qt.

    Steps:

        1. non-dominated sort;
        2. add whole fronts while they fit;
        3. reduce the splitting front by repeated least-hypervolume-
           contribution deletion.
    """

    if len(population) < population_size:
        raise ValueError(
            "Combined population is smaller than the requested next "
            "population size."
        )

    fronts = fast_non_dominated_sort(population)

    next_population = []

    for front in fronts:

        if len(next_population) + len(front) <= population_size:

            next_population.extend(front)

            if len(next_population) == population_size:
                return next_population

        else:

            remaining = population_size - len(next_population)

            reduced = reduce_front_by_hv_contribution(front, remaining)

            next_population.extend(reduced)

            return next_population

    return next_population


# ============================================================
# Mating Selection (binary tournament on rank, random tie-break)
# ============================================================

def _tournament_winner(a, b):
    if a["rank"] < b["rank"]:
        return a
    if b["rank"] < a["rank"]:
        return b
    return random.choice([a, b])


def build_sms_emoa_mating_pool(population_results, population_size):
    """
    Construct a mating pool of size `population_size` using independent
    binary tournaments on non-domination rank. Assumes `["rank"]` has
    already been assigned (e.g. via `fast_non_dominated_sort`).
    """

    if not population_results:
        raise ValueError(
            "Cannot construct mating pool from an empty population."
        )

    if population_size <= 0:
        raise ValueError("population_size must be greater than zero.")

    if len(population_results) == 1:
        return [
            copy.deepcopy(population_results[0]["chromosome"])
            for _ in range(population_size)
        ]

    mating_pool = []

    while len(mating_pool) < population_size:

        candidate1, candidate2 = random.sample(population_results, 2)

        winner = _tournament_winner(candidate1, candidate2)

        mating_pool.append(winner["chromosome"])

    return mating_pool


# ============================================================
# Generate Offspring Q_t
# ============================================================

def generate_offspring(mating_pool, population_size):
    """Generate offspring Q_t using the same crossover/mutation operators as
    the other comparators."""

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
# Standard SMS-EMOA
# ============================================================

def run_SMS_EMOA(
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
    Run the SMS-EMOA comparator.

    Signature matches run_UNSGAIII / run_NSGAIII / run_NSGAII / run_SPEA2 /
    run_MOEAD so it plugs directly into the existing comparison harness.

    generation_callback, local_search, right_shift:
        Optional hooks with identical semantics to run_UNSGAIII. Defaults
        (None/None/False) reproduce plain SMS-EMOA; supplying a
        ParetoLocalSearch turns this into memetic SMS-EMOA, enabling the
        cross-framework robustness study (a 6th host).

    Returns
    -------
    list
        Final chromosome population (size population_size).
    """

    if population_size <= 0:
        raise ValueError("population_size must be greater than zero.")

    if generations < 0:
        raise ValueError("generations cannot be negative.")

    population = initialize_MME_population(instance, population_size)

    if len(population) != population_size:
        raise RuntimeError(
            "Initialization produced "
            f"{len(population)} individuals; expected {population_size}."
        )

    for generation in range(generations):

        if verbose and (generation == 1 or (
            generation > 1 and generation % 10 == 0
        )):
            print(f"SMS-EMOA Generation {generation}/{generations}")

        parent_results = evaluate_population(
            instance, population, due_dates, weights, machines,
            right_shift=right_shift
        )

        fast_non_dominated_sort(parent_results)  # assigns ["rank"] in place

        if generation_callback is not None:
            generation_callback(generation, parent_results)

        mating_pool = build_sms_emoa_mating_pool(
            parent_results, population_size
        )

        offspring = generate_offspring(mating_pool, population_size)

        # Memetic local search on Q_t (optional; None = baseline).
        if local_search is not None:
            offspring = local_search.improve(
                instance, offspring, due_dates, weights, machines
            )
            if len(offspring) != population_size:
                raise RuntimeError(
                    "Local search changed the SMS-EMOA offspring size: "
                    f"expected {population_size}, got {len(offspring)}."
                )

        offspring_results = evaluate_population(
            instance, offspring, due_dates, weights, machines,
            right_shift=right_shift
        )

        combined = parent_results + offspring_results

        if len(combined) != 2 * population_size:
            raise RuntimeError(
                "Combined population R_t has "
                f"{len(combined)} individuals; "
                f"expected {2 * population_size}."
            )

        selected = sms_emoa_environmental_selection(
            combined, population_size
        )

        population = [solution["chromosome"] for solution in selected]

        if len(population) != population_size:
            raise RuntimeError(
                "Next SMS-EMOA population has "
                f"{len(population)} individuals; "
                f"expected {population_size}."
            )

    return population
