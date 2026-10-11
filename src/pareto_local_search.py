"""
Wei2021 Reproduction — Memetic extension

Pareto Local Search (PLS) driver

Turns the standalone neighborhood operators (currently the N5 makespan move in
`ls_critical_block.py`; later the tardiness and energy moves) into the memetic
local-search step of U-NSGA-III. It is invoked once per generation on the
offspring population Q_t, via the optional `local_search` hook in
`unsga3.run_UNSGAIII`. With no hook installed, the baseline is unchanged.

Design (see `MEMETIC_DESIGN.md` §5)
----------------------------------
* Subset application. Local search is applied only to a small subset of Q_t —
  its non-dominated front plus a few random individuals — not the whole
  population. This bounds the added evaluation cost and protects diversity.

* Dominance acceptance (default). A neighbor replaces the incumbent only if it
  *dominates* it (`pareto.dominates`). First-improvement, capped at `max_moves`
  steps per solution. The acceptance test is injectable so Stage 2 can supply a
  reference-direction / scalar rule; the archiving below always uses Pareto
  dominance regardless.

* Archive. Neighbors that are non-dominated w.r.t. the incumbent (e.g. a move
  that lowers Cmax but trades TWET/NEC) are not accepted but are collected in a
  bounded, Pareto-filtered archive and injected back into Q_t, replacing random
  non-subset offspring. This is what lets a single-objective move (makespan)
  still improve the multi-objective front instead of collapsing it: pure wins
  advance the incumbent, trade-offs enter the population laterally, and NSGA-III
  environmental selection on R_t = P_t ∪ Q_t keeps whatever helps.

Budget note. Every decode performed here is counted in `self.evaluations`, so the
equal-evaluation-budget comparison in the experiment protocol can account for
local-search cost honestly.
"""


import random as _random_module

from evaluation import evaluate_individual

from pareto import dominates, get_pareto_front

from ls_critical_block import critical_block_neighbors


# Number of objectives (Cmax, TWET, NEC) — used for axis-extreme protection.
_NUM_OBJECTIVES = 3


# ============================================================
# Default acceptance: Pareto dominance
# ============================================================

def dominance_accept(current, neighbor):
    """Accept a neighbor iff it dominates the current incumbent."""

    return dominates(neighbor, current)


def cmax_greedy_accept(current, neighbor):
    """Naive single-objective acceptance: accept iff Cmax strictly improves,
    ignoring any effect on TWET/NEC. Represents the classical Nowicki-Smutnicki
    N5 usage (designed for single-objective makespan minimisation) bolted onto
    the multi-objective loop without Pareto-safety. Ablation-only: isolates
    whether the acceptance rule itself, not just the N5 neighbourhood, drives
    the reported gain (component-ablation study)."""

    return neighbor["Cmax"] < current["Cmax"]


# ============================================================
# Helpers
# ============================================================

def _objectives(result):
    return (result["Cmax"], result["TWET"], result["NEC"])


def _same_objectives(a, b):
    return _objectives(a) == _objectives(b)


# ============================================================
# Pareto Local Search
# ============================================================

class ParetoLocalSearch:

    """
    Per-generation memetic local search over the offspring population.

    Parameters
    ----------
    neighborhoods:
        List of callables `f(instance, chromosome) -> list[Chromosome]`. Defaults
        to `[critical_block_neighbors]` (N5 makespan). Later stages append the
        tardiness and energy neighborhoods.

    subset_size:
        Number of offspring the local search is applied to each generation
        (front first, then random fill).

    max_moves:
        Cap on first-improvement hill-climb steps per solution.

    max_archive:
        Cap on the injected, Pareto-filtered archive.

    inject:
        Whether to inject archived trade-off solutions back into Q_t.

    accept:
        Acceptance predicate `accept(current_result, neighbor_result) -> bool`.
        Defaults to Pareto dominance.

    rng:
        Random source (defaults to the global `random` module, so experiment
        seeds set upstream control it, consistent with the rest of the algorithm).
    """

    def __init__(
            self,
            neighborhoods=None,
            subset_size=20,
            max_moves=20,
            max_archive=50,
            inject=True,
            accept=dominance_accept,
            rng=None,
            right_shift=False,
    ):

        self.neighborhoods = (
            neighborhoods
            if neighborhoods is not None
            else [critical_block_neighbors]
        )

        self.subset_size = subset_size

        self.max_moves = max_moves

        self.max_archive = max_archive

        self.inject = inject

        self.accept = accept

        self.rng = rng if rng is not None else _random_module

        # Energy-aware right-shift NEC scoring. Defaults to False (validated
        # semi-active scoring); set True to score the Pareto-safe NEC refinement
        # consistently during search. Must match the surrounding run's setting
        # (see memetic_comparison).
        self.right_shift = right_shift

        # Cumulative decode counter (budget accounting).
        self.evaluations = 0

    # --------------------------------------------------------
    # Public entry point (called from unsga3 hook)
    # --------------------------------------------------------

    def improve(
            self,
            instance,
            population,
            due_dates,
            weights,
            machines,
            reference_points=None,
    ):

        """
        Apply local search to a subset of `population` and return a new
        population of the same size (list of chromosomes).

        `reference_points` is accepted for forward compatibility with the Stage-2
        reference-direction-guided acceptance; it is unused by the default
        dominance acceptance.
        """

        n = len(population)

        if n == 0:
            return population

        # 1. Evaluate the offspring so we can pick a subset by dominance.
        results = [
            evaluate_individual(
                instance, c, due_dates, weights, machines,
                right_shift=self.right_shift,
            )
            for c in population
        ]

        self.evaluations += n

        # 2. Select the subset to improve: non-dominated front + random fill.
        subset_indices = self._select_subset(results)

        subset_lookup = set(subset_indices)

        new_population = list(population)

        archive = []

        # 3. Hill-climb each selected solution.
        for i in subset_indices:

            improved, discovered = self._hill_climb(
                instance,
                results[i],
                due_dates,
                weights,
                machines,
            )

            new_population[i] = improved["chromosome"]

            archive.extend(discovered)

        # 4. Reduce the archive to a bounded, deduplicated Pareto set.
        archive = self._filter_archive(archive)

        # 5. Inject archived trade-offs into random non-subset slots.
        if self.inject and archive:

            non_subset = [
                i for i in range(n) if i not in subset_lookup
            ]

            self.rng.shuffle(non_subset)

            inject_count = min(len(archive), len(non_subset), self.subset_size)

            for slot, archived in zip(non_subset, archive[:inject_count]):

                new_population[slot] = archived["chromosome"]

        return new_population

    # --------------------------------------------------------
    # Subset selection
    # --------------------------------------------------------

    def _select_subset(self, results):

        front = get_pareto_front(results)

        front_ids = {id(r) for r in front}

        subset = [
            i for i, r in enumerate(results) if id(r) in front_ids
        ]

        # Trim an over-large front, but always keep the axis-extreme members so
        # the low-Cmax / low-TWET / low-NEC corners keep receiving local search
        # (front-stretching, MEMETIC_DESIGN.md §5). Random sampling here would
        # let the front collapse toward the ideal and lose the TWET/NEC extremes.
        if len(subset) > self.subset_size:
            subset = self._select_extreme_indices(
                subset, results, self.subset_size
            )

        # Fill up to subset_size with random non-front individuals.
        elif len(subset) < self.subset_size:

            subset_lookup = set(subset)

            remaining = [
                i for i in range(len(results)) if i not in subset_lookup
            ]

            fill = min(self.subset_size - len(subset), len(remaining))

            if fill > 0:
                subset += self.rng.sample(remaining, fill)

        return subset

    # --------------------------------------------------------
    # First-improvement hill-climb
    # --------------------------------------------------------

    def _hill_climb(
            self,
            instance,
            start_result,
            due_dates,
            weights,
            machines,
    ):

        """
        First-improvement local search from one solution.

        Returns (final_result, discovered_non_dominated_neighbors).
        """

        current = start_result

        discovered = []

        for _ in range(self.max_moves):

            neighbors = []

            for neighborhood in self.neighborhoods:

                neighbors.extend(
                    neighborhood(
                        instance,
                        current["chromosome"],
                        due_dates,
                        weights,
                        machines,
                    )
                )

            if not neighbors:
                break

            accepted = None

            for candidate in neighbors:

                result = evaluate_individual(
                    instance, candidate, due_dates, weights, machines,
                    right_shift=self.right_shift,
                )

                self.evaluations += 1

                if self.accept(current, result):

                    accepted = result

                    break  # first improvement

                # Non-dominated w.r.t. the incumbent (a genuine trade-off):
                # keep it for the archive.
                elif (
                    not dominates(current, result)
                    and not _same_objectives(current, result)
                ):
                    discovered.append(result)

            if accepted is None:
                break

            current = accepted

        return current, discovered

    # --------------------------------------------------------
    # Archive reduction
    # --------------------------------------------------------

    def _filter_archive(self, archive):

        # Deduplicate by objective vector.
        seen = set()

        unique = []

        for result in archive:

            key = _objectives(result)

            if key in seen:
                continue

            seen.add(key)

            unique.append(result)

        # Keep only mutually non-dominated members.
        front = get_pareto_front(unique)

        # Order axis-extreme members first so they survive the cap AND are
        # injected into Q_t preferentially — this is what lets a single-objective
        # move push the front's TWET/NEC corners outward instead of only feeding
        # near-ideal trade-offs back in (MEMETIC_DESIGN.md §5, TWET-extremes arm).
        front = self._order_extremes_first(front)

        # Cap (extremes are at the head, so they are retained when max_archive
        # >= number of objectives).
        if len(front) > self.max_archive:
            front = front[: self.max_archive]

        return front

    # --------------------------------------------------------
    # Axis-extreme protection (front-stretching)
    # --------------------------------------------------------

    def _order_extremes_first(self, results):
        """Reorder `results` so the axis-extreme members (minimum in each
        objective) lead, followed by the remaining members in shuffled order.
        Ties are broken by first occurrence."""

        if len(results) <= 1:
            return list(results)

        extreme_ids = set()

        for k in range(_NUM_OBJECTIVES):
            best = min(results, key=lambda r: _objectives(r)[k])
            extreme_ids.add(id(best))

        extremes = [r for r in results if id(r) in extreme_ids]

        rest = [r for r in results if id(r) not in extreme_ids]

        self.rng.shuffle(rest)

        return extremes + rest

    def _select_extreme_indices(self, indices, results, cap):
        """Trim `indices` (into `results`) to `cap`, always retaining the
        axis-extreme members so the objective corners keep being locally
        searched."""

        extreme_indices = set()

        for k in range(_NUM_OBJECTIVES):
            best_i = min(indices, key=lambda i: _objectives(results[i])[k])
            extreme_indices.add(best_i)

        keep = list(extreme_indices)

        if len(keep) >= cap:
            return keep[:cap]

        rest = [i for i in indices if i not in extreme_indices]

        self.rng.shuffle(rest)

        return keep + rest[: cap - len(keep)]
