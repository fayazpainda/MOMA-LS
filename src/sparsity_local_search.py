"""
Wei2021 Reproduction — ENSGA-II baseline local search driver

Reimplements the sparsity-theory subset-selection rule of Luan et al.
(2023), "Enhanced NSGA-II for multi-objective energy-saving flexible job
shop scheduling" (Sustainable Computing: Informatics and Systems 39:100901),
Sec. 3.8(1), Eqs. (15)-(17):

    f_bar(x) = (f(x) - f_min) / (f_max - f_min)        normalize each
                                                        objective over the
                                                        current population
    rho(x_i, x_j) = || f_bar(x_i) - f_bar(x_j) ||_2     Euclidean distance
                                                        in normalized
                                                        objective space
    SP(x_i) = z_i / N                                   z_i = count of other
                                                        individuals with
                                                        rho < judgment
                                                        threshold k (0<k<1)

Individuals with the smallest SP(x_i) sit in the least-crowded region of
objective space ("sparse solutions"); the source paper directs local search
at these individuals instead of the non-dominated front. This is the one
structural difference between this baseline and MOMA-LS's own subset rule
(`pareto_local_search.py::_select_subset`, which targets the front plus
axis-extremes): MOMA-LS concentrates search on quality (the front),
ENSGA-II concentrates it on diversity (under-covered regions of objective
space), regardless of dominance rank.

`k=0.2` is the source paper's own tuned value (its Table 4 Taguchi
parameter study, MK01 instance: judgment threshold ranked 2nd by influence
of its five parameters, with 0.2 the best-response level among the four
levels tested {0.1, 0.15, 0.2, 0.25}).

Everything else (neighborhood application, hill-climb, acceptance,
archiving/injection) is inherited unchanged from `ParetoLocalSearch` so the
two baselines are compared on equal footing: same evaluation-budget
accounting, same Pareto-dominance acceptance, same archive/injection
mechanics. Only the neighborhoods (`ls_ensga2.py`'s N1/N2, supplied by the
caller) and the subset-selection rule below actually differ.
"""


from pareto_local_search import ParetoLocalSearch


def _objectives(result):
    return (result["Cmax"], result["TWET"], result["NEC"])


class SparsityLocalSearch(ParetoLocalSearch):
    """
    ENSGA-II-style Pareto Local Search: identical driver to
    `ParetoLocalSearch`, but selects the per-generation subset by sparsity
    (Eqs. 15-17) instead of non-dominated front + random fill.

    Parameters
    ----------
    judgment_threshold:
        `k` in Eq. (17), the normalized-distance radius under which two
        individuals count as neighbors. Defaults to 0.2 (Luan et al. 2023's
        own tuned value). Must satisfy 0 < k < 1.

    See `ParetoLocalSearch` for all other parameters (neighborhoods,
    subset_size, max_moves, max_archive, inject, accept, rng, right_shift).
    """

    def __init__(self, judgment_threshold=0.2, **kwargs):

        super().__init__(**kwargs)

        if not (0.0 < judgment_threshold < 1.0):
            raise ValueError("judgment_threshold (k) must satisfy 0 < k < 1")

        self.judgment_threshold = judgment_threshold

    # --------------------------------------------------------
    # Sparsity-based subset selection (overrides front-based selection)
    # --------------------------------------------------------

    def _select_subset(self, results):

        n = len(results)

        if n <= self.subset_size:
            return list(range(n))

        sparsity = self._sparsity_scores(results)

        # Smallest SP(x_i) first: the sparsest (least-crowded) individuals.
        ranked = sorted(range(n), key=lambda i: sparsity[i])

        return ranked[: self.subset_size]

    def _sparsity_scores(self, results):

        n = len(results)

        raw = [_objectives(r) for r in results]

        num_objectives = len(raw[0])

        # Eq. (15): per-objective min-max normalization over the population.
        mins = [min(r[k] for r in raw) for k in range(num_objectives)]
        maxs = [max(r[k] for r in raw) for k in range(num_objectives)]

        spans = [
            (maxs[k] - mins[k]) if maxs[k] > mins[k] else 1.0
            for k in range(num_objectives)
        ]

        normalized = [
            tuple(
                (raw[i][k] - mins[k]) / spans[k]
                for k in range(num_objectives)
            )
            for i in range(n)
        ]

        # Eq. (16)-(17): count neighbors within the judgment threshold,
        # divide by population size.
        sparsity = [0] * n

        for i in range(n):

            count = 0

            for j in range(n):

                if i == j:
                    continue

                distance = sum(
                    (normalized[i][k] - normalized[j][k]) ** 2
                    for k in range(num_objectives)
                ) ** 0.5

                if distance < self.judgment_threshold:
                    count += 1

            sparsity[i] = count / n

        return sparsity
