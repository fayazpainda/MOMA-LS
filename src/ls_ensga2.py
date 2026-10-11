"""
Wei2021 Reproduction — ENSGA-II baseline local search

Reimplements the two neighborhood structures of Luan et al. (2023),
"Enhanced NSGA-II for multi-objective energy-saving flexible job shop
scheduling" (Sustainable Computing: Informatics and Systems 39:100901), that
are applicable to classical (non-flexible) JSP:

    N1: randomly select two elements in the operation sequence string and
        swap their positions (position-exchange move).
    N2: randomly select two elements in the operation sequence string and
        insert the earlier one directly behind the later one (remove-insert
        move).

The paper's third neighborhood structure, N3 (replace the machine assigned
to a randomly selected multi-option operation with another eligible
machine), is a Flexible JSP move: classical JSP has no machine-assignment
decision variable (`chromosome.py`'s machine_sequence is a fixed benchmark
route, not a decision), so N3 has no analogue here and is intentionally not
reimplemented.

Both operators act only on `chromosome.operation_sequence` and reuse the
same OS encoding as `ls_critical_block.py`'s N5 move, so a swap/insert never
changes any job's occurrence count and the result is always feasible
without repair.

Unlike N5 (which deterministically enumerates the small set of critical-path
border pairs), N1/N2 are stochastic per the source paper ("randomly
select... "), so each call draws `num_samples` independent random moves.
This mirrors how the paper's mutation operator is used (each iteration draws
one random move) while still handing the Pareto Local Search driver several
candidate neighbors per hill-climb step, consistent with the uniform
neighborhood interface `f(instance, chromosome, due_dates, weights,
machines) -> list[Chromosome]` shared with `critical_block_neighbors`.
"""


import random as _random_module

from chromosome import Chromosome


# Default number of random neighbors drawn per call, per operator. Matched to
# N5's typical per-call neighbor count (2-6 border pairs) so the hill-climb
# loop sees a comparable branching factor per step across operators.
_DEFAULT_NUM_SAMPLES = 4


# ============================================================
# N1: position-exchange (swap) mutation
# ============================================================

def _swap_positions(operation_sequence, i, j, rng):

    new_sequence = operation_sequence.copy()

    new_sequence[i], new_sequence[j] = new_sequence[j], new_sequence[i]

    return new_sequence


def ensga2_n1_neighbors(
        instance, chromosome,
        due_dates=None, weights=None, machines=None,
        num_samples=_DEFAULT_NUM_SAMPLES, rng=None,
):
    """
    N1 neighborhood (Luan et al. 2023, Sec. 3.7): randomly choose two
    positions e1, e2 in the operation sequence and exchange their values.

    Draws `num_samples` independent random swaps and returns one neighbor
    chromosome per draw. `due_dates`/`weights`/`machines` are accepted for a
    uniform neighborhood interface but unused (a swap needs only the OS
    string).
    """

    rng = rng if rng is not None else _random_module

    length = len(chromosome.operation_sequence)

    if length < 2:
        return []

    neighbors = []

    for _ in range(num_samples):

        i, j = rng.sample(range(length), 2)

        new_sequence = _swap_positions(
            chromosome.operation_sequence, i, j, rng
        )

        neighbors.append(
            Chromosome(
                operation_sequence=new_sequence,
                machine_sequence=chromosome.machine_sequence.copy(),
            )
        )

    return neighbors


# ============================================================
# N2: remove-insert mutation
# ============================================================

def _insert_position(operation_sequence, i, j):

    """Remove the gene at index i and reinsert it immediately behind index
    j (per the paper: "insert the element of e2 into the position in front
    of e1" — realized here as moving the earlier-drawn index behind the
    later-drawn one, matching the operator's intent of relocating one gene
    relative to another without disturbing the rest of the order)."""

    new_sequence = operation_sequence.copy()

    gene = new_sequence.pop(i)

    # After popping index i, an index j that was after i shifts left by one.
    target = j - 1 if j > i else j

    new_sequence.insert(target + 1, gene)

    return new_sequence


def ensga2_n2_neighbors(
        instance, chromosome,
        due_dates=None, weights=None, machines=None,
        num_samples=_DEFAULT_NUM_SAMPLES, rng=None,
):
    """
    N2 neighborhood (Luan et al. 2023, Sec. 3.7): randomly choose two
    positions e1, e2 in the operation sequence and relocate the gene at e1
    to immediately follow the gene at e2.

    Draws `num_samples` independent random relocations and returns one
    neighbor chromosome per draw.
    """

    rng = rng if rng is not None else _random_module

    length = len(chromosome.operation_sequence)

    if length < 2:
        return []

    neighbors = []

    for _ in range(num_samples):

        i, j = rng.sample(range(length), 2)

        new_sequence = _insert_position(
            chromosome.operation_sequence, i, j
        )

        neighbors.append(
            Chromosome(
                operation_sequence=new_sequence,
                machine_sequence=chromosome.machine_sequence.copy(),
            )
        )

    return neighbors
