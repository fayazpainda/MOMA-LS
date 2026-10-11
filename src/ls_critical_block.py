"""
Wei2021 Reproduction — Memetic extension

N5 Critical-Block Local Search (makespan)

Generates the N5 neighborhood of Nowicki & Smutnicki (1996,
`papers/N5_1996...pdf`) for a chromosome: the small, makespan-oriented set of
moves that reverse adjacent operations only at the *borders* of critical blocks.

N5 move rule
------------
Let the makespan critical path decompose into blocks `B_1, ..., B_r`
(`critical_path.critical_path_and_blocks`). The N5 neighborhood is:

    * First block  B_1        : swap only its LAST two operations.
    * Internal blocks B_2..B_{r-1}: swap its FIRST two AND its LAST two operations.
    * Last block   B_r        : swap only its FIRST two operations.
    * A block contributes a move only if it holds >= 2 operations.
    * The neighborhood is empty when r <= 1 (a single block, or a critical path
      lying entirely on one machine / within one job) — such a schedule is
      N5-optimal.

Nowicki & Smutnicki prove that (a) every N5 move yields a feasible schedule and
(b) reversing a *non-border* adjacent critical pair can never reduce the makespan,
which is why only the block borders are tried. The result is a neighborhood far
smaller than "all adjacent swaps on critical arcs", so local search is cheap.

Realizing a move on the operation-based encoding
------------------------------------------------
A move is defined on the decoded schedule as "reverse the processing order of two
adjacent same-machine operations u, v". On the OS chromosome we realize it by
swapping the two genes that decode to u and v (the k-th occurrence of a job is its
k-th operation, so each operation maps to one gene position). Swapping two gene
values never changes any job's occurrence count, so the child is always feasible
and needs no repair.

Caveat (documented on purpose): the gene swap reproduces the arc reversal exactly
when no other occurrence of either job lies between the two genes; otherwise it is
a slightly broader move. This is harmless to the search because the memetic driver
accepts a neighbor only when it does not worsen the solution (Pareto Local Search,
`pareto_local_search.py`) — an over-general move that happens to help is kept, one
that hurts is rejected. This matches common OS-encoded JSP local-search practice.

This module only *builds* neighbor chromosomes; evaluation and acceptance belong
to the local-search driver.
"""


from chromosome import Chromosome

from decoder import decode

from critical_path import critical_path_and_blocks


# ============================================================
# Operation -> OS position map
# ============================================================

def operation_positions(chromosome):
    """
    Map each operation to its gene index in the operation sequence.

    Returns
    -------
    dict job_id -> list[int]
        `positions[job_id][operation_id]` is the index in
        `chromosome.operation_sequence` of that operation (the
        (operation_id+1)-th occurrence of the job).
    """

    positions = {}

    for index, job_id in enumerate(chromosome.operation_sequence):

        positions.setdefault(job_id, []).append(index)

    return positions


# ============================================================
# Realize one swap
# ============================================================

def swap_operations(chromosome, op_a, op_b, positions=None):
    """
    Return a new chromosome with the genes decoding to `op_a` and `op_b`
    swapped in the operation sequence.

    `positions` may be supplied (from `operation_positions`) to avoid rebuilding
    the map for every move.
    """

    if positions is None:
        positions = operation_positions(chromosome)

    pa = positions[op_a.job_id][op_a.operation_id]

    pb = positions[op_b.job_id][op_b.operation_id]

    new_sequence = chromosome.operation_sequence.copy()

    new_sequence[pa], new_sequence[pb] = (
        new_sequence[pb],
        new_sequence[pa],
    )

    return Chromosome(
        operation_sequence=new_sequence,
        machine_sequence=chromosome.machine_sequence.copy(),
    )


# ============================================================
# N5 border-move enumeration
# ============================================================

def n5_move_pairs(blocks):
    """
    Enumerate the N5 border swaps for a block decomposition.

    Returns
    -------
    list[tuple(ScheduledOperation, ScheduledOperation)]
        Adjacent operation pairs to swap, per the N5 rule (see module docstring).
        Duplicate pairs (e.g. an internal block of size 2, whose first-two and
        last-two coincide) are removed while preserving order.
    """

    r = len(blocks)

    # A single block (or none) is N5-optimal: no border moves exist.
    if r <= 1:
        return []

    pairs = []

    for block_index, block in enumerate(blocks):

        if len(block) < 2:
            continue

        is_first = (block_index == 0)

        is_last = (block_index == r - 1)

        # First block: swap only the last two.
        if is_first:
            pairs.append((block[-2], block[-1]))

        # Last block: swap only the first two.
        elif is_last:
            pairs.append((block[0], block[1]))

        # Internal block: swap first two AND last two.
        else:
            pairs.append((block[0], block[1]))
            pairs.append((block[-2], block[-1]))

    # De-duplicate by operation identity, preserving order.
    seen = set()

    unique_pairs = []

    for a, b in pairs:

        key = (id(a), id(b))

        if key in seen:
            continue

        seen.add(key)

        unique_pairs.append((a, b))

    return unique_pairs


# ============================================================
# Neighborhood
# ============================================================

def critical_block_neighbors(
        instance, chromosome,
        due_dates=None, weights=None, machines=None,
):
    """
    Build the N5 neighborhood of `chromosome`.

    Decodes the chromosome, extracts the makespan critical path and its blocks,
    and returns one neighbor chromosome per N5 border swap.

    `due_dates`, `weights`, `machines` are accepted for a uniform neighborhood
    interface (shared with the tardiness / energy operators) but unused here —
    the makespan move needs only the schedule.

    Returns
    -------
    list[Chromosome]
        May be empty (an N5-optimal schedule, r <= 1).
    """

    schedule = decode(instance, chromosome)

    _, blocks = critical_path_and_blocks(schedule)

    pairs = n5_move_pairs(blocks)

    if not pairs:
        return []

    positions = operation_positions(chromosome)

    neighbors = [
        swap_operations(chromosome, a, b, positions)
        for a, b in pairs
    ]

    return neighbors
