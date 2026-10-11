"""
Wei2021 Reproduction — Memetic extension

Critical Path and Critical Block Extraction

This module recovers the *critical path* of a decoded schedule and decomposes it
into *critical blocks*. It is the shared engine for the makespan local search
(N5, `ls_critical_block.py`, traced back from the makespan operation) and — later
— for the tardiness local search (CET, traced back from each tardy job's
completion operation; Kuhpfahl & Bierwirth 2016).

Background (Nowicki & Smutnicki 1996, `papers/N5_1996...pdf`)
------------------------------------------------------------
In the disjunctive-graph view of a JSP schedule every operation has at most two
predecessors:

    * its JOB predecessor    — the previous operation of the same job, and
    * its MACHINE predecessor — the operation processed immediately before it on
                                the same machine.

An operation's start time equals `max(job_pred.finish, machine_pred.finish)`
(the decoder in `decoder.py` computes exactly this). A predecessor is *tight* when
its finish time equals the operation's start time — i.e. it is the reason the
operation could not start earlier. Following tight predecessors backwards from a
sink operation traces a *longest (critical) path*: any single tight predecessor at
each step is on a longest path to the sink.

A *critical block* is a maximal run of consecutive critical-path operations that
are processed on the SAME machine. Because two consecutive operations of one job
always run on different machines (standard JSP assumption), consecutive path
operations share a machine iff the arc between them is a machine arc; so grouping
the path by "same machine as the previous path operation" yields the blocks.

This module reads a `Schedule` (from `decoder.decode`) and does not modify it.
"""


# ============================================================
# Tight-comparison tolerance
#
# Processing times are integers, but start/finish times are accumulated as
# floats. Times are integer-valued in practice; compare with a small tolerance
# to be safe against float representation.
# ============================================================

_EPS = 1e-9


def _tight(predecessor, operation):
    """Return True if `predecessor` is a tight predecessor of `operation`
    (its finish coincides with the operation's start)."""

    if predecessor is None:
        return False

    return abs(predecessor.finish - operation.start) < _EPS


# ============================================================
# Predecessor maps
# ============================================================

def _build_predecessor_maps(schedule):
    """
    Build the job- and machine-predecessor lookups for a schedule.

    Returns
    -------
    (op_by_job_op, machine_pred)

    op_by_job_op:
        dict (job_id, operation_id) -> ScheduledOperation.

    machine_pred:
        dict id(ScheduledOperation) -> ScheduledOperation or None,
        the operation processed immediately before it on the same machine.
    """

    # (job_id, operation_id) -> ScheduledOperation
    op_by_job_op = {}

    # machine_id -> list of that machine's ScheduledOperations
    ops_on_machine = {}

    for op in schedule.operations:

        op_by_job_op[(op.job_id, op.operation_id)] = op

        ops_on_machine.setdefault(op.machine, []).append(op)

    # Machine predecessor = previous operation on the same machine, ordered by
    # start time. On any machine, operations do not overlap, so start order is
    # well defined.
    machine_pred = {}

    for machine_id, ops in ops_on_machine.items():

        ordered = sorted(ops, key=lambda o: o.start)

        previous = None

        for op in ordered:

            machine_pred[id(op)] = previous

            previous = op

    return op_by_job_op, machine_pred


def _job_predecessor(op, op_by_job_op):
    """Previous operation of the same job, or None for the job's first op."""

    if op.operation_id == 0:
        return None

    return op_by_job_op.get((op.job_id, op.operation_id - 1))


# ============================================================
# Critical path back to a given sink
# ============================================================

def critical_path_to(schedule, sink):
    """
    Trace one critical (longest) path ending at `sink`.

    Starting from `sink`, repeatedly step to a *tight* predecessor until an
    operation that starts at time 0 (no tight predecessor) is reached. When both
    the machine and job predecessors are tight, the MACHINE predecessor is
    preferred: this extends same-machine runs and therefore tends to produce
    longer critical blocks, which is what the N5 neighborhood operates on. Any
    tight predecessor lies on a longest path, so either choice is a valid
    critical path; the preference only affects which one is returned.

    Parameters
    ----------
    schedule:
        Decoded `Schedule` (see `decoder.decode`).

    sink:
        The `ScheduledOperation` to trace back from (e.g. an operation whose
        finish equals the makespan, or a job's last operation for tardiness).

    Returns
    -------
    list[ScheduledOperation]
        Operations ordered from the source end to `sink`.
    """

    op_by_job_op, machine_pred = _build_predecessor_maps(schedule)

    path = [sink]

    current = sink

    while True:

        mp = machine_pred.get(id(current))

        jp = _job_predecessor(current, op_by_job_op)

        mp_tight = _tight(mp, current)

        jp_tight = _tight(jp, current)

        if mp_tight:
            # Prefer the machine predecessor (longer blocks).
            current = mp

        elif jp_tight:
            current = jp

        else:
            # No tight predecessor: current.start == 0, source reached.
            break

        path.append(current)

    path.reverse()

    return path


# ============================================================
# Makespan critical path
# ============================================================

def makespan_sink(schedule):
    """
    Return an operation whose finish equals the schedule makespan.

    Ties (several operations finishing at the makespan) are broken
    deterministically by taking the last such operation in decode order, so the
    extracted critical path is reproducible.
    """

    sink = None

    for op in schedule.operations:

        if abs(op.finish - schedule.makespan) < _EPS:
            sink = op

    return sink


def critical_path(schedule):
    """
    Trace one makespan critical path (source -> makespan operation).

    Returns an empty list for an empty schedule.
    """

    if not schedule.operations:
        return []

    sink = makespan_sink(schedule)

    if sink is None:
        return []

    return critical_path_to(schedule, sink)


# ============================================================
# Block decomposition
# ============================================================

def decompose_blocks(path):
    """
    Decompose a critical path into critical blocks.

    A block is a maximal run of consecutive path operations processed on the same
    machine. Returns a list of blocks, each a list of `ScheduledOperation`
    preserving path order.
    """

    blocks = []

    current_block = []

    for op in path:

        if current_block and op.machine == current_block[-1].machine:

            current_block.append(op)

        else:

            if current_block:
                blocks.append(current_block)

            current_block = [op]

    if current_block:
        blocks.append(current_block)

    return blocks


def critical_path_and_blocks(schedule):
    """
    Convenience wrapper: makespan critical path plus its block decomposition.

    Returns
    -------
    (path, blocks)
    """

    path = critical_path(schedule)

    blocks = decompose_blocks(path)

    return path, blocks
