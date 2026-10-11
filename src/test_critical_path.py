"""Critical-path extraction on a hand-checked schedule.

The schedule is built directly (independent of the decoder) so the expected
critical path and block decomposition can be verified by hand:

    X = job0 op0  on machine 0,  [0, 3]
    Y = job0 op1  on machine 1,  [3, 5]   (job successor of X; X is tight)
    Z = job1 op0  on machine 1,  [5, 8]   (machine successor of Y; Y is tight)

Makespan = 8. Back from Z: machine-pred Y is tight -> Y; job-pred X is tight -> X;
X starts at 0. Critical path = [X, Y, Z]. Blocks by machine = [[X], [Y, Z]].
"""

from decoder import ScheduledOperation, Schedule
from critical_path import (
    critical_path,
    critical_path_and_blocks,
    decompose_blocks,
    makespan_sink,
)
from ls_critical_block import n5_move_pairs


def _schedule():
    X = ScheduledOperation(job_id=0, operation_id=0, machine=0, start=0.0, finish=3.0)
    Y = ScheduledOperation(job_id=0, operation_id=1, machine=1, start=3.0, finish=5.0)
    Z = ScheduledOperation(job_id=1, operation_id=0, machine=1, start=5.0, finish=8.0)
    return Schedule(operations=[X, Y, Z], makespan=8.0), X, Y, Z


def test_makespan_sink_is_last_finishing_op():
    schedule, X, Y, Z = _schedule()
    assert makespan_sink(schedule) is Z


def test_critical_path_follows_tight_predecessors():
    schedule, X, Y, Z = _schedule()
    path = critical_path(schedule)
    assert path == [X, Y, Z]


def test_blocks_group_by_machine():
    schedule, X, Y, Z = _schedule()
    path, blocks = critical_path_and_blocks(schedule)
    assert blocks == [[X], [Y, Z]]


def test_empty_schedule_has_no_path():
    empty = Schedule(operations=[], makespan=0.0)
    assert critical_path(empty) == []
    assert decompose_blocks([]) == []


def test_n5_pairs_on_hand_checked_blocks():
    schedule, X, Y, Z = _schedule()
    _, blocks = critical_path_and_blocks(schedule)
    # r = 2: first block [X] too small; last block [Y, Z] -> swap first two.
    assert n5_move_pairs(blocks) == [(Y, Z)]
