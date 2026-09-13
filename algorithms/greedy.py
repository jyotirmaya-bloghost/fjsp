"""
Global Earliest-Completion-Time (ECT) dispatching heuristic for FJSP.

CORE IDEA
---------
At any point in the simulation, each job has at most one "ready" operation:
the first operation of that job which hasn't been scheduled yet, PROVIDED
its predecessor (if any) has already been scheduled. (Job precedence means
a job can never have two ready operations at once.)

For every ready operation, and for every machine it's eligible for, compute
the earliest possible start/completion time on that machine:
    start = max(job_ready_time[job], machine_free_time[machine])
    finish = start + p(job, op, machine)

Among ALL (ready operation, eligible machine) pairs across ALL jobs, pick
the one with the smallest `finish` time. Commit it: update the job's ready
time and the machine's free time. Repeat until every operation is scheduled.

This makes the routing decision (which machine) and the sequencing decision
(which operation goes next) TOGETHER, greedily, rather than fixing an
operation order first and only then picking machines.

REPRESENTATION
--------------
State = (job_ready_time: dict[job_id -> float], machine_free_time: dict[machine_id -> float],
          next_op_index: dict[job_id -> int])
A solution is the sequence of (job, op, machine, start, finish) decisions made,
in the order they were committed.

SEARCH SPACE
------------
This is NOT a search algorithm -- it makes one irrevocable greedy decision
per step, no backtracking. So "search space" here really means the branching
factor at each step: at most n_jobs ready operations, times up to m eligible
machines each. There is no exploration beyond that single greedy choice.

OBJECTIVE
---------
Minimize C_max (makespan), implicitly, by always advancing the
currently-most-promising (operation, machine) pair. This is a myopic
(one-step-lookahead) proxy for the real objective -- it does NOT guarantee
optimality, and its failure modes are exactly what Part D's failure
analysis should probe.

COMPLEXITY
----------
Let N = total number of operations, M = number of machines.
Each of the N iterations scans up to n_jobs ready operations, each with up
to M eligible machines -> O(N * n_jobs * M) worst case. For typical
instances (n_jobs << N), this is fast and easily handles hundreds of
operations.

WHAT THIS IS NOT
-----------------
This is a pure greedy dispatching rule -- not a metaheuristic. There's no
paper being followed here; it's a standard baseline construction heuristic
for FJSP (see e.g. the "global selection" dispatching family described in
Dauzère-Pérès et al.'s FJSP review). It is deliberately simple so its
failures are structurally traceable -- that traceability is the point.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from generator.model import Instance


@dataclass
class Decision:
    job_id: int
    op_id: int
    machine_id: int
    start_time: int
    completion_time: int

    def to_schedule_entry(self) -> dict:
        return {
            "job_id": self.job_id,
            "operation_id": self.op_id,
            "machine_id": self.machine_id,
            "start_time": self.start_time,
            "completion_time": self.completion_time,
        }


def schedule_global_ect(instance: Instance) -> List[dict]:
    """Returns a schedule as a list of dicts, ready to hand straight to validate_schedule()."""

    job_ready_time: Dict[int, int] = {job.job_id: 0 for job in instance.jobs}
    machine_free_time: Dict[int, int] = {m: 0 for m in range(1, instance.num_machines + 1)}
    next_op_index: Dict[int, int] = {job.job_id: 0 for job in instance.jobs}
    jobs_by_id = {job.job_id: job for job in instance.jobs}

    total_ops = instance.total_operations()
    decisions: List[Decision] = []

    for _ in range(total_ops):
        best = None  # (finish_time, start_time, job_id, op_id, machine_id)

        for job in instance.jobs:
            idx = next_op_index[job.job_id]
            if idx >= len(job.operations):
                continue  # this job is fully scheduled
            op = job.operations[idx]
            ready_time = job_ready_time[job.job_id]

            for m in op.eligible_machines:
                start = max(ready_time, machine_free_time[m])
                finish = start + op.proc_times[m]
                candidate = (finish, start, job.job_id, op.op_id, m)
                if best is None or candidate < best:
                    best = candidate

        finish, start, job_id, op_id, machine_id = best
        decisions.append(Decision(job_id, op_id, machine_id, start, finish))

        job_ready_time[job_id] = finish
        machine_free_time[machine_id] = finish
        next_op_index[job_id] += 1

    return [d.to_schedule_entry() for d in decisions]