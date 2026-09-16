"""
Metrics used across experiments.

LOWER BOUND
-----------
A simple, provably valid lower bound on C_max: for each job, sum the
MINIMUM possible processing time across eligible machines for each of its
operations (i.e. pretend every operation gets its best-case machine with
zero waiting). That sum is a lower bound on that job's completion time,
since precedence means the job can't finish faster even in the best case.
The lower bound on C_max is the max of this over all jobs -- because
C_max >= C_j for every job.

This bound IGNORES machine contention entirely (two jobs could both want
their fastest machine at the same time), so it is usually loose, but it's
cheap, always valid, and gives a sense of how much slack the algorithm is
leaving on the table.

MACHINE UTILIZATION
--------------------
For a valid schedule, utilization of machine m = (total busy time on m) / makespan.
Low utilization on a machine that's frequently eligible suggests the
algorithm isn't exploiting it; utilization near 1.0 on one specific machine
while others sit idle is the signature of a bottleneck.
"""

import os
import sys
from typing import Dict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from generator.model import Instance


def job_lower_bound(instance: Instance) -> int:
    """LB1 -- critical-path / precedence bound (the original bound).

    For each job, sum the minimum processing time of each of its operations.
    Valid because a job's operations form a strict chain, so even with zero
    waiting and best-case machines the job cannot finish sooner. C_max >= C_j
    for every j, so the max over jobs is a lower bound on C_max.
    """
    best = 0
    for job in instance.jobs:
        job_lb = sum(min(op.proc_times.values()) for op in job.operations)
        best = max(best, job_lb)
    return best


def machine_load_lower_bound(instance: Instance) -> int:
    """LB2 -- work-content / machine-load bound.

    Every operation must be processed by exactly one machine, and it occupies
    that machine for at least its fastest eligible time. So the TOTAL work any
    schedule must perform is at least sum over all operations of min p(j,k,m).
    That work is spread over at most `num_machines` machines running in
    parallel, so at least one machine is busy for >= total_work / m time units,
    giving C_max >= ceil(total_work / m).

    Valid for the same reason a pigeonhole argument is: it assumes the most
    optimistic possible assignment (every operation on its fastest machine)
    AND perfectly even load balancing AND zero idle time. No real schedule can
    beat all three simultaneously.
    """
    total_min_work = sum(
        min(op.proc_times.values())
        for job in instance.jobs
        for op in job.operations
    )
    m = max(1, instance.num_machines)
    return -(-total_min_work // m)  # ceiling division


def lower_bound(instance: Instance) -> int:
    """Combined lower bound: max(LB1, LB2).

    Both components are independently valid lower bounds, so their maximum is
    also valid and is never weaker than either alone.

    Why this matters: LB1 alone ignores machine contention entirely. On
    machine-constrained instances (few machines, many jobs) it is badly loose
    -- e.g. a single-machine instance where every job is short has a tiny LB1
    but a provably large optimum. LB2 captures exactly that contention and
    fixes the reported gap% on those instances. Conversely LB1 dominates on
    long-chain instances where one job's precedence chain, not machine
    capacity, is what forces the makespan. Neither bound dominates the other,
    which is why both are computed.
    """
    return max(job_lower_bound(instance), machine_load_lower_bound(instance))


def machine_utilization(instance: Instance, validation_result) -> Dict[int, float]:
    makespan = validation_result.makespan
    util = {}
    for m in range(1, instance.num_machines + 1):
        ops = validation_result.machine_schedules.get(m, [])
        busy = sum(o.completion_time - o.start_time for o in ops)
        util[m] = (busy / makespan) if makespan > 0 else 0.0
    return util


def critical_job(validation_result) -> int:
    """Job with the largest completion time -- the one determining the makespan."""
    return max(validation_result.job_completion_times, key=validation_result.job_completion_times.get)