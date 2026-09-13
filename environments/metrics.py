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


def lower_bound(instance: Instance) -> int:
    best = 0
    for job in instance.jobs:
        job_lb = sum(min(op.proc_times.values()) for op in job.operations)
        best = max(best, job_lb)
    return best


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