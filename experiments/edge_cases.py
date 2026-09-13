"""
Hand-built edge cases (Part D requirement -- not relying on random generation).
Covers: single machine, one job, bottleneck machine, extreme time gaps,
machine advantage, near-total/near-zero flexibility, identical processing
times, long critical chains, many very short operations.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generator.model import Instance, Job, Operation
from algorithms.greedy import schedule_global_ect
from environments.validator import validate_schedule
from environments.metrics import lower_bound, machine_utilization


def make_single_machine():
    # Everything must run on the one machine -- makespan = sum of all processing times.
    jobs = []
    for j in range(1, 4):
        ops = [Operation(j, k, [1], {1: 3 + k}) for k in range(1, 4)]
        jobs.append(Job(j, ops))
    return Instance(1, jobs, seed=0, params={}, instance_class="single_machine")


def make_one_job():
    # A single job, several machines -- no contention possible at all.
    ops = [
        Operation(1, 1, [1, 2, 3], {1: 5, 2: 7, 3: 6}),
        Operation(1, 2, [1, 2], {1: 4, 2: 3}),
        Operation(1, 3, [2, 3], {2: 8, 3: 5}),
    ]
    return Instance(3, [Job(1, ops)], seed=0, params={}, instance_class="one_job")


def make_bottleneck_machine():
    # Every operation of every job CAN use machine 1, and it's fastest there --
    # forces heavy contention on one machine.
    jobs = []
    for j in range(1, 6):
        ops = [
            Operation(j, 1, [1, 2], {1: 3, 2: 10}),
            Operation(j, 2, [1, 3], {1: 3, 3: 10}),
        ]
        jobs.append(Job(j, ops))
    return Instance(3, jobs, seed=0, params={}, instance_class="bottleneck_machine")


def make_extreme_time_gap():
    # M1 -> 1, M2 -> 10000 for the same operation: an obviously bad choice exists.
    jobs = []
    for j in range(1, 4):
        ops = [Operation(j, 1, [1, 2], {1: 1, 2: 10000})]
        jobs.append(Job(j, ops))
    return Instance(2, jobs, seed=0, params={}, instance_class="extreme_time_gap")


def make_machine_advantage():
    # Machine 1 is dramatically faster at everything it's eligible for.
    jobs = []
    for j in range(1, 5):
        ops = [Operation(j, 1, [1, 2, 3], {1: 2, 2: 15, 3: 15})]
        jobs.append(Job(j, ops))
    return Instance(3, jobs, seed=0, params={}, instance_class="machine_advantage_edge")


def make_near_total_flexibility():
    # Every operation eligible on every machine.
    jobs = []
    machines = list(range(1, 6))
    for j in range(1, 5):
        ops = [Operation(j, k, machines, {m: 5 + (m + k) % 4 for m in machines}) for k in range(1, 3)]
        jobs.append(Job(j, ops))
    return Instance(5, jobs, seed=0, params={}, instance_class="near_total_flexibility")


def make_near_zero_flexibility():
    # Every operation eligible on exactly one machine -- pure sequencing problem.
    jobs = []
    for j in range(1, 5):
        m = ((j - 1) % 3) + 1
        ops = [Operation(j, 1, [m], {m: 6}), Operation(j, 2, [(m % 3) + 1], {(m % 3) + 1: 4})]
        jobs.append(Job(j, ops))
    return Instance(3, jobs, seed=0, params={}, instance_class="near_zero_flexibility")


def make_identical_times():
    # Every operation takes exactly the same time everywhere -- flexibility gives
    # no routing advantage, only ordering matters.
    jobs = []
    for j in range(1, 5):
        ops = [Operation(j, k, [1, 2], {1: 5, 2: 5}) for k in range(1, 3)]
        jobs.append(Job(j, ops))
    return Instance(2, jobs, seed=0, params={}, instance_class="identical_times")


def make_long_critical_chain():
    # One job with a very long operation chain, dwarfing all others.
    jobs = []
    long_ops = [Operation(1, k, [1, 2], {1: 4, 2: 5}) for k in range(1, 13)]
    jobs.append(Job(1, long_ops))
    for j in range(2, 4):
        ops = [Operation(j, 1, [1, 2], {1: 3, 2: 3})]
        jobs.append(Job(j, ops))
    return Instance(2, jobs, seed=0, params={}, instance_class="long_critical_chain")


def make_many_short_ops():
    # Many jobs, each with many very short (1-2 unit) operations.
    jobs = []
    for j in range(1, 11):
        ops = [Operation(j, k, [1, 2, 3], {1: 1, 2: 2, 3: 1}) for k in range(1, 6)]
        jobs.append(Job(j, ops))
    return Instance(3, jobs, seed=0, params={}, instance_class="many_short_ops")


EDGE_CASES = {
    "single_machine": make_single_machine,
    "one_job": make_one_job,
    "bottleneck_machine": make_bottleneck_machine,
    "extreme_time_gap": make_extreme_time_gap,
    "machine_advantage_edge": make_machine_advantage,
    "near_total_flexibility": make_near_total_flexibility,
    "near_zero_flexibility": make_near_zero_flexibility,
    "identical_times": make_identical_times,
    "long_critical_chain": make_long_critical_chain,
    "many_short_ops": make_many_short_ops,
}


def main():
    print(f"{'edge case':24s} | {'jobs':>4s} {'ops':>4s} | {'makespan':>8s} | {'LB':>6s} | {'gap%':>6s} | result")
    print("-" * 80)
    for name, builder in EDGE_CASES.items():
        inst = builder()
        schedule = schedule_global_ect(inst)
        result = validate_schedule(inst, schedule)
        status = "VALID" if result.is_valid else "INVALID"
        if result.is_valid:
            lb = lower_bound(inst)
            gap = 100 * (result.makespan - lb) / lb if lb > 0 else 0.0
            print(f"{name:24s} | {len(inst.jobs):4d} {inst.total_operations():4d} | "
                  f"{result.makespan:8d} | {lb:6d} | {gap:6.1f} | {status}")
        else:
            print(f"{name:24s} | {len(inst.jobs):4d} {inst.total_operations():4d} | "
                  f"{'-':>8s} | {'-':>6s} | {'-':>6s} | {status}: {result.errors[0]}")


if __name__ == "__main__":
    main()