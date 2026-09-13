"""
Part D -- Experiments across instance classes.

Runs every named instance class from INSTANCE_CLASS_PRESETS across multiple
seeds, records the full metric set per run, and writes results to CSV.
Never draws conclusions from a single instance -- each class gets N_SEEDS
independent draws.
"""

import os
import sys
import csv
import time
import statistics

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generator.instance_generator import FJSPGenerator, INSTANCE_CLASS_PRESETS
from algorithms.greedy import schedule_global_ect
from environments.validator import validate_schedule
from environments.metrics import lower_bound, machine_utilization

N_SEEDS = 5
NUM_JOBS = 10
NUM_MACHINES = 6
OPS_PER_JOB = (3, 6)

OUT_CSV = os.path.join(os.path.dirname(__file__), "results_by_class.csv")


def run_one(cls_name: str, seed: int) -> dict:
    gen = FJSPGenerator(seed=seed)
    inst = gen.generate_preset(cls_name, number_of_jobs=NUM_JOBS, number_of_machines=NUM_MACHINES,
                                operations_per_job=OPS_PER_JOB)

    t0 = time.perf_counter()
    schedule = schedule_global_ect(inst)
    runtime_ms = (time.perf_counter() - t0) * 1000

    result = validate_schedule(inst, schedule)
    row = dict(
        instance_class=cls_name, seed=seed,
        num_jobs=len(inst.jobs), num_machines=inst.num_machines, num_ops=inst.total_operations(),
        valid=result.is_valid, runtime_ms=round(runtime_ms, 3),
    )
    if result.is_valid:
        lb = lower_bound(inst)
        util = machine_utilization(inst, result)
        row.update(
            makespan=result.makespan,
            lower_bound=lb,
            gap_pct=round(100 * (result.makespan - lb) / lb, 1) if lb > 0 else None,
            avg_utilization=round(statistics.mean(util.values()), 3),
            min_utilization=round(min(util.values()), 3),
            max_utilization=round(max(util.values()), 3),
        )
    else:
        row.update(makespan=None, lower_bound=None, gap_pct=None,
                    avg_utilization=None, min_utilization=None, max_utilization=None)
    return row


def main():
    rows = []
    for cls_name in INSTANCE_CLASS_PRESETS:
        for seed in range(1, N_SEEDS + 1):
            rows.append(run_one(cls_name, seed))

    fieldnames = list(rows[0].keys())
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Aggregate per class and print a summary
    print(f"{'class':20s} | {'n':>2s} | {'avg makespan':>12s} | {'avg gap%':>8s} | {'avg util':>8s} | {'avg runtime_ms':>14s}")
    print("-" * 90)
    by_class = {}
    for r in rows:
        by_class.setdefault(r["instance_class"], []).append(r)

    for cls_name, rs in by_class.items():
        valid_rs = [r for r in rs if r["valid"]]
        avg_makespan = statistics.mean(r["makespan"] for r in valid_rs)
        avg_gap = statistics.mean(r["gap_pct"] for r in valid_rs)
        avg_util = statistics.mean(r["avg_utilization"] for r in valid_rs)
        avg_rt = statistics.mean(r["runtime_ms"] for r in rs)
        print(f"{cls_name:20s} | {len(rs):2d} | {avg_makespan:12.1f} | {avg_gap:8.1f} | {avg_util:8.3f} | {avg_rt:14.3f}")

    print(f"\nFull results written to {OUT_CSV}")


if __name__ == "__main__":
    main()