"""End-to-end pipeline smoke test: Generator -> Algorithm -> Validator, across all presets."""

import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generator.instance_generator import FJSPGenerator, INSTANCE_CLASS_PRESETS
from algorithms.greedy import schedule_global_ect
from environments.validator import validate_schedule

print(f"{'class':20s} | {'jobs':>4s} {'ops':>4s} | {'makespan':>8s} | {'runtime_ms':>10s} | result")
print("-" * 75)

for cls_name in INSTANCE_CLASS_PRESETS:
    gen = FJSPGenerator(seed=42)
    inst = gen.generate_preset(cls_name, number_of_jobs=8, number_of_machines=5, operations_per_job=(3, 6))

    t0 = time.perf_counter()
    schedule = schedule_global_ect(inst)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    result = validate_schedule(inst, schedule)
    status = "VALID" if result.is_valid else f"INVALID ({len(result.errors)} errors)"
    makespan = result.makespan if result.is_valid else "-"

    print(f"{cls_name:20s} | {len(inst.jobs):4d} {inst.total_operations():4d} | "
          f"{str(makespan):>8s} | {elapsed_ms:10.2f} | {status}")

    if not result.is_valid:
        print("  First error:", result.errors[0])