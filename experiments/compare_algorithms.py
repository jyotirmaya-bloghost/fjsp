
"""
Part D -- Compare the existing greedy baseline with GA and SLGA.

The experiment is deliberately paired: for each instance class and seed, all
three algorithms solve the exact same generated instance. This isolates the
algorithmic effect from random instance generation.

Outputs:
  results_by_algorithm.csv  -- one row per algorithm/class/seed
  convergence.csv           -- best-so-far makespan at every generation for
                               GA and SLGA (greedy has no generations)

No algorithm is considered "better" from a single run. Use the class-level
aggregates printed at the end and the CSV for failure analysis.
"""

import csv
import os
import statistics
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generator.instance_generator import FJSPGenerator, INSTANCE_CLASS_PRESETS
from algorithms.greedy import schedule_global_ect
from algorithms.ga import genetic_algorithm
from algorithms.slga import self_learning_ga
from environments.validator import validate_schedule
from environments.metrics import lower_bound, machine_utilization, critical_job


N_SEEDS = 5
NUM_JOBS = 10
NUM_MACHINES = 6
OPS_PER_JOB = (3, 6)

POPULATION_SIZE = 40
GENERATIONS = 80
GA_PC = 0.85
GA_PM = 0.15

OUT_CSV = os.path.join(os.path.dirname(__file__), "results_by_algorithm.csv")
CONV_CSV = os.path.join(os.path.dirname(__file__), "convergence.csv")


def evaluate(inst, schedule, runtime_ms, algorithm, seed, info=None):
    result = validate_schedule(inst, schedule)

    row = {
        "algorithm": algorithm,
        "instance_class": inst.instance_class,
        "seed": seed,
        "num_jobs": len(inst.jobs),
        "num_machines": inst.num_machines,
        "num_ops": inst.total_operations(),
        "valid": result.is_valid,
        "runtime_ms": round(runtime_ms, 3),
        "population_size": info.get("population_size") if info else None,
        "generations": info.get("generations") if info else 0,
        "crossover_probability": info.get("crossover_probability") if info else None,
        "mutation_probability": info.get("mutation_probability") if info else None,
        "best_makespan": result.makespan if result.is_valid else None,
        "lower_bound": None,
        "gap_pct": None,
        "avg_utilization": None,
        "min_utilization": None,
        "max_utilization": None,
        "critical_job": None,
    }

    if result.is_valid:
        lb = lower_bound(inst)
        util = machine_utilization(inst, result)
        row["lower_bound"] = lb
        row["gap_pct"] = round(100 * (result.makespan - lb) / lb, 3) if lb else 0.0
        row["avg_utilization"] = round(statistics.mean(util.values()), 4)
        row["min_utilization"] = round(min(util.values()), 4)
        row["max_utilization"] = round(max(util.values()), 4)
        row["critical_job"] = critical_job(result)

    return row


def run_one(inst, algorithm_name, seed):
    if algorithm_name == "Greedy ECT":
        t0 = time.perf_counter()
        schedule = schedule_global_ect(inst)
        runtime = (time.perf_counter() - t0) * 1000
        row = evaluate(inst, schedule, runtime, algorithm_name, seed)
        return row, []

    if algorithm_name == "GA":
        t0 = time.perf_counter()
        schedule, info = genetic_algorithm(
            inst,
            population_size=POPULATION_SIZE,
            generations=GENERATIONS,
            crossover_probability=GA_PC,
            mutation_probability=GA_PM,
            seed=seed,
            return_history=True,
        )
        runtime = (time.perf_counter() - t0) * 1000
        row = evaluate(inst, schedule, runtime, algorithm_name, seed, info)
        return row, info["history"]

    if algorithm_name == "SLGA":
        t0 = time.perf_counter()
        schedule, info = self_learning_ga(
            inst,
            population_size=POPULATION_SIZE,
            generations=GENERATIONS,
            initial_crossover_probability=GA_PC,
            initial_mutation_probability=GA_PM,
            seed=seed,
            return_history=True,
        )
        runtime = (time.perf_counter() - t0) * 1000
        row = evaluate(inst, schedule, runtime, algorithm_name, seed, info)
        return row, info["history"]

    raise ValueError(algorithm_name)


def main():
    rows = []
    convergence = []

    algorithms = ["Greedy ECT", "GA", "SLGA"]

    for cls_name in INSTANCE_CLASS_PRESETS:
        for seed in range(1, N_SEEDS + 1):
            # IMPORTANT: same instance for all algorithms.
            inst = FJSPGenerator(seed=seed).generate_preset(
                cls_name,
                number_of_jobs=NUM_JOBS,
                number_of_machines=NUM_MACHINES,
                operations_per_job=OPS_PER_JOB,
            )

            for algorithm_name in algorithms:
                row, history = run_one(inst, algorithm_name, seed)
                rows.append(row)

                for generation, makespan in enumerate(history):
                    convergence.append({
                        "algorithm": algorithm_name,
                        "instance_class": cls_name,
                        "seed": seed,
                        "generation": generation,
                        "best_makespan": makespan,
                    })

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with open(CONV_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(convergence[0].keys()))
        writer.writeheader()
        writer.writerows(convergence)

    print()
    print("ALGORITHM COMPARISON")
    print(f"{'class':20s} | {'Greedy':>8s} | {'GA':>8s} | {'SLGA':>8s} | {'SLGA vs GA':>10s}")
    print("-" * 70)

    by_key = {}
    for r in rows:
        if r["valid"]:
            by_key.setdefault(r["instance_class"], {}).setdefault(r["algorithm"], []).append(
                r["best_makespan"]
            )

    for cls in INSTANCE_CLASS_PRESETS:
        vals = by_key[cls]
        g = statistics.mean(vals["Greedy ECT"])
        ga = statistics.mean(vals["GA"])
        slga = statistics.mean(vals["SLGA"])
        pct = 100 * (ga - slga) / ga if ga else 0.0
        print(f"{cls:20s} | {g:8.1f} | {ga:8.1f} | {slga:8.1f} | {pct:9.1f}%")

    print()
    print(f"Detailed results: {OUT_CSV}")
    print(f"Convergence data: {CONV_CSV}")


if __name__ == "__main__":
    main()
