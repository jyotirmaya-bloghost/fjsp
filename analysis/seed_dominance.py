"""
Why does the GA stagnate completely on some instance classes and improve on
others? This script produces the evidence for Failure Mode 1 in
analysis/FAILURE_ANALYSIS.md.

HYPOTHESIS UNDER TEST
----------------------
The GA population is initialized with ONE informed individual (seeded from the
greedy ECT schedule) and N-1 random individuals. If the greedy seed is far
better than any random individual, it dominates immediately: every crossover
pairs the elite with a much worse partner, offspring are worse, elitism keeps
the elite, and the best-so-far never moves. The GA degenerates into an
expensive way of returning the greedy solution.

If instead random individuals are COMPETITIVE with (or better than) the greedy
seed, the population holds genuinely different good building blocks, crossover
has useful material, and the GA improves.

MEASUREMENT
-----------
For each class/seed, sample `samples` random chromosomes using the GA's own
initializer and decoder, and compare their makespan distribution against the
greedy seed's makespan. The key statistic is the "seed dominance ratio":

    dominance = (best random makespan) / (greedy makespan)

    dominance <= 1.0  ->  random search finds something at least as good as
                          the seed; the GA has material to work with.
    dominance >> 1.0  ->  the seed is unreachable by random sampling; the
                          population is effectively elite + noise.

Run:
    python analysis/seed_dominance.py
Writes analysis/seed_dominance.csv and prints the table.
"""

import csv
import os
import random
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.append(ROOT)

from generator.instance_generator import FJSPGenerator
from environments.metrics import lower_bound
from environments.validator import validate_schedule
from algorithms.greedy import schedule_global_ect
from algorithms.ga import chromosome_makespan, _make_individual

CLASSES = [
    "average", "low_flexibility", "high_flexibility", "bottleneck_heavy",
    "balanced", "high_variance", "machine_advantage", "hard", "extreme",
]


def run(seeds=(1, 2, 3), samples=1500, num_jobs=10, num_machines=6, ops=(3, 6)):
    out = []
    for cls in CLASSES:
        for seed in seeds:
            gen = FJSPGenerator(seed=seed)
            inst = gen.generate_preset(
                cls, number_of_jobs=num_jobs,
                number_of_machines=num_machines, operations_per_job=ops,
            )
            result = validate_schedule(inst, schedule_global_ect(inst))
            assert result.is_valid, "greedy produced an invalid schedule"
            greedy_mk = result.makespan
            lb = lower_bound(inst)

            rng = random.Random(0)
            mks = [
                chromosome_makespan(inst, _make_individual(inst, rng).chromosome)
                for _ in range(samples)
            ]
            out.append({
                "instance_class": cls,
                "seed": seed,
                "lower_bound": lb,
                "greedy_makespan": greedy_mk,
                "random_best": min(mks),
                "random_mean": round(statistics.mean(mks), 1),
                "random_worst": max(mks),
                "distinct_values": len(set(mks)),
                "dominance_ratio": round(min(mks) / greedy_mk, 3),
                "samples": samples,
            })
    return out


def main():
    rows = run()
    path = os.path.join(HERE, "seed_dominance.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print("SEED DOMINANCE: can random chromosomes compete with the greedy seed?")
    print(f"{'class':20s} {'seed':>4s} {'LB':>6s} {'greedy':>7s} {'randBest':>9s} "
          f"{'randMean':>9s} {'ratio':>7s}")
    print("-" * 72)
    for r in rows:
        print(f"{r['instance_class']:20s} {r['seed']:4d} {r['lower_bound']:6d} "
              f"{r['greedy_makespan']:7d} {r['random_best']:9d} "
              f"{r['random_mean']:9.1f} {r['dominance_ratio']:7.2f}")

    print("\nratio <= 1.0 : random search competes -> GA has useful material")
    print("ratio  > 1.3 : greedy seed dominates   -> population is elite + noise")
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
