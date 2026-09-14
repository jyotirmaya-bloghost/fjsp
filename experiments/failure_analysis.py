
"""
Failure-analysis helper for the algorithm comparison.

For each instance class, this script reports:
  * mean makespan by algorithm,
  * how often GA/SLGA beat Greedy ECT on the same seed,
  * how often SLGA beats fixed-parameter GA,
  * average runtime,
  * the largest observed regression of SLGA against GA.

The output is evidence for Part D's Observation -> Evidence -> Hypothesis
workflow. It does not automatically declare a causal explanation.
"""

import csv
import os
import statistics
from collections import defaultdict


RESULTS = os.path.join(os.path.dirname(__file__), "results_by_algorithm.csv")


def main():
    if not os.path.exists(RESULTS):
        raise SystemExit("Run compare_algorithms.py first.")

    rows = list(csv.DictReader(open(RESULTS)))
    groups = defaultdict(list)
    for r in rows:
        groups[(r["instance_class"], r["algorithm"])].append(r)

    algorithms = ["Greedy ECT", "GA", "SLGA"]

    print(f"{'class':20s} | {'GA wins':>8s} | {'SLGA wins':>9s} | {'SLGA<GA':>8s} | {'runtime SLGA':>12s}")
    print("-" * 75)

    for cls in sorted(set(r["instance_class"] for r in rows)):
        by_seed = defaultdict(dict)
        for alg in algorithms:
            for r in groups[(cls, alg)]:
                by_seed[r["seed"]][alg] = float(r["best_makespan"])

        ga_wins = slga_wins = slga_better = 0
        for vals in by_seed.values():
            if vals["GA"] < vals["Greedy ECT"]:
                ga_wins += 1
            if vals["SLGA"] < vals["Greedy ECT"]:
                slga_wins += 1
            if vals["SLGA"] < vals["GA"]:
                slga_better += 1

        slga_rt = statistics.mean(
            float(r["runtime_ms"]) for r in groups[(cls, "SLGA")]
        )
        print(
            f"{cls:20s} | {ga_wins:8d} | {slga_wins:9d} | "
            f"{slga_better:8d} | {slga_rt:12.2f}"
        )

    print("\nInterpretation rule:")
    print("Use these counts together with the raw CSV and convergence curves.")
    print("A class is a meaningful failure mode only when the pattern repeats")
    print("across seeds; then explain it using the instance structure.")


if __name__ == "__main__":
    main()
