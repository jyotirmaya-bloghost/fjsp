"""
Aggregates experiments/results_by_algorithm.csv and experiments/convergence.csv
into the summary tables quoted in analysis/FAILURE_ANALYSIS.md.

Every number in that document is produced by this script, so the analysis is
reproducible rather than hand-copied. Run:

    python analysis/aggregate.py

It writes:
    analysis/summary_by_class.csv    mean makespan / gap / runtime per class
    analysis/convergence_summary.csv where each algorithm stops improving

and prints both tables to stdout.
"""

import csv
import os
import statistics
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "experiments", "results_by_algorithm.csv")
CONVERGENCE = os.path.join(ROOT, "experiments", "convergence.csv")

ALGORITHMS = ["Greedy ECT", "GA", "SLGA"]
CLASS_ORDER = [
    "average", "low_flexibility", "high_flexibility", "bottleneck_heavy",
    "balanced", "high_variance", "machine_advantage", "hard", "extreme",
]


def _mean(values):
    values = list(values)
    return statistics.mean(values) if values else 0.0


def summarize_results():
    rows = list(csv.DictReader(open(RESULTS)))
    by = defaultdict(list)
    for r in rows:
        by[(r["instance_class"], r["algorithm"])].append(r)

    out = []
    for cls in CLASS_ORDER:
        rec = {"instance_class": cls}
        for alg in ALGORITHMS:
            group = by[(cls, alg)]
            if not group:
                continue
            key = alg.replace(" ", "_").lower()
            rec[f"{key}_makespan"] = round(_mean(float(r["best_makespan"]) for r in group), 1)
            rec[f"{key}_gap_pct"] = round(_mean(float(r["gap_pct"]) for r in group), 1)
            rec[f"{key}_runtime_ms"] = round(_mean(float(r["runtime_ms"]) for r in group), 1)
            rec[f"{key}_avg_util"] = round(_mean(float(r["avg_utilization"]) for r in group), 3)

        # paired win counts against the greedy baseline, per seed
        per_seed = defaultdict(dict)
        for alg in ALGORITHMS:
            for r in by[(cls, alg)]:
                per_seed[r["seed"]][alg] = float(r["best_makespan"])
        rec["ga_beats_greedy"] = sum(
            1 for v in per_seed.values() if v.get("GA", 0) < v.get("Greedy ECT", 0)
        )
        rec["slga_beats_greedy"] = sum(
            1 for v in per_seed.values() if v.get("SLGA", 0) < v.get("Greedy ECT", 0)
        )
        rec["slga_beats_ga"] = sum(
            1 for v in per_seed.values() if v.get("SLGA", 0) < v.get("GA", 0)
        )
        rec["seeds"] = len(per_seed)
        out.append(rec)
    return out


def summarize_convergence():
    if not os.path.exists(CONVERGENCE):
        return []
    rows = list(csv.DictReader(open(CONVERGENCE)))
    curves = defaultdict(dict)
    for r in rows:
        curves[(r["algorithm"], r["instance_class"], r["seed"])][int(r["generation"])] = int(
            r["best_makespan"]
        )

    agg = defaultdict(list)
    for (alg, cls, _seed), curve in curves.items():
        gens = sorted(curve)
        first = curve[gens[0]]
        best, last_improve = first, 0
        for g in gens:
            if curve[g] < best:
                best, last_improve = curve[g], g
        agg[(cls, alg)].append((first, best, last_improve, gens[-1]))

    out = []
    for cls in CLASS_ORDER:
        for alg in ["GA", "SLGA"]:
            v = agg[(cls, alg)]
            if not v:
                continue
            first = _mean(x[0] for x in v)
            final = _mean(x[1] for x in v)
            out.append({
                "instance_class": cls,
                "algorithm": alg,
                "gen0_makespan": round(first, 1),
                "final_makespan": round(final, 1),
                "improvement_pct": round(100 * (first - final) / first, 2) if first else 0.0,
                "last_improving_generation": round(_mean(x[2] for x in v), 1),
                "total_generations": int(_mean(x[3] for x in v)),
                "stagnant_runs": sum(1 for x in v if x[2] == 0),
                "runs": len(v),
            })
    return out


def _write(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    results = summarize_results()
    convergence = summarize_convergence()

    _write(os.path.join(HERE, "summary_by_class.csv"), results)
    _write(os.path.join(HERE, "convergence_summary.csv"), convergence)

    print("MEAN MAKESPAN AND GAP BY CLASS (5 seeds each)")
    print(f"{'class':20s} {'Greedy':>8s} {'GA':>8s} {'SLGA':>8s} "
          f"{'gapG%':>7s} {'gapGA%':>7s} {'gapSL%':>7s} {'GA>base':>8s} {'SL>base':>8s}")
    print("-" * 95)
    for r in results:
        print(f"{r['instance_class']:20s} "
              f"{r.get('greedy_ect_makespan', 0):8.1f} "
              f"{r.get('ga_makespan', 0):8.1f} "
              f"{r.get('slga_makespan', 0):8.1f} "
              f"{r.get('greedy_ect_gap_pct', 0):7.1f} "
              f"{r.get('ga_gap_pct', 0):7.1f} "
              f"{r.get('slga_gap_pct', 0):7.1f} "
              f"{r['ga_beats_greedy']:5d}/{r['seeds']:<2d} "
              f"{r['slga_beats_greedy']:5d}/{r['seeds']:<2d}")

    print("\nCONVERGENCE: WHERE SEARCH STOPS PAYING OFF")
    print(f"{'class':20s} {'alg':5s} {'gen0':>7s} {'final':>7s} {'impr%':>7s} "
          f"{'lastImpr':>9s} {'ofGens':>7s} {'stagnant':>9s}")
    print("-" * 85)
    for r in convergence:
        print(f"{r['instance_class']:20s} {r['algorithm']:5s} "
              f"{r['gen0_makespan']:7.1f} {r['final_makespan']:7.1f} "
              f"{r['improvement_pct']:7.2f} {r['last_improving_generation']:9.1f} "
              f"{r['total_generations']:7d} {r['stagnant_runs']:4d}/{r['runs']:<4d}")

    print(f"\nWrote analysis/summary_by_class.csv and analysis/convergence_summary.csv")


if __name__ == "__main__":
    main()
