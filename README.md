# Flexible Job-Shop Scheduling (FJSP)

This project implements a complete, reproducible FJSP pipeline:

**instance generator → schedule construction/optimization → independent validator → metrics → multi-seed experiments → failure analysis**

The project is intentionally structured so that an algorithm cannot "validate itself". Every produced schedule is checked by `environments/validator.py`.

## What was already in the project

### 1. Instance generator

`generator/instance_generator.py` generates valid instances from numeric parameters and provides nine named classes:

- average
- low_flexibility
- high_flexibility
- bottleneck_heavy
- balanced
- high_variance
- machine_advantage
- hard
- extreme

Generation is seeded and validity is enforced by construction.

#### How eligible machine sets are sampled

`machine_flexibility ∈ (0,1]` sets the target average size of `E(j,k)` as a
fraction of all machines: `target = round(flexibility × m)`. A ±1 jitter is
added so instances are not perfectly uniform (every operation having identical
`|E|` would be unrealistic). The size is then clamped to `[1, m]`, which is what
guarantees `|E(j,k)| ≥ 1` structurally.

- `flexibility = 0.15` on 6 machines → ~1 eligible machine (rigid routing)
- `flexibility = 0.85` on 6 machines → ~5 eligible machines (very flexible)

#### How processing times are distributed

Times are drawn from a **triangular distribution** peaked at the midpoint of
`processing_time_range`, not uniform — real operation durations cluster around a
typical value while still allowing outliers. Variance is tuned by scaling the
half-width: `spread = (hi − lo)/2 × processing_time_variance`, so
`variance = 0.3` clusters tightly around the midpoint and `variance = 3.0`
spreads to the edges. Values are rounded and clipped into `[1, hi]`, which is
what guarantees every processing time is a strictly positive integer.

#### How bottlenecks, machine advantage and imbalance are injected

- **Bottleneck:** one machine is chosen *per instance* (not re-rolled per
  operation) and force-added to `E(j,k)` with probability
  `bottleneck_probability`. Choosing it once is what makes it a genuine
  contention point rather than noise, and it is independent of flexibility, so
  contention can be injected even into otherwise-flexible instances.
- **Machine advantage:** `machine_advantage={m: multiplier}` scales a specific
  machine's times *after* sampling, e.g. `{1: 0.2}` makes machine 1 about 5×
  faster wherever it is eligible.
- **Imbalance:** produced by combining a high bottleneck probability with an
  uneven `operations_per_job` range.

#### How named classes map to parameters

All nine classes are parameter presets over a single `generate()` method — there
are no special-cased code paths, which keeps the logic auditable in one place
(`INSTANCE_CLASS_PRESETS`):

| class | flexibility | time range | variance | bottleneck p | advantage |
|---|---|---|---|---|---|
| average | 0.40 | (1, 20) | 1.0 | 0.05 | — |
| low_flexibility | 0.15 | (1, 20) | 1.0 | 0.0 | — |
| high_flexibility | 0.85 | (1, 20) | 1.0 | 0.0 | — |
| bottleneck_heavy | 0.40 | (1, 20) | 1.0 | 0.70 | — |
| balanced | 0.40 | (8, 12) | 0.5 | 0.0 | — |
| high_variance | 0.40 | (1, 50) | 2.0 | 0.0 | — |
| machine_advantage | 0.50 | (5, 20) | 1.0 | 0.0 | `{1: 0.2}` |
| hard | 0.60 | (1, 40) | 1.6 | 0.40 | — |
| extreme | 0.90 | (1, 200) | 3.0 | 0.60 | `{1: 0.05}` |

#### Why the construction guarantees validity

There is no generate-then-repair step. Each rule from Section 2 of the problem
statement is enforced at the moment of creation:

| rule | enforced by |
|---|---|
| `\|E(j,k)\| ≥ 1` | size clamped to `max(1, …)` before sampling |
| times defined for exactly `E(j,k)` | `proc_times` built by iterating over `eligible` |
| times strictly positive integers | `max(1, min(hi, round(·)))` |
| contiguous IDs, no duplicates | jobs/ops numbered by `range()` loops |
| linear precedence chain | operations stored in an ordered list per job |
| feasible by construction | every operation has ≥1 machine, so scheduling each in order is always possible |

It is therefore structurally impossible for `generate()` to emit a malformed
instance. All randomness comes from a single `random.Random(seed)`, so the same
seed and parameters reproduce a byte-identical instance.

### 2. Greedy ECT baseline

`algorithms/greedy.py` is the original baseline.

At every step it considers every currently ready operation and every eligible machine and chooses the pair with the earliest completion time:

`finish = max(job_ready_time, machine_free_time) + processing_time`

This is intentionally myopic. It is useful because its failures are easy to interpret.

---

# New contribution: GA → adaptive SLGA

Two optimization algorithms have been added:

- `algorithms/ga.py` — conventional Genetic Algorithm
- `algorithms/slga.py` — Self-Learning Genetic Algorithm (SLGA), inspired by Chen et al. (2020)

The progression is:

**Greedy ECT → fixed-parameter GA → adaptive GA**

That gives the project a much stronger experimental story than simply replacing the original algorithm.

## Why GA fits FJSP

FJSP has two coupled decisions:

1. operation sequencing
2. machine assignment

The GA represents both in one chromosome.

### Operation-sequence (OS) part

Example:

```text
J1 J2 J1 J3 J2 J3
```

If Job 1 has three operations, its first occurrence means Op1, its second occurrence means Op2, etc.

Every job appears exactly as many times as it has operations. Therefore the representation itself respects the operation-count constraint.

### Machine-assignment (MA) part

Each OS position has a machine ID:

```text
OS: J1 J2 J1 J3 J2 J3
MA: M2 M1 M3 M2 M3 M1
```

For each position, the selected machine must be eligible for the corresponding operation.

## Decoder

`decode_chromosome()` scans the chromosome left-to-right.

For each operation:

```text
start = max(job predecessor completion,
            selected machine free time)

finish = start + processing time
```

So the decoder constructs a non-preemptive schedule satisfying job precedence and machine non-overlap.

The independent validator is still run afterwards.

---

# GA operators

The implementation follows the main representation/operator ideas described in Chen et al. (2020), adapted to this repository.

### Population initialization

Most individuals are randomized valid chromosomes.

One informed individual is seeded from the existing greedy ECT solution:

- its OS is taken from the greedy schedule
- machine assignments prefer the fastest eligible machine

This means the new method builds on the existing baseline rather than throwing it away.

### Crossover

The OS uses **precedence-preserving order-based crossover (POX)**.

The MA part uses uniform crossover followed by an eligibility repair step.

### Mutation

Two mutations are used:

- OS: swap two genes belonging to different jobs
- MA: change an operation's assigned machine to another eligible machine

The machine repair step guarantees that crossover/mutation cannot leave an illegal machine assignment.

### Elitism

The best chromosome survives every generation.

---

# SLGA: the research-inspired improvement

The main weakness of ordinary GA is that its crossover and mutation probabilities are fixed.

SLGA makes those probabilities adaptive.

The controller observes the population and chooses an action such as:

```text
increase Pc
decrease Pm
increase both
decrease both
leave both unchanged
...
```

There are 9 small parameter-adjustment actions.

## Population state

Following the state-design idea in Chen et al. (2020), the controller uses:

- normalized average population fitness
- normalized population diversity
- normalized best fitness

The paper weights these components:

```text
0.35 average fitness
0.35 diversity
0.30 best fitness
```

This implementation uses the same weights and maps the resulting value to 20 discrete states.

For diversity, this repository uses a transparent normalized Hamming-distance proxy over OS+MA chromosomes.

## Reinforcement learning controller

The controller uses a small 20 × 9 Q-table.

It uses:

- **SARSA** during the early half of the run
- **Q-learning** during the later half

The reward is primarily normalized improvement in the best makespan, with a small diversity component.

The purpose is not to claim a new RL algorithm. The contribution is the integration of an adaptive parameter controller into this project's existing GA/FJSP pipeline.

> Important: this is an implementation inspired by Chen et al. (2020), not an exact reproduction of their experimental setup or benchmark results.

---

# Running the project

From the repository root:

## Validator tests

```bash
python environments/test_validator.py
```

## GA/SLGA tests

```bash
python algorithms/test_algorithms.py
```

## Existing greedy smoke test

```bash
python experiments/smoketest.py
```

## Hand-built edge cases

```bash
python experiments/edge_cases.py
```

## Compare all algorithms

```bash
python experiments/compare_algorithms.py
```

This runs the **same generated instance** through:

- Greedy ECT
- GA
- SLGA

for multiple seeds and all nine generated instance classes.

It writes:

```text
experiments/results_by_algorithm.csv
experiments/convergence.csv
```

## Failure-analysis summary

After the comparison:

```bash
python experiments/failure_analysis.py
```

---

# Experimental design

The comparison uses:

```text
5 seeds
9 instance classes
10 jobs
6 machines
3–6 operations per job
population = 40
generations = 80
```

For every `(instance_class, seed)` pair, the instance is generated **once** and passed to all algorithms. Therefore the comparison is paired and does not mix algorithm quality with different random problem instances.

The recorded metrics include:

- validity
- runtime
- makespan
- lower bound
- gap percentage
- average machine utilization
- minimum machine utilization
- maximum machine utilization
- critical job

For GA and SLGA, convergence is also recorded generation-by-generation.

---

# Results and failure analysis

The full write-up is in **[`analysis/FAILURE_ANALYSIS.md`](analysis/FAILURE_ANALYSIS.md)**.
It follows the required Observation → Evidence → Hypothesis → Structural
explanation → Proposed improvement structure for each failure mode.

## Headline numbers

9 classes × 5 seeds, 10 jobs, 6 machines, 3–6 ops/job, population 40,
80 generations. Instances are generated once per `(class, seed)` and shared by
all three algorithms, so comparisons are paired. All 135 runs passed the
independent validator.

| class | Greedy | GA | SLGA | gap Greedy | gap GA | gap SLGA |
|---|---|---|---|---|---|---|
| average | 99.0 | 95.2 | 95.4 | 57.9% | 51.6% | 51.8% |
| low_flexibility | 142.6 | 125.4 | 127.4 | 79.7% | 57.3% | 59.9% |
| high_flexibility | 70.0 | 70.0 | 70.0 | 49.0% | 49.0% | 49.0% |
| bottleneck_heavy | 91.0 | 91.0 | 90.2 | 54.1% | 54.1% | 52.7% |
| balanced | 111.4 | 104.6 | 105.4 | 51.4% | 41.7% | 42.7% |
| high_variance | 344.6 | 308.8 | 306.0 | 72.5% | 52.4% | 51.2% |
| machine_advantage | 86.4 | 84.8 | 82.6 | 56.9% | 54.1% | 49.8% |
| hard | 158.0 | 158.0 | 157.8 | 56.0% | 56.0% | 55.8% |
| extreme | 364.8 | 364.8 | 364.4 | 54.2% | 54.2% | 54.1% |

## The five failure modes found

1. **The GA degenerates into an expensive greedy on four of nine classes.** On
   `high_flexibility`, `hard`, `extreme` and `bottleneck_heavy` the GA returns
   the *identical* greedy makespan in 5/5 seeds, with zero improvement across
   80 generations. The cause was isolated with a new measurement, the
   **seed dominance ratio** = (best of 1500 random chromosomes) / (greedy
   makespan). When that ratio exceeds ~1.2, the greedy-seeded elite dominates
   the population at generation 0 and crossover has no useful material.
   Spearman ρ between dominance ratio and GA improvement is **−0.70** across
   the nine classes. This reframes "the GA fails on flexible instances" as an
   initialization-design problem, and the ratio is measurable *before* running
   the GA.

2. **Greedy ECT is worst exactly where routing freedom is lowest**
   (`low_flexibility`, gap 79.7%) — with ~1 eligible machine per operation the
   makespan is decided by sequencing, which greedy never revisits.

3. **SLGA converges earlier but often lands slightly worse than the fixed GA.**
   An honest negative result: a 20×9 Q-table has 180 entries to learn inside a
   single 80-generation run with no cross-run persistence, so the controller is
   noisy and its reward is dominated by short-horizon gains.

4. **More than half the search budget produces nothing** — the last improving
   generation averages 18–36 out of 80.

5. **The lower bound was too weak to measure anything.** Fixed (see below).

## Lower-bound correction

The original bound was purely precedence-based (per job, sum the minimum
processing time of its operations; take the max over jobs). That is valid but
ignores machine contention completely, so on machine-constrained instances it
was badly loose — the hand-built `single_machine` edge case reported a 200% gap
even though the greedy schedule there is *provably optimal*.

A work-content bound was added: every operation must occupy some machine for at
least its fastest eligible time, so total unavoidable work is `Σ min_m p(j,k,m)`,
spread over at most `m` machines, giving `C_max ≥ ⌈Σ min_m p / m⌉`. The reported
bound is now `max(LB1, LB2)`.

| edge case | gap (old) | gap (new) |
|---|---|---|
| single_machine | 200.0% | **0.0%** |
| identical_times | 100.0% | **0.0%** |
| many_short_ops | 340.0% | 29.4% |
| bottleneck_machine | 300.0% | 140.0% |
| near_zero_flexibility | 100.0% | 42.9% |

Two edge cases are now proven optimal instead of appearing badly suboptimal.
Neither bound dominates the other — LB2 is stronger on 26 of 45 generated
instances, LB1 remains stronger on `extreme` and `machine_advantage` — so both
are computed. **Validity check: across all 45 generated instances and 10 edge
cases, the number of times `LB > achieved makespan` is 0.**

## Reproducing the analysis

```bash
python experiments/compare_algorithms.py   # raw runs   -> experiments/*.csv
python analysis/aggregate.py               # summaries  -> analysis/summary_by_class.csv
python analysis/seed_dominance.py          # FM1 evidence -> analysis/seed_dominance.csv
python experiments/edge_cases.py           # hand-built edge cases
```

Every number quoted in the analysis is produced by one of these scripts.

---

# Research basis

The main research reference for the new method is:

**R. Chen, B. Yang, S. Li, and S. Wang (2020), "A self-learning genetic algorithm based on reinforcement learning for flexible job-shop scheduling problem", Computers & Industrial Engineering, 149, 106778.**

The paper describes an FJSP GA using an operation-sequence + machine-assignment representation, POX crossover for the operation sequence, swap mutation, and a reinforcement-learning controller that dynamically adjusts crossover and mutation probabilities.

This repository adapts those ideas to its own `Instance` model, decoder, validator, generator, and experiment pipeline.

A broader background reference is:

**Dauzère-Pérès et al. (2024), "The flexible job shop scheduling problem: A review", European Journal of Operational Research, 314(2), 409–432.**

These papers are used to justify design choices only. This repository does **not**
reproduce either paper's published benchmark values: the SLGA implementation
borrows the representation, POX crossover, state weighting (0.35 average fitness
/ 0.35 diversity / 0.30 best fitness) and the SARSA-then-Q-learning schedule from
Chen et al. (2020), but is evaluated on this project's own generated instances
rather than on the standard Brandimarte/Hurink benchmarks, and no attempt is made
to match reported results.

---

# Project structure

```text
fjsp/
├── algorithms/
│   ├── greedy.py                    # ECT dispatching baseline
│   ├── ga.py                        # genetic algorithm
│   ├── slga.py                      # self-learning (RL-adaptive) GA
│   └── test_algorithms.py
├── environments/
│   ├── metrics.py                   # lower bounds, utilization, critical job
│   ├── validator.py                 # independent schedule validator
│   └── test_validator.py            # 7 tests proving invalid schedules are rejected
├── experiments/
│   ├── smoketest.py                 # end-to-end pipeline check
│   ├── edge_cases.py                # 10 hand-built edge cases
│   ├── compare_algorithms.py        # paired 3-algorithm comparison
│   ├── failure_analysis.py          # per-class win counts
│   ├── run_exp.py
│   ├── results_by_algorithm.csv
│   ├── results_by_class.csv
│   └── convergence.csv
├── analysis/
│   ├── FAILURE_ANALYSIS.md          # main analysis document
│   ├── aggregate.py                 # builds the summary tables
│   ├── seed_dominance.py            # evidence for failure mode 1
│   ├── summary_by_class.csv
│   ├── convergence_summary.csv
│   └── seed_dominance.csv
├── generator/
│   ├── instance_generator.py        # FJSPGenerator + 9 class presets
│   ├── model.py                     # Instance/Job/Operation + JSON I/O
│   ├── demo_generate.py
│   └── bottleneck.json
└── instances/                       # 9 pre-generated instances (seed 42)
```

## Final project story

The strongest way to present the project is:

> **We first built a valid and independently verifiable FJSP environment. The original greedy ECT heuristic gave us a transparent baseline and exposed structural failure modes. We then introduced a chromosome-based GA for simultaneous operation sequencing and machine assignment. Finally, motivated by Chen et al. (2020), we made the GA's crossover and mutation probabilities adaptive using a small reinforcement-learning controller. We evaluate all three methods on the same generated instances across multiple seeds and analyze where the adaptive method helps and where it fails.**

That is a much more defensible contribution than claiming a completely novel scheduling algorithm.
