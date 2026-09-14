<<<<<<< HEAD
# fjsp
=======
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

# How to do failure analysis

Do not write:

> "SLGA is better because reinforcement learning is smarter."

Instead use the required structure:

### Observation
What happened?

Example:

> SLGA improved over fixed GA on high-flexibility instances in 4/5 seeds.

### Evidence
Show the actual numbers from `results_by_algorithm.csv` and, where useful, `convergence.csv`.

### Hypothesis
Why might this happen?

Example:

> High flexibility creates many possible machine assignments, so a fixed GA parameter setting may either converge too quickly or explore too slowly.

### Structural explanation
Connect the behavior to the FJSP structure.

### Proposed improvement
Suggest a targeted change and test it.

Possible failure modes worth investigating include:

- high flexibility → large routing search space
- bottleneck-heavy → many operations compete for the same machine
- high variance → short processing-time choices can create downstream contention
- low flexibility → machine-assignment search is small, so adaptive GA may have less room to help
- extreme → large processing-time differences can dominate the search

These are hypotheses, not conclusions. The CSV must determine which ones actually occur.

---

# Research basis

The main research reference for the new method is:

**R. Chen, B. Yang, S. Li, and S. Wang (2020), "A self-learning genetic algorithm based on reinforcement learning for flexible job-shop scheduling problem", Computers & Industrial Engineering, 149, 106778.**

The paper describes an FJSP GA using an operation-sequence + machine-assignment representation, POX crossover for the operation sequence, swap mutation, and a reinforcement-learning controller that dynamically adjusts crossover and mutation probabilities.

This repository adapts those ideas to its own `Instance` model, decoder, validator, generator, and experiment pipeline.

A broader background reference is:

**Dauzère-Pérès et al. (2024), "The flexible job shop scheduling problem: A review", European Journal of Operational Research, 314(2), 409–432.**

Use the papers to justify design choices; do not claim that this repository reproduces the paper's published benchmark values unless an exact reproduction is actually performed.

---

# Project structure

```text
fjsp/
├── algorithms/
│   ├── greedy.py
│   ├── ga.py
│   └── slga.py
├── environments/
│   ├── metrics.py
│   ├── test_validator.py
│   └── validator.py
├── experiments/
│   ├── edge_cases.py
│   ├── failure_analysis.py
│   ├── compare_algorithms.py
│   ├── convergence.csv
│   ├── results_by_algorithm.csv
│   ├── results_by_class.csv
│   ├── run_exp.py
│   └── smoketest.py
├── generator/
│   ├── bottleneck.json
│   ├── demo_generate.py
│   ├── instance_generator.py
│   └── model.py
└── instances/
```

## Final project story

The strongest way to present the project is:

> **We first built a valid and independently verifiable FJSP environment. The original greedy ECT heuristic gave us a transparent baseline and exposed structural failure modes. We then introduced a chromosome-based GA for simultaneous operation sequencing and machine assignment. Finally, motivated by Chen et al. (2020), we made the GA's crossover and mutation probabilities adaptive using a small reinforcement-learning controller. We evaluate all three methods on the same generated instances across multiple seeds and analyze where the adaptive method helps and where it fails.**

That is a much more defensible contribution than claiming a completely novel scheduling algorithm.
>>>>>>> b3cc8f5 (improved)
