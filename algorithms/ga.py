
"""
Genetic Algorithm baselines for the Flexible Job-Shop Scheduling Problem.

The representation follows the operation-sequence (OS) + machine-assignment
(MA) chromosome used in Chen et al. (2020).  The implementation is adapted
to this repository's Instance model and independent validator; it is not a
claim of exact reproduction of the paper.

OS:
    A list of job IDs. Job j appears once for every operation in that job.
    The first occurrence of j means Op1, the second means Op2, etc.

MA:
    A machine ID for each OS position. When position p contains job j for
    the r-th time, MA[p] is the machine selected for Job j's Op r.

Decoder:
    Scan OS left-to-right. For each gene, use the corresponding MA machine,
    then schedule the operation at the earliest feasible time after its
    job predecessor and after that machine becomes free.

Because initialization and the genetic operators preserve OS multiplicity
and MA eligibility, decoded schedules are feasible by construction.
The repository validator is still run independently in experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional
import random
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from generator.model import Instance


@dataclass
class Chromosome:
    os: List[int]
    ma: List[int]

    def clone(self) -> "Chromosome":
        return Chromosome(self.os.copy(), self.ma.copy())


@dataclass
class Individual:
    chromosome: Chromosome
    makespan: Optional[int] = None

    def clone(self) -> "Individual":
        return Individual(self.chromosome.clone(), self.makespan)


def _job_operation_map(instance: Instance):
    """Return (job_id -> ordered operations) and operation count by job."""
    return {job.job_id: job.operations for job in instance.jobs}


def _random_os(instance: Instance, rng: random.Random) -> List[int]:
    """
    Generate a valid operation sequence.

    Every job appears exactly len(job.operations) times. At each position a
    job is chosen from the jobs that still have unscheduled operations.
    Weighting by remaining work gives a little more diversity than a flat
    shuffle while never violating precedence.
    """
    remaining = {job.job_id: len(job.operations) for job in instance.jobs}
    os_seq = []

    while remaining:
        jobs = list(remaining)
        # More remaining operations => somewhat more likely to be selected.
        weights = [remaining[j] for j in jobs]
        chosen = rng.choices(jobs, weights=weights, k=1)[0]
        os_seq.append(chosen)
        remaining[chosen] -= 1
        if remaining[chosen] == 0:
            del remaining[chosen]

    return os_seq


def _greedy_chromosome(instance: Instance) -> Chromosome:
    """
    Encode the existing greedy ECT solution exactly as one chromosome.

    This is deliberately stronger than merely using the greedy OS: both the
    operation order and the actual machine decisions are retained. Elitism
    then guarantees the GA cannot finish worse than this seeded solution on
    the same instance.
    """
    from algorithms.greedy import schedule_global_ect

    schedule = schedule_global_ect(instance)
    return Chromosome(
        os=[entry["job_id"] for entry in schedule],
        ma=[entry["machine_id"] for entry in schedule],
    )


def _random_ma(instance: Instance, os_seq: List[int], rng: random.Random) -> List[int]:
    """Choose an eligible machine for every OS gene."""
    ops_by_job = _job_operation_map(instance)
    seen = {job.job_id: 0 for job in instance.jobs}
    ma = []

    for job_id in os_seq:
        op_index = seen[job_id]
        op = ops_by_job[job_id][op_index]
        ma.append(rng.choice(op.eligible_machines))
        seen[job_id] += 1

    return ma


def _greedy_ma(instance: Instance, os_seq: List[int], rng: random.Random) -> List[int]:
    """
    Prefer the fastest eligible machine, with a small tie/random component.
    This is an HCMS-style initialization inspired by Chen et al. (2020).
    """
    ops_by_job = _job_operation_map(instance)
    seen = {job.job_id: 0 for job in instance.jobs}
    ma = []

    for job_id in os_seq:
        op = ops_by_job[job_id][seen[job_id]]
        best = min(op.proc_times.values())
        candidates = [m for m in op.eligible_machines if op.proc_times[m] == best]
        # Usually choose the fastest machine; ties are randomized.
        ma.append(rng.choice(candidates))
        seen[job_id] += 1

    return ma


def _make_individual(instance: Instance, rng: random.Random, informed: bool = False) -> Individual:
    if informed:
        return Individual(_greedy_chromosome(instance))
    os_seq = _random_os(instance, rng)
    ma = _random_ma(instance, os_seq, rng)
    return Individual(Chromosome(os_seq, ma))


def decode_chromosome(instance: Instance, chromosome: Chromosome) -> List[dict]:
    """
    Decode OS+MA into the repository's schedule format.

    Raises ValueError for an invalid chromosome. Normal GA operators never
    create one, but explicit checks make bugs obvious during development.
    """
    total = instance.total_operations()
    if len(chromosome.os) != total or len(chromosome.ma) != total:
        raise ValueError("Chromosome OS and MA must both have length total_operations().")

    ops_by_job = _job_operation_map(instance)
    occurrence = {job.job_id: 0 for job in instance.jobs}
    job_ready = {job.job_id: 0 for job in instance.jobs}
    machine_free = {m: 0 for m in range(1, instance.num_machines + 1)}

    schedule = []

    for pos, job_id in enumerate(chromosome.os):
        if job_id not in ops_by_job:
            raise ValueError(f"Unknown job_id in chromosome: {job_id}")

        k = occurrence[job_id]
        if k >= len(ops_by_job[job_id]):
            raise ValueError(f"Too many occurrences of job {job_id} in OS.")

        op = ops_by_job[job_id][k]
        machine = chromosome.ma[pos]

        if machine not in op.eligible_machines:
            raise ValueError(
                f"Machine {machine} is not eligible for Job {job_id} Op {op.op_id}."
            )

        start = max(job_ready[job_id], machine_free[machine])
        finish = start + op.proc_times[machine]

        schedule.append({
            "job_id": job_id,
            "operation_id": op.op_id,
            "machine_id": machine,
            "start_time": start,
            "completion_time": finish,
        })

        occurrence[job_id] += 1
        job_ready[job_id] = finish
        machine_free[machine] = finish

    if any(occurrence[j.job_id] != len(j.operations) for j in instance.jobs):
        raise ValueError("Chromosome does not contain every job's operations.")

    return schedule


def chromosome_makespan(instance: Instance, chromosome: Chromosome) -> int:
    schedule = decode_chromosome(instance, chromosome)
    return max(x["completion_time"] for x in schedule) if schedule else 0


def _evaluate_population(instance: Instance, population: List[Individual]) -> None:
    for ind in population:
        if ind.makespan is None:
            ind.makespan = chromosome_makespan(instance, ind.chromosome)


def _pox(parent1: List[int], parent2: List[int], rng: random.Random) -> List[int]:
    """
    Precedence-preserving order-based crossover (POX) for operation sequence.

    Select a subset of job IDs. Child positions belonging to selected jobs
    are copied from parent1. Remaining positions are filled, in order, from
    parent2 after skipping selected-job genes. Since both parents contain the
    same multiplicity for every job, the child does too.
    """
    jobs = sorted(set(parent1))
    if len(jobs) <= 1:
        return parent1.copy()

    chosen = {j for j in jobs if rng.random() < 0.5}
    if not chosen or len(chosen) == len(jobs):
        chosen = {rng.choice(jobs)}

    child = [None] * len(parent1)

    for i, job in enumerate(parent1):
        if job in chosen:
            child[i] = job

    fill = [job for job in parent2 if job not in chosen]
    it = iter(fill)
    for i in range(len(child)):
        if child[i] is None:
            child[i] = next(it)

    return child


def crossover(parent1: Chromosome, parent2: Chromosome, rng: random.Random) -> Chromosome:
    """
    Crossover OS with POX and MA with uniform crossover.

    The MA gene at each child position is taken from one parent. If the
    inherited machine is ineligible for the operation represented by that
    position, the other parent's machine is tried; if both are invalid,
    choose a random eligible machine.
    """
    child_os = _pox(parent1.os, parent2.os, rng)

    # We need the operation represented by each child OS position.
    # This is independent of which parent supplied the OS gene.
    # The caller's instance is needed for eligibility, so this function
    # initially creates a parent-derived MA and repair is done below.
    child_ma = [
        parent1.ma[i] if rng.random() < 0.5 else parent2.ma[i]
        for i in range(len(child_os))
    ]
    return Chromosome(child_os, child_ma)


def repair_ma(instance: Instance, chromosome: Chromosome, rng: random.Random) -> None:
    """Repair MA genes after crossover/mutation so every gene is eligible."""
    ops_by_job = _job_operation_map(instance)
    occurrence = {job.job_id: 0 for job in instance.jobs}

    for i, job_id in enumerate(chromosome.os):
        op = ops_by_job[job_id][occurrence[job_id]]
        machine = chromosome.ma[i]
        if machine not in op.eligible_machines:
            # Prefer the fastest legal machine. This makes repair useful,
            # not merely a random validity patch.
            best = min(op.proc_times.values())
            fastest = [m for m in op.eligible_machines if op.proc_times[m] == best]
            chromosome.ma[i] = rng.choice(fastest)
        occurrence[job_id] += 1


def mutate(chromosome: Chromosome, instance: Instance, rng: random.Random,
           os_mutation_prob: float = 1.0, ma_mutation_prob: float = 1.0) -> None:
    """
    Apply swap mutation to OS and a machine reassignment mutation to MA.

    OS swap only swaps two different job IDs. Swapping occurrences of the
    same job has no effect and is avoided.
    """
    n = len(chromosome.os)

    if n >= 2 and rng.random() < os_mutation_prob:
        i, j = rng.sample(range(n), 2)
        attempts = 0
        while chromosome.os[i] == chromosome.os[j] and attempts < 10:
            i, j = rng.sample(range(n), 2)
            attempts += 1
        if chromosome.os[i] != chromosome.os[j]:
            chromosome.os[i], chromosome.os[j] = chromosome.os[j], chromosome.os[i]

    if rng.random() < ma_mutation_prob:
        ops_by_job = _job_operation_map(instance)
        occurrence = {job.job_id: 0 for job in instance.jobs}
        mutable_positions = []

        for pos, job_id in enumerate(chromosome.os):
            op = ops_by_job[job_id][occurrence[job_id]]
            if len(op.eligible_machines) > 1:
                mutable_positions.append((pos, op))
            occurrence[job_id] += 1

        if mutable_positions:
            pos, op = rng.choice(mutable_positions)
            alternatives = [m for m in op.eligible_machines if m != chromosome.ma[pos]]
            if alternatives:
                # Bias toward useful alternative machines.
                fastest = min(op.proc_times[m] for m in alternatives)
                best = [m for m in alternatives if op.proc_times[m] == fastest]
                chromosome.ma[pos] = rng.choice(best)


def genetic_algorithm(
    instance: Instance,
    population_size: int = 40,
    generations: int = 80,
    crossover_probability: float = 0.85,
    mutation_probability: float = 0.15,
    seed: int = 0,
    return_history: bool = False,
) -> Tuple[List[dict], dict]:
    """
    Conventional GA baseline.

    Returns (schedule, info). `info` includes best makespan per generation.
    """
    if population_size < 2:
        raise ValueError("population_size must be >= 2")
    if generations < 1:
        raise ValueError("generations must be >= 1")

    rng = random.Random(seed)

    population = [
        _make_individual(instance, rng, informed=(i == 0))
        for i in range(population_size)
    ]
    _evaluate_population(instance, population)

    history = [min(ind.makespan for ind in population)]

    for _ in range(generations):
        population.sort(key=lambda x: x.makespan)

        # Elitism: never lose the best schedule.
        elite = population[0].clone()
        new_population = [elite]

        while len(new_population) < population_size:
            p1 = rng.choice(population)
            p2 = rng.choice(population)

            if rng.random() < crossover_probability:
                child_chr = crossover(p1.chromosome, p2.chromosome, rng)
            else:
                child_chr = p1.chromosome.clone()

            repair_ma(instance, child_chr, rng)

            if rng.random() < mutation_probability:
                mutate(child_chr, instance, rng)
                # OS mutation changes which operation each MA position refers to.
                repair_ma(instance, child_chr, rng)

            child = Individual(child_chr)
            child.makespan = chromosome_makespan(instance, child_chr)
            new_population.append(child)

        population = new_population
        history.append(min(ind.makespan for ind in population))

    population.sort(key=lambda x: x.makespan)
    best = population[0]
    info = {
        "algorithm": "GA",
        "seed": seed,
        "population_size": population_size,
        "generations": generations,
        "crossover_probability": crossover_probability,
        "mutation_probability": mutation_probability,
        "best_makespan": best.makespan,
        "history": history if return_history else None,
    }
    return decode_chromosome(instance, best.chromosome), info
