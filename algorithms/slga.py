
"""
Self-Learning Genetic Algorithm (SLGA) inspired by Chen et al. (2020).

This is a repository-sized adaptation of the paper's main idea:
use reinforcement learning to adjust GA crossover probability (Pc) and
mutation probability (Pm) instead of keeping them fixed.

The paper describes a population state using:
  - normalized average fitness,
  - normalized population diversity,
  - normalized best fitness,
weighted 0.35 / 0.35 / 0.30, and a Q-table over discrete states.
It discusses SARSA and Q-learning and uses them to learn parameter-selection
policies.

Here we keep that idea but make the implementation transparent and compact:
  * 20 discrete states from the weighted population-state value.
  * 9 actions, each nudging Pc/Pm up, down, or leaving them unchanged.
  * SARSA in the early phase, Q-learning in the later phase.
  * reward = normalized improvement in best makespan, with a small diversity
    bonus so that a completely stagnant population is not treated as ideal.

This is NOT an exact reproduction of the published experiments.
"""

from __future__ import annotations

from typing import List, Tuple
import random
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from generator.model import Instance
from algorithms.ga import (
    Individual, Chromosome, _make_individual, _evaluate_population,
    chromosome_makespan, crossover, repair_ma, mutate, decode_chromosome,
)


ACTIONS = (
    (-0.05, -0.05),  # reduce both
    (-0.05,  0.00),  # reduce crossover
    (-0.05, +0.05),  # reduce crossover, increase mutation
    ( 0.00, -0.05),  # reduce mutation
    ( 0.00,  0.00),  # unchanged
    ( 0.00, +0.05),  # increase mutation
    (+0.05, -0.05),  # increase crossover, reduce mutation
    (+0.05,  0.00),  # increase crossover
    (+0.05, +0.05),  # increase both
)


def _chromosome_diversity(population: List[Individual]) -> float:
    """
    Mean normalized Hamming distance from the population's best chromosome.

    This is a simple, robust proxy for the paper's population-diversity
    concept. It measures how much genetic material differs from the best
    individual across OS and MA genes.
    """
    if len(population) <= 1:
        return 0.0

    best = min(population, key=lambda x: x.makespan).chromosome
    n = len(best.os) + len(best.ma)
    if n == 0:
        return 0.0

    distances = []
    for ind in population:
        diff = sum(a != b for a, b in zip(ind.chromosome.os, best.os))
        diff += sum(a != b for a, b in zip(ind.chromosome.ma, best.ma))
        distances.append(diff / n)
    return sum(distances) / len(distances)


def _population_state(
    population: List[Individual],
    first_avg: float,
    first_diversity: float,
    first_best: float,
) -> Tuple[int, float, float, float, float]:
    """Return (discrete_state, avg_norm, diversity_norm, best_norm, weighted)."""
    avg = sum(ind.makespan for ind in population) / len(population)
    diversity = _chromosome_diversity(population)
    best = min(ind.makespan for ind in population)

    avg_norm = avg / max(first_avg, 1e-12)
    div_norm = diversity / max(first_diversity, 1e-12)
    best_norm = best / max(first_best, 1e-12)

    weighted = 0.35 * avg_norm + 0.35 * div_norm + 0.30 * best_norm
    state = min(19, max(0, int(weighted * 10)))
    return state, avg_norm, div_norm, best_norm, weighted


def _choose_action(q, state: int, epsilon: float, rng: random.Random) -> int:
    if rng.random() < epsilon:
        return rng.randrange(len(ACTIONS))
    row = q[state]
    best = max(row)
    candidates = [i for i, value in enumerate(row) if value == best]
    return rng.choice(candidates)


def _clamp_parameters(pc: float, pm: float) -> Tuple[float, float]:
    return min(0.98, max(0.50, pc)), min(0.50, max(0.02, pm))


def self_learning_ga(
    instance: Instance,
    population_size: int = 40,
    generations: int = 80,
    initial_crossover_probability: float = 0.85,
    initial_mutation_probability: float = 0.15,
    learning_rate: float = 0.20,
    discount: float = 0.80,
    epsilon: float = 0.20,
    seed: int = 0,
    return_history: bool = False,
) -> Tuple[List[dict], dict]:
    """
    Adaptive GA with a small SARSA -> Q-learning controller.

    The controller chooses how Pc/Pm change at every generation.
    """
    rng = random.Random(seed)

    population = [
        _make_individual(instance, rng, informed=(i == 0))
        for i in range(population_size)
    ]
    _evaluate_population(instance, population)

    first_avg = sum(ind.makespan for ind in population) / len(population)
    first_diversity = _chromosome_diversity(population)
    first_best = min(ind.makespan for ind in population)

    # 20 states x 9 parameter-adjustment actions.
    q = [[0.0 for _ in ACTIONS] for _ in range(20)]

    pc, pm = _clamp_parameters(
        initial_crossover_probability, initial_mutation_probability
    )
    history = [min(ind.makespan for ind in population)]
    parameter_history = [(pc, pm)]

    prev_state, *_ = _population_state(
        population, first_avg, first_diversity, first_best
    )
    prev_action = _choose_action(q, prev_state, epsilon, rng)

    early_generations = max(1, generations // 2)

    for gen in range(generations):
        # Apply current learned action to the GA parameters.
        dpc, dpm = ACTIONS[prev_action]
        pc, pm = _clamp_parameters(pc + dpc, pm + dpm)

        population.sort(key=lambda x: x.makespan)
        elite = population[0].clone()
        new_population = [elite]

        while len(new_population) < population_size:
            p1 = rng.choice(population)
            p2 = rng.choice(population)

            if rng.random() < pc:
                child_chr = crossover(p1.chromosome, p2.chromosome, rng)
            else:
                child_chr = p1.chromosome.clone()

            repair_ma(instance, child_chr, rng)

            if rng.random() < pm:
                mutate(child_chr, instance, rng)
                repair_ma(instance, child_chr, rng)

            child = Individual(child_chr)
            child.makespan = chromosome_makespan(instance, child_chr)
            new_population.append(child)

        population = new_population

        old_best = history[-1]
        new_best = min(ind.makespan for ind in population)
        state, avg_norm, div_norm, best_norm, weighted = _population_state(
            population, first_avg, first_diversity, first_best
        )

        # Positive when the best makespan improves. A tiny diversity term
        # discourages total convergence when no immediate improvement occurs.
        improvement = max(0.0, (old_best - new_best) / max(old_best, 1e-12))
        diversity_reward = 0.01 * min(1.0, div_norm)
        reward = improvement + diversity_reward

        next_action = _choose_action(q, state, epsilon, rng)

        if gen < early_generations:
            # SARSA: use the actually selected next action.
            target = reward + discount * q[state][next_action]
            q[prev_state][prev_action] = (
                (1 - learning_rate) * q[prev_state][prev_action]
                + learning_rate * target
            )
            controller = "SARSA"
        else:
            # Q-learning: use the best next action.
            target = reward + discount * max(q[state])
            q[prev_state][prev_action] = (
                (1 - learning_rate) * q[prev_state][prev_action]
                + learning_rate * target
            )
            controller = "Q-learning"

        prev_state = state
        prev_action = next_action
        history.append(new_best)
        parameter_history.append((pc, pm))

    population.sort(key=lambda x: x.makespan)
    best = population[0]

    info = {
        "algorithm": "SLGA",
        "seed": seed,
        "population_size": population_size,
        "generations": generations,
        "initial_crossover_probability": initial_crossover_probability,
        "initial_mutation_probability": initial_mutation_probability,
        "learning_rate": learning_rate,
        "discount": discount,
        "epsilon": epsilon,
        "best_makespan": best.makespan,
        "history": history if return_history else None,
        "parameter_history": parameter_history if return_history else None,
        "q_table": q,
        "controller": "SARSA -> Q-learning",
        "state_weights": (0.35, 0.35, 0.30),
    }
    return decode_chromosome(instance, best.chromosome), info
