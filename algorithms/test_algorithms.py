
"""Small executable tests for the GA and SLGA implementations."""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generator.instance_generator import FJSPGenerator
from algorithms.ga import (
    _make_individual, decode_chromosome, chromosome_makespan
)
from algorithms.ga import genetic_algorithm
from algorithms.slga import self_learning_ga
from algorithms.greedy import schedule_global_ect
from environments.validator import validate_schedule


def check_algorithm(fn, instance, seed):
    schedule, info = fn(
        instance,
        population_size=12,
        generations=12,
        seed=seed,
        return_history=True,
    )
    result = validate_schedule(instance, schedule)
    assert result.is_valid, result.errors
    assert result.makespan == info["best_makespan"]
    assert len(info["history"]) == 13
    return result.makespan


def main():
    inst = FJSPGenerator(123).generate_preset(
        "high_flexibility", 6, 5, (2, 4)
    )

    # Chromosome invariants.
    ind = _make_individual(inst, __import__("random").Random(1), informed=False)
    assert len(ind.chromosome.os) == inst.total_operations()
    assert len(ind.chromosome.ma) == inst.total_operations()
    schedule = decode_chromosome(inst, ind.chromosome)
    assert validate_schedule(inst, schedule).is_valid

    greedy = validate_schedule(inst, schedule_global_ect(inst)).makespan
    ga = check_algorithm(genetic_algorithm, inst, 7)
    slga = check_algorithm(self_learning_ga, inst, 7)

    # Baseline-seeding + elitism guarantee.
    assert ga <= greedy
    assert slga <= greedy

    print("GA/SLGA tests: PASS")
    print(f"Greedy={greedy}, GA={ga}, SLGA={slga}")


if __name__ == "__main__":
    main()
