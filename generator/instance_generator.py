"""
FJSPGenerator: hand-written random instance generator for Flexible Job-Shop
Scheduling.

DESIGN PRINCIPLES
------------------
1. Validity by construction. There is no "generate then repair/check" step.
   Every operation is assigned a non-empty eligible-machine set, and
   processing times are defined for EXACTLY that set, at the moment of
   creation. It is structurally impossible for generate() to emit a
   malformed instance.

2. One algorithm, many presets. There is a single generate() method driven
   by numeric knobs. "Bottleneck-heavy", "high-flexibility", etc. are just
   named parameter presets (see INSTANCE_CLASS_PRESETS below) -- not special
   cased code paths. This keeps the generator's logic auditable in one place.

3. Full reproducibility. All randomness is drawn from a single
   random.Random(seed) instance. Same seed + same params -> byte-identical
   instance, forever.
"""

import random
import os
import sys
from typing import Optional, Tuple, Union

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from generator.model import Instance, Job, Operation


class FJSPGenerator:

    def __init__(self, seed: int):
        self.seed = seed
        self.rng = random.Random(seed)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def generate(
        self,
        number_of_jobs: int,
        number_of_machines: int,
        operations_per_job: Union[int, Tuple[int, int]],
        machine_flexibility: float,              # in (0, 1]: avg fraction of machines eligible per op
        processing_time_range: Tuple[int, int] = (1, 20),
        processing_time_variance: float = 1.0,    # 1.0 = full range spread; <1 tighter, >1 wider(clipped)
        bottleneck_probability: float = 0.0,      # chance an op is forced onto the bottleneck machine
        machine_advantage: Optional[dict] = None, # {machine_id: multiplier}, e.g. {3: 0.25} = machine 3 is 4x faster
        instance_class: str = "custom",
    ) -> Instance:

        assert 0 < machine_flexibility <= 1, "machine_flexibility must be in (0, 1]"
        assert 0 <= bottleneck_probability <= 1

        params = dict(
            number_of_jobs=number_of_jobs,
            number_of_machines=number_of_machines,
            operations_per_job=operations_per_job,
            machine_flexibility=machine_flexibility,
            processing_time_range=processing_time_range,
            processing_time_variance=processing_time_variance,
            bottleneck_probability=bottleneck_probability,
            machine_advantage=machine_advantage,
        )

        machines = list(range(1, number_of_machines + 1))
        # One bottleneck machine is chosen PER INSTANCE (not re-rolled per
        # operation) -- that's what makes it a genuine contention point
        # rather than noise.
        bottleneck_machine = self.rng.choice(machines)

        jobs = []
        for j in range(1, number_of_jobs + 1):
            k_j = self._sample_op_count(operations_per_job)
            ops = []
            for k in range(1, k_j + 1):
                elig = self._sample_eligible_machines(
                    machines, machine_flexibility, bottleneck_machine, bottleneck_probability
                )
                proc_times = self._sample_proc_times(
                    elig, processing_time_range, processing_time_variance, machine_advantage
                )
                ops.append(Operation(job_id=j, op_id=k, eligible_machines=elig, proc_times=proc_times))
            jobs.append(Job(job_id=j, operations=ops))

        return Instance(
            num_machines=number_of_machines,
            jobs=jobs,
            seed=self.seed,
            params=params,
            instance_class=instance_class,
        )

    def generate_preset(self, preset_name: str, number_of_jobs: int, number_of_machines: int,
                         operations_per_job: Union[int, Tuple[int, int]]) -> Instance:
        """Convenience wrapper: apply a named preset from INSTANCE_CLASS_PRESETS on top
        of the given size parameters."""
        preset = INSTANCE_CLASS_PRESETS[preset_name]
        return self.generate(
            number_of_jobs=number_of_jobs,
            number_of_machines=number_of_machines,
            operations_per_job=operations_per_job,
            instance_class=preset_name,
            **preset,
        )

    # ------------------------------------------------------------------ #
    # Sampling internals
    # ------------------------------------------------------------------ #

    def _sample_op_count(self, operations_per_job):
        if isinstance(operations_per_job, tuple):
            lo, hi = operations_per_job
            return self.rng.randint(lo, hi)
        return operations_per_job

    def _sample_eligible_machines(self, machines, flexibility, bottleneck_machine, bottleneck_probability):
        """
        How E(j,k) is sampled:
        - `flexibility` (0,1] sets the TARGET average size as a fraction of
          all machines: target = round(flexibility * m).
            flexibility=0.15 on 10 machines -> ~1-2 eligible machines (rigid)
            flexibility=0.8  on 10 machines -> ~8 eligible machines (very flexible)
        - a small +-1 jitter is added so instances aren't perfectly uniform
          (every operation having exactly the same |E| would be unrealistic).
        - separately, with `bottleneck_probability`, the pre-chosen
          bottleneck machine for this instance is force-added to E(j,k).
          This is INDEPENDENT of flexibility -- it's how contention gets
          injected even into otherwise-flexible instances.
        """
        m = len(machines)
        target = max(1, round(flexibility * m))
        jitter = self.rng.choice([-1, 0, 0, 1])
        size = min(m, max(1, target + jitter))

        elig = set(self.rng.sample(machines, size))

        if self.rng.random() < bottleneck_probability:
            elig.add(bottleneck_machine)

        return sorted(elig)

    def _sample_proc_times(self, eligible, time_range, variance, machine_advantage):
        """
        How processing times are sampled:
        - base = midpoint of processing_time_range
        - spread = half-width of the range, scaled by `variance`
              variance=1.0 -> spread covers the full given range
              variance=0.3 -> durations cluster tightly near the midpoint
              variance=2.0 -> spread is wider (still clipped into range)
        - draw from a TRIANGULAR distribution peaked at `base` -- this gives
          realistic clustering around a typical duration instead of flat
          uniform noise, while still allowing outliers near the edges.
        - round to nearest int, clip into [1, hi] to guarantee integer > 0.
        - `machine_advantage`, if given, scales specific machines' times
          down (or up) AFTER the base sample, to deliberately create a
          "this machine is just better" instance.
        """
        lo, hi = time_range
        base = (lo + hi) / 2
        spread = (hi - lo) / 2 * variance

        times = {}
        for m in eligible:
            low = max(1, base - spread)
            high = base + spread
            raw = self.rng.triangular(low=low, high=high, mode=base)
            val = int(round(raw))
            val = max(1, min(hi, val))
            if machine_advantage and m in machine_advantage:
                val = max(1, int(round(val * machine_advantage[m])))
            times[m] = val
        return times


# ---------------------------------------------------------------------- #
# Named instance classes -> concrete parameter presets.
# These map directly to Part D's "Instance Type" table.
# Size params (jobs/machines/ops_per_job) are passed separately via
# generate_preset() so the same preset can be reused at different scales.
# ---------------------------------------------------------------------- #

INSTANCE_CLASS_PRESETS = {
    "average": dict(
        machine_flexibility=0.4, processing_time_range=(1, 20),
        processing_time_variance=1.0, bottleneck_probability=0.05,
    ),
    "low_flexibility": dict(
        machine_flexibility=0.15, processing_time_range=(1, 20),
        processing_time_variance=1.0, bottleneck_probability=0.0,
    ),
    "high_flexibility": dict(
        machine_flexibility=0.85, processing_time_range=(1, 20),
        processing_time_variance=1.0, bottleneck_probability=0.0,
    ),
    "bottleneck_heavy": dict(
        machine_flexibility=0.4, processing_time_range=(1, 20),
        processing_time_variance=1.0, bottleneck_probability=0.7,
    ),
    "balanced": dict(
        machine_flexibility=0.4, processing_time_range=(8, 12),
        processing_time_variance=0.5, bottleneck_probability=0.0,
    ),
    "high_variance": dict(
        machine_flexibility=0.4, processing_time_range=(1, 50),
        processing_time_variance=2.0, bottleneck_probability=0.0,
    ),
    "machine_advantage": dict(
        machine_flexibility=0.5, processing_time_range=(5, 20),
        processing_time_variance=1.0, bottleneck_probability=0.0,
        machine_advantage={1: 0.2},   # machine 1 is ~5x faster wherever eligible
    ),
    "hard": dict(
        machine_flexibility=0.6, processing_time_range=(1, 40),
        processing_time_variance=1.6, bottleneck_probability=0.4,
    ),
    "extreme": dict(
        machine_flexibility=0.9, processing_time_range=(1, 200),
        processing_time_variance=3.0, bottleneck_probability=0.6,
        machine_advantage={1: 0.05},
    ),
}