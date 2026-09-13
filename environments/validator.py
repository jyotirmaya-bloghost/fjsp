"""
Independent FJSP schedule validator.

This module trusts NOTHING from whatever produced the schedule. It is given:
  1. an Instance (the ground-truth problem: eligible machines, processing times)
  2. a proposed Schedule (a flat list of scheduled operations)

...and re-derives everything from first principles: which machine each
operation legally could use, how long it should take, what order jobs and
machines require -- then checks the proposed schedule against all of that.

It never calls into the algorithm's internal state. It only reads the
instance data and the schedule's own numbers.
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple
import json
import os
import sys

# Add the fjsp/ project root (parent of this file's folder) to the path,
# regardless of the caller's working directory, so package-style imports
# below always resolve the same way.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generator.model import Instance, Operation


@dataclass
class ScheduledOp:
    job_id: int
    op_id: int
    machine_id: int
    start_time: int
    completion_time: int


class ValidationResult:
    def __init__(self):
        self.errors: List[str] = []
        self.machine_schedules: Dict[int, List[ScheduledOp]] = {}
        self.job_schedules: Dict[int, List[ScheduledOp]] = {}
        self.job_completion_times: Dict[int, int] = {}
        self.makespan: int = 0

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0

    def add_error(self, msg: str):
        self.errors.append(msg)

    def report(self) -> str:
        lines = []
        if self.is_valid:
            lines.append("VALID")
            lines.append(f"Makespan (C_max): {self.makespan}")
            lines.append("")
            lines.append("Job completion times:")
            for j in sorted(self.job_completion_times):
                lines.append(f"  Job {j}: C_j = {self.job_completion_times[j]}")
            lines.append("")
            lines.append("Machine schedules:")
            for m in sorted(self.machine_schedules):
                ops = sorted(self.machine_schedules[m], key=lambda o: o.start_time)
                seq = ", ".join(f"J{o.job_id}Op{o.op_id}[{o.start_time},{o.completion_time}]" for o in ops)
                lines.append(f"  Machine {m}: {seq}")
        else:
            lines.append("INVALID")
            lines.append(f"{len(self.errors)} error(s) found:")
            for e in self.errors:
                lines.append(f"  Error: {e}")
        return "\n".join(lines)


def validate_schedule(instance: Instance, schedule: List[dict]) -> ValidationResult:
    """
    schedule: list of dicts, each with keys
        job_id, operation_id, machine_id, start_time, completion_time
    (operation_id maps to Operation.op_id)
    """
    result = ValidationResult()

    # ---- 0. Parse / malformed-schedule check --------------------------- #
    parsed: List[ScheduledOp] = []
    required_keys = {"job_id", "operation_id", "machine_id", "start_time", "completion_time"}
    for i, entry in enumerate(schedule):
        missing = required_keys - set(entry.keys())
        if missing:
            result.add_error(f"Entry {i} is malformed, missing fields: {sorted(missing)}")
            continue
        try:
            parsed.append(ScheduledOp(
                job_id=int(entry["job_id"]),
                op_id=int(entry["operation_id"]),
                machine_id=int(entry["machine_id"]),
                start_time=int(entry["start_time"]),
                completion_time=int(entry["completion_time"]),
            ))
        except (ValueError, TypeError):
            result.add_error(f"Entry {i} has non-numeric field(s): {entry}")

    # Build a lookup of the ground truth from the instance.
    op_lookup: Dict[Tuple[int, int], Operation] = {}
    valid_job_ids = set()
    for job in instance.jobs:
        valid_job_ids.add(job.job_id)
        for op in job.operations:
            op_lookup[(job.job_id, op.op_id)] = op
    valid_machine_ids = set(range(1, instance.num_machines + 1))

    # ---- 1. Unknown IDs, ineligible machine, wrong duration ------------- #
    seen_ops = set()
    duplicates = set()
    for s in parsed:
        key = (s.job_id, s.op_id)

        if s.job_id not in valid_job_ids:
            result.add_error(f"Unknown job_id {s.job_id} (Job {s.job_id} Op {s.op_id})")
            continue
        if key not in op_lookup:
            result.add_error(f"Unknown operation_id {s.op_id} for job {s.job_id}")
            continue

        if key in seen_ops:
            duplicates.add(key)
        seen_ops.add(key)

        if s.machine_id not in valid_machine_ids:
            result.add_error(f"Job {s.job_id} Op {s.op_id}: unknown machine_id {s.machine_id}")
            continue

        op = op_lookup[key]

        if s.machine_id not in op.eligible_machines:
            result.add_error(
                f"Job {s.job_id} Op {s.op_id}: assigned to ineligible machine {s.machine_id} "
                f"(eligible: {op.eligible_machines})"
            )

        if s.start_time < 0:
            result.add_error(f"Job {s.job_id} Op {s.op_id}: negative start time {s.start_time}")

        expected_dur = op.proc_times.get(s.machine_id)
        if expected_dur is not None:
            actual_dur = s.completion_time - s.start_time
            if actual_dur != expected_dur:
                result.add_error(
                    f"Job {s.job_id} Op {s.op_id}: duration mismatch on machine {s.machine_id} "
                    f"(scheduled {actual_dur}, expected {expected_dur})"
                )

    for (j, k) in sorted(duplicates):
        result.add_error(f"Job {j} Op {k}: appears more than once in the schedule")

    # ---- 2. Completeness: every operation of every job appears exactly once --- #
    for job in instance.jobs:
        for op in job.operations:
            key = (job.job_id, op.op_id)
            if key not in seen_ops:
                result.add_error(f"Job {job.job_id} Op {op.op_id}: missing from schedule")

    # If structure is already broken, stop before ordering checks (they'd be noise).
    if not result.is_valid:
        return result

    # ---- 3. Job precedence: op k+1 can't start before op k finishes ---- #
    by_job: Dict[int, List[ScheduledOp]] = {}
    for s in parsed:
        by_job.setdefault(s.job_id, []).append(s)

    for job_id, ops in by_job.items():
        ops_sorted = sorted(ops, key=lambda o: o.op_id)
        for a, b in zip(ops_sorted, ops_sorted[1:]):
            if b.start_time < a.completion_time:
                result.add_error(
                    f"Job {job_id}: precedence violated -- Op {b.op_id} starts at {b.start_time} "
                    f"before Op {a.op_id} finishes at {a.completion_time}"
                )

    # ---- 4. Machine non-overlap ----------------------------------------- #
    by_machine: Dict[int, List[ScheduledOp]] = {}
    for s in parsed:
        by_machine.setdefault(s.machine_id, []).append(s)

    for machine_id, ops in by_machine.items():
        ops_sorted = sorted(ops, key=lambda o: o.start_time)
        for a, b in zip(ops_sorted, ops_sorted[1:]):
            if a.completion_time > b.start_time:
                result.add_error(
                    f"Machine {machine_id} contains overlapping operations:\n"
                    f"    Job {a.job_id} Op {a.op_id}: [{a.start_time}, {a.completion_time}]\n"
                    f"    Job {b.job_id} Op {b.op_id}: [{b.start_time}, {b.completion_time}]"
                )

    if not result.is_valid:
        return result

    # ---- 5. Reconstruction (only if fully valid) ------------------------ #
    result.machine_schedules = by_machine
    result.job_schedules = by_job
    for job_id, ops in by_job.items():
        result.job_completion_times[job_id] = max(o.completion_time for o in ops)
    result.makespan = max(result.job_completion_times.values())

    return result


def validate_schedule_file(instance_path: str, schedule_path: str) -> ValidationResult:
    instance = Instance.load(instance_path)
    with open(schedule_path) as f:
        schedule = json.load(f)
    return validate_schedule(instance, schedule)