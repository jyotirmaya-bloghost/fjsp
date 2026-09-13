"""
Proves the validator works by:
  1. Building a tiny 2-job, 2-machine instance by hand.
  2. Constructing a genuinely valid schedule for it -> expect VALID.
  3. Breaking that schedule in 5 different specific ways -> expect INVALID,
     each with the right diagnostic.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generator.model import Instance, Job, Operation
from environments.validator import validate_schedule

# ---- Hand-built tiny instance ---- #
# Job 1: Op1 (M1:5, M2:8) -> Op2 (M1:4, M2:3)
# Job 2: Op1 (M2:6)       -> Op2 (M1:3, M2:5)
job1 = Job(1, [
    Operation(1, 1, [1, 2], {1: 5, 2: 8}),
    Operation(1, 2, [1, 2], {1: 4, 2: 3}),
])
job2 = Job(2, [
    Operation(2, 1, [2], {2: 6}),
    Operation(2, 2, [1, 2], {1: 3, 2: 5}),
])
instance = Instance(num_machines=2, jobs=[job1, job2], seed=1, params={}, instance_class="handbuilt")

# ---- A genuinely valid schedule ---- #
valid_schedule = [
    {"job_id": 1, "operation_id": 1, "machine_id": 1, "start_time": 0, "completion_time": 5},
    {"job_id": 1, "operation_id": 2, "machine_id": 2, "start_time": 6, "completion_time": 9},
    {"job_id": 2, "operation_id": 1, "machine_id": 2, "start_time": 0, "completion_time": 6},
    {"job_id": 2, "operation_id": 2, "machine_id": 1, "start_time": 6, "completion_time": 9},
]

print("=" * 60)
print("TEST 1: valid schedule")
print("=" * 60)
r = validate_schedule(instance, valid_schedule)
print(r.report())

print()
print("=" * 60)
print("TEST 2: machine overlap (both use M2 at overlapping times)")
print("=" * 60)
broken = [dict(x) for x in valid_schedule]
broken[1] = {"job_id": 1, "operation_id": 2, "machine_id": 2, "start_time": 5, "completion_time": 8}  # overlaps J2Op1 [0,6] on M2
r = validate_schedule(instance, broken)
print(r.report())

print()
print("=" * 60)
print("TEST 3: ineligible machine (J2Op1 forced onto M1, not eligible)")
print("=" * 60)
broken = [dict(x) for x in valid_schedule]
broken[2] = {"job_id": 2, "operation_id": 1, "machine_id": 1, "start_time": 0, "completion_time": 6}
r = validate_schedule(instance, broken)
print(r.report())

print()
print("=" * 60)
print("TEST 4: precedence violation (J1Op2 starts before J1Op1 finishes)")
print("=" * 60)
broken = [dict(x) for x in valid_schedule]
broken[1] = {"job_id": 1, "operation_id": 2, "machine_id": 2, "start_time": 2, "completion_time": 5}
r = validate_schedule(instance, broken)
print(r.report())

print()
print("=" * 60)
print("TEST 5: wrong duration (J1Op1 claims 3 units on M1 instead of 5)")
print("=" * 60)
broken = [dict(x) for x in valid_schedule]
broken[0] = {"job_id": 1, "operation_id": 1, "machine_id": 1, "start_time": 0, "completion_time": 3}
r = validate_schedule(instance, broken)
print(r.report())

print()
print("=" * 60)
print("TEST 6: missing operation (J2Op2 dropped entirely)")
print("=" * 60)
broken = valid_schedule[:3]
r = validate_schedule(instance, broken)
print(r.report())

print()
print("=" * 60)
print("TEST 7: negative start time")
print("=" * 60)
broken = [dict(x) for x in valid_schedule]
broken[0] = {"job_id": 1, "operation_id": 1, "machine_id": 1, "start_time": -2, "completion_time": 3}
r = validate_schedule(instance, broken)
print(r.report())