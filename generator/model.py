"""
Core data model for FJSP instances.

An Instance is just: how many machines, and a list of Jobs.
A Job is: an ordered list of Operations (the precedence chain).
An Operation is: which machines can do it, and how long it takes on each.

This module has NO randomness in it -- it's pure structure + serialization.
All randomness lives in instance_generator.py.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import json


@dataclass
class Operation:
    job_id: int
    op_id: int  # position within the job, 1-indexed
    eligible_machines: List[int]     # E(j,k)  -- non-empty by construction
    proc_times: Dict[int, int]       # p(j,k,m) for every m in eligible_machines, and ONLY those

    def __post_init__(self):
        # Cheap internal sanity checks -- NOT a substitute for the real
        # validator in Part B, just guards against generator bugs early.
        assert len(self.eligible_machines) >= 1, "operation has no eligible machine"
        assert set(self.eligible_machines) == set(self.proc_times.keys()), \
            "proc_times must be defined for exactly the eligible machines"
        assert all(t > 0 for t in self.proc_times.values()), "processing time must be > 0"

    def to_dict(self):
        return {
            "job_id": self.job_id,
            "op_id": self.op_id,
            "eligible_machines": self.eligible_machines,
            "proc_times": {str(k): v for k, v in self.proc_times.items()},
        }

    @staticmethod
    def from_dict(d):
        proc_times = {int(k): v for k, v in d["proc_times"].items()}
        return Operation(d["job_id"], d["op_id"], list(d["eligible_machines"]), proc_times)


@dataclass
class Job:
    job_id: int
    operations: List[Operation]  # O(j, 1..k_j), already in the correct order

    def to_dict(self):
        return {"job_id": self.job_id, "operations": [op.to_dict() for op in self.operations]}

    @staticmethod
    def from_dict(d):
        return Job(d["job_id"], [Operation.from_dict(od) for od in d["operations"]])


@dataclass
class Instance:
    num_machines: int
    jobs: List[Job]
    seed: int
    params: dict              # every generator parameter used to build this instance
    instance_class: str = "custom"   # e.g. "bottleneck_heavy", "high_flexibility", ...

    def total_operations(self) -> int:
        return sum(len(j.operations) for j in self.jobs)

    def to_dict(self):
        return {
            "meta": {
                "seed": self.seed,
                "instance_class": self.instance_class,
                "params": self.params,
                "num_machines": self.num_machines,
                "num_jobs": len(self.jobs),
                "total_operations": self.total_operations(),
            },
            "jobs": [j.to_dict() for j in self.jobs],
        }

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @staticmethod
    def load(path: str) -> "Instance":
        with open(path) as f:
            d = json.load(f)
        jobs = [Job.from_dict(jd) for jd in d["jobs"]]
        return Instance(
            num_machines=d["meta"]["num_machines"],
            jobs=jobs,
            seed=d["meta"]["seed"],
            params=d["meta"]["params"],
            instance_class=d["meta"]["instance_class"],
        )