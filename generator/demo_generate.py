"""Quick smoke test: generate one instance per class, print a summary, save to JSON."""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generator.instance_generator import FJSPGenerator, INSTANCE_CLASS_PRESETS

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "instances")
os.makedirs(OUT_DIR, exist_ok=True)

for cls_name in INSTANCE_CLASS_PRESETS:
    gen = FJSPGenerator(seed=42)
    inst = gen.generate_preset(
        cls_name, number_of_jobs=6, number_of_machines=5, operations_per_job=(2, 4)
    )
    path = os.path.join(OUT_DIR, f"{cls_name}_seed42.json")
    inst.save(path)

    elig_sizes = [len(op.eligible_machines) for job in inst.jobs for op in job.operations]
    all_times = [t for job in inst.jobs for op in job.operations for t in op.proc_times.values()]
    print(f"{cls_name:20s} | jobs={len(inst.jobs):2d} ops={inst.total_operations():3d} "
          f"| avg|E|={sum(elig_sizes)/len(elig_sizes):.2f} "
          f"| time range=[{min(all_times)},{max(all_times)}] -> {path}")