---
name: reference
description: Find NeuroGolf reference implementations and cross-reference task IDs across NeuroGolf, ARC-GEN, and ARC-DSL.
---

# Reference skill

Use this skill when an agent needs inspiration or prior art for a NeuroGolf task before designing an ONNX solution.

The reference files are for inspection only. Submitted Python must stay self-contained: do not import these scripts or read `mapping.csv` at runtime.

## What is included

- `mapping.csv`: maps `taskXXX` to ARC-GEN ids, ARC-DSL ids, and ARC-DSL solver paths.
- `scripts/arc-dsl/`: ARC-DSL primitives and solver scripts.
- `scripts/ARC-GEN/`: ARC-GEN generator code, including per-task generators.

Licenses from the source projects are included in their copied folders.

## Setup

Set `SKILL_DIR` to this skill folder, then run commands from any directory using the `$SKILL_DIR` paths shown below. This skill has no runtime dependencies beyond the Python standard library.

## Look up references for one task

```bash
python "$SKILL_DIR/scripts/reference_lookup.py" task123
```

The command prints JSON with the mapped task ids and candidate reference file paths:

```json
{
  "neurogolf_task_id": "task123",
  "arc_gen_task_id": "539a4f51",
  "arc_dsl_task_id": "539a4f51",
  "references": {
    "arc_dsl_solver": {
      "path": "scripts/arc-dsl/solver_scripts/solve_539a4f51.py",
      "exists": true
    },
    "arc_gen_generator": {
      "path": "scripts/ARC-GEN/tasks/task_539a4f51.py",
      "exists": true
    }
  }
}
```

Read the existing files that look relevant. ARC-DSL usually gives a clear transformation recipe; ARC-GEN shows generator constraints and the kinds of inputs the task expects.
