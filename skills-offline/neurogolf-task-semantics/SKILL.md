---
name: neurogolf-task-semantics
description: Maintain a durable per-task NeuroGolf semantic cache containing an English pattern description, readable Python solve(grid) implementation, ARC-GEN/ARC-DSL/Code Golf 2025 reference paths, generator constraints, and edge cases. Use before optimizing any NeuroGolf task or sweep task.
---

# NeuroGolf Task Semantics

Use this skill before graph optimization. The cache records the task rule in human terms so later agents do not re-derive it from scratch or anchor too early on the current champion graph.

## Cache Path

Set `SEMANTICS_SKILL_DIR` to this skill folder; the cache lives at:

```text
$SEMANTICS_SKILL_DIR/cache/taskNNN.md
```

## Workflow

```bash
python "$SEMANTICS_SKILL_DIR/scripts/task_semantics.py" status taskNNN
```

If `cache_exists=true`, read it:

```bash
python "$SEMANTICS_SKILL_DIR/scripts/task_semantics.py" show taskNNN
```

If missing, inspect the ARC-GEN generator, ARC-DSL solver, and Code Golf 2025 solution reported by `status`, then create and fill the cache:

```bash
python "$SEMANTICS_SKILL_DIR/scripts/task_semantics.py" init taskNNN
```

Replace every `TODO` with a concrete description. The readable Python must be a clear standalone `solve(grid)` sketch for understanding; it is not the submitted ONNX builder.

Validate before continuing:

```bash
python "$SEMANTICS_SKILL_DIR/scripts/task_semantics.py" validate taskNNN
```

## Required Cache Content

- `Pattern`: concise English transformation rule.
- `Readable Python Solver`: a readable `solve(grid)` function or pseudocode-like Python.
- `Generator Constraints`: sizes, colors, geometry, randomness, and edge cases from ARC-GEN.
- `Reference Notes`: what ARC-DSL and Code Golf 2025 clarify or contradict.
- `Optimization Implications`: likely compact state, memory floor, and risky shortcuts.

Reference files are inspection-only. Do not import them or read this cache from submitted builders.
