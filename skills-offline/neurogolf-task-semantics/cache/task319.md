# task319 Semantics

## Sources

- Current champion builder: `solutions_py/task319.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task319.json`
- ARC-GEN task id: `ce602527`
- ARC-DSL task id: `ce602527`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_ce602527.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_ce602527.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task319.py`

## Pattern

The input contains two small same-color sprite objects and one 2x magnified copy
of one of those sprites in a separate magnification color, on a background color.
The magnified copy is deliberately partly outside the input frame, so only part
of it is visible. The task is to identify which small sprite matches the visible
2x magnified pattern and output that original small sprite crop in its own color
on the original background.

## Readable Python Solver

The previous cache implementation did not survive replay. The compact executable oracle below is the bundled Code Golf 2025 reference; the Pattern section remains the readable specification.

```python
p = lambda g, h=0: [[L[-(e != L[0])] for e in r] for *r, in zip(*(h or p(g, g))) if (L := sorted(set((w := sum(g, []))), key=w.count)[(o := hash((*w, *b'$,\x04>N'))) // b'=\x16=w~Y`NyFgvkM_~hqx2g\x15`\tQyrjqx~~}#Km<\x07q~\x7fML'[o % 43] & 1:])[0] in r]


def solve(grid):
    result = p([list(row) for row in grid])
    return [list(row) for row in result]
```

## Generator Constraints

ARC-GEN samples input width/height in `[15, 19]`, a background color, a magnified
color, and two distinct sprite colors. Each original sprite has width/tall in
`[3,5]`, is diagonally connected, and is placed in the frame without overlap.
The magnified copy is 2x scale of sprite index 0 and is placed so that part of
it is outside one side of the frame; examples are rejected unless some of the
magnified copy is hidden and the visible magnification does not collide with
other non-background cells. The output is the original unscaled sprite-0 crop.

## Reference Notes

The ARC-DSL solver mirrors vertically, partitions foreground objects, takes the largest object as the magnified object, scores the remaining objects by the intersection between their 2x normalized shape and the normalized magnified shape, then extracts the winning object's subgrid and mirrors back. The Code Golf solution uses color-frequency/hash tricks to choose the relevant object, but the generator confirms the semantic basis is matching a partially visible 2x magnified sprite.
