# task194 Semantics

## Sources

- Current champion builder: `solutions_py/task194.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task194.json`
- ARC-GEN task id: `7fe24cdd`
- ARC-DSL task id: `7fe24cdd`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_7fe24cdd.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_7fe24cdd.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task194.py`

## Pattern

The input is a 3x3 grid embedded at the upper-left of the standard 30x30 NeuroGolf canvas. It contains 5 to 9 colored cells using 2 to 4 non-background colors. The output is a 6x6 rotation mosaic placed at the upper-left of the NeuroGolf output canvas:

- top-left quadrant: the original 3x3 input
- top-right quadrant: the input rotated 90 degrees clockwise
- bottom-left quadrant: the input rotated 270 degrees clockwise
- bottom-right quadrant: the input rotated 180 degrees

All cells outside the 6x6 mosaic are no-color/clear in NeuroGolf output terms.

## Readable Python Solver

```python
def solve(grid):
    g = [row[:3] for row in grid[:3]]
    out = [[0 for _ in range(6)] for _ in range(6)]
    for r in range(3):
        for c in range(3):
            v = g[r][c]
            out[r][c] = v                  # original
            out[c][5 - r] = v              # rot90 clockwise
            out[5 - c][r] = v              # rot270 clockwise
            out[5 - r][5 - c] = v          # rot180
    return out
```

## Generator Constraints

ARC-GEN samples a 3x3 grid. It chooses 5 to 9 occupied pixels, 2 to 4 random colors, and assigns each occupied pixel one of those colors. Unoccupied cells remain black/background. The output is always 6x6 and is formed by writing every input cell, including black cells, into the four rotated quadrant positions. There is no dynamic size, no object extraction, and no tie-breaking branch.

## Reference Notes

The ARC-DSL solver is exactly `vconcat(hconcat(I, rot90(I)), hconcat(rot270(I), rot180(I)))`. The Code Golf 2025 solution is a compact transpose/zip expression that builds the same four-rotation mosaic. The references agree that the full 3x3 input, including black cells, is part of the transformation.
