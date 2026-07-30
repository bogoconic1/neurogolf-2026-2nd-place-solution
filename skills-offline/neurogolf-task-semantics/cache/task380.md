# task380 Semantics

## Sources

- Current champion builder: `solutions_py/task380.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task380.json`
- ARC-GEN task id: `ed36ccf7`
- ARC-DSL task id: `ed36ccf7`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_ed36ccf7.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_ed36ccf7.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task380.py`

## Pattern

Input and output are 3x3 ARC grids. The input has 2 to 8 colored cells, all of the same nonzero color, on a black `0` background. The output is the input rotated 90 degrees counterclockwise (`rot270`): an input cell at row `r`, column `c` moves to row `size - 1 - c`, column `r`.

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])
    assert h == w == 3
    out = [[0 for _ in range(w)] for _ in range(h)]
    for r in range(h):
        for c in range(w):
            out[h - 1 - c][r] = grid[r][c]
    return out
```

## Generator Constraints

ARC-GEN uses `size=3`. Random instances sample 2 to 8 distinct pixels from the 3x3 canvas and choose one random nonzero color. All selected cells are painted with that color; all other cells remain black. There is no object overlap issue because selected pixels are unique. The output applies the fixed coordinate map above and preserves the same single foreground color.

## Reference Notes

The ARC-DSL solver is exactly `rot270(I)`. The Code Golf 2025 solution `[*zip(*g)][::-1]` is the same 90-degree counterclockwise rotation. The generator confirms there are no multi-color, variable-size, or tie-breaking cases; the only dynamic content is the foreground color and which 3x3 cells are active.
