# task179 Semantics

## Sources

- Current champion builder: `solutions_py/task179.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task179.json`
- ARC-GEN task id: `74dd1130`
- ARC-DSL task id: `74dd1130`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_74dd1130.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_74dd1130.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task179.py`

## Pattern

The input is a square color grid, generated as a 3x3 grid for the canonical examples. The output is the input reflected across the main diagonal, i.e. the matrix transpose. Every color at row r, column c moves to row c, column r. There is no color filtering, object selection, or tie-breaking.

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])
    return [[grid[r][c] for r in range(h)] for c in range(w)]
```

## Generator Constraints

ARC-GEN creates a square grid with `size=3` by default. It chooses three random nonzero ARC colors, then fills each of the nine cells by sampling from those colors. The train/test fixtures are all 3x3. The generator implementation supports a `size` argument, but the task examples use the 3x3 square case. The output is always the transpose of the full grid.

## Reference Notes

ARC-DSL is the single operation `dmirror(I)`, which mirrors the grid over the main diagonal. The Code Golf 2025 solution is `zip(*g)`, the standard Python transpose. These agree exactly with the ARC-GEN output construction.
