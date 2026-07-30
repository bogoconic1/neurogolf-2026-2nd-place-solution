# task040 Semantics

## Sources

- Current champion builder: `solutions_py/task040.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task040.json`
- ARC-GEN task id: `2204b7a8`
- ARC-DSL task id: `2204b7a8`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_2204b7a8.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_2204b7a8.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task040.py`

## Pattern

The visible ARC grid is `10x10`. Two non-green colors define opposite border lines: either left/right vertical borders or, after transpose, top/bottom horizontal borders. Interior green cells are recolored by the border color of their half of the grid. All non-green cells are preserved.

Equivalently, each green cell is replaced by the color at its quadrant corner: top-left `(0,0)`, top-right `(0,9)`, bottom-left `(9,0)`, or bottom-right `(9,9)`.

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    for r in range(h):
        for c in range(w):
            if grid[r][c] == 3:
                rr = 0 if r < h // 2 else h - 1
                cc = 0 if c < w // 2 else w - 1
                out[r][c] = grid[rr][cc]
    return out
```

## Generator Constraints

ARC-GEN samples two different colors excluding green, places them as the two opposite border lines, samples 2..10 interior green pixels in columns `1..8`, and optionally transposes the entire input/output. The border colors are therefore always available at the four corners, and green never appears as a corner color.

## Reference Notes

The ARC-DSL solver decides whether the prominent line objects are vertical or horizontal, splits the grid into halves, replaces green in each half with the corresponding edge/corner color, and concatenates the halves back together. The Code Golf solution uses recursive/list operations to perform the same side-based green replacement.
