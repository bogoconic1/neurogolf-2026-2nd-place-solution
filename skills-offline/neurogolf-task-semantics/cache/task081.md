# task081 Semantics

## Sources

- Current champion builder: `solutions_py/task081.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task081.json`
- ARC-GEN task id: `3aa6fb7a`
- ARC-DSL task id: `3aa6fb7a`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_3aa6fb7a.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_3aa6fb7a.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task081.py`

## Pattern

The input is a square grid, normally 7x7 in the ARC-GEN generator, with black background `0` and one or more cyan (`8`) L-shaped objects. Each object occupies three cells of a 2x2 block: exactly one corner of that 2x2 block is missing. The output keeps the input unchanged except that every missing corner of every cyan 2x2 L-shape is filled with blue (`1`). Cyan cells remain cyan, background outside these missing corners remains black.

There is no tie-breaking step. Every qualifying 2x2 cyan L-shape contributes one blue corner in the corresponding absent cell.

## Readable Python Solver

```python
def solve(grid):
    out = [row[:] for row in grid]
    h, w = len(grid), len(grid[0])
    for r in range(h - 1):
        for c in range(w - 1):
            cells = [
                grid[r][c],
                grid[r][c + 1],
                grid[r + 1][c],
                grid[r + 1][c + 1],
            ]
            if cells.count(8) == 3 and cells.count(0) == 1:
                k = cells.index(0)
                rr = r + (k // 2)
                cc = c + (k % 2)
                out[rr][cc] = 1
    return out
```

## Generator Constraints

ARC-GEN `generate()` builds a square grid with default `size=7`. It samples up to nine candidate 2x2 positions. For each candidate, it chooses one missing corner `0..3`; the other three cells become cyan in the input. Candidate cells that would be painted cyan are skipped when they already have a neighbor in the generator bitmap, so accepted cyan cells are separated enough that distinct objects do not merge into ambiguous larger components.

The output paints the missing corner of each accepted object blue and leaves the three cyan cells unchanged. The only colors used by the generator are black `0`, blue `1`, and cyan `8`.

## Reference Notes

ARC-DSL solves the task by finding foreground objects, taking their bounding-box corners, and `underfill`ing those corner locations with blue. Because each foreground object is a 2x2 box missing one corner, the only object corner still under background is exactly the missing cell.

The Code Golf 2025 solution uses a transpose/recursion trick and a regex over serialized rows to replace the black cell in a local cyan-corner pattern with blue. It confirms the same local rule: detect a 2x2 cyan L and color the absent corner blue.
