# task310 Semantics

## Sources

- Current champion builder: `solutions_py/task310.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task310.json`
- ARC-GEN task id: `c909285e`
- ARC-DSL task id: `c909285e`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_c909285e.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_c909285e.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task310.py`

## Pattern

The input is a `20..30` square grid containing several periodic horizontal and
vertical wire colors plus one hollow square border in a distinct color. The box
color is the least-present nonzero color. The output is the tight square crop
inside the box border: the hollow border plus any wire segments crossing the
box, padded to the standard NeuroGolf canvas.

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])
    counts = {}
    for r in range(h):
        for c in range(w):
            x = grid[r][c]
            if x:
                counts[x] = counts.get(x, 0) + 1
    color = min(counts, key=counts.get)
    rows = [r for r in range(h) for c in range(w) if grid[r][c] == color]
    cols = [c for r in range(h) for c in range(w) if grid[r][c] == color]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    return [row[c0:c1 + 1] for row in grid[r0:r1 + 1]]
```

## Generator Constraints

- Grid size is `20..30`.
- There are 3 or 4 wire colors with spacings sampled from `[3,4,5,6]`.
- A distinct box color is chosen from the color list and removed from the wire
  colors.
- Box side length is `5..8`.
- Box origin is any position where the square fits in the grid.
- The box border overwrites wires in the input; the output crop contains the
  border and any wire colors inside the box.
- The box color count is `4 * L - 4`, and it is the least-present nonzero color
  under the generator.

## Reference Notes

ARC-GEN gives the exact wire-plus-hollow-square construction. ARC-DSL identifies the least color and returns the subgrid spanned by that color. The Code Golf solution is a compact crop-by-border implementation. No disagreement was found.
