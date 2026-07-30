# task351 Semantics

## Sources

- Current champion builder: `solutions_py/task351.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task351.json`
- ARC-GEN task id: `dc0a314f`
- ARC-DSL task id: `dc0a314f`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_dc0a314f.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_dc0a314f.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task351.py`

## Pattern

The input is a `16x16` four-way mirrored color pattern with one missing `5x5`
square. The missing square is marked in green (`3`). The output is exactly the
hidden `5x5` patch that belongs under that green block. Because the source
pattern is mirrored across both horizontal and vertical center axes (and the
base quadrant is itself diagonally symmetric), the missing patch can be
recovered by copying the corresponding mirrored patch from the opposite side of
the input after ignoring the green marker cells.

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])

    rows = [r for r, row in enumerate(grid) if 3 in row]
    r0, r1 = min(rows), max(rows)
    cols = [c for c in range(w) if any(grid[r][c] == 3 for r in range(h))]
    c0, c1 = min(cols), max(cols)

    # Reconstruct the full symmetric canvas by replacing the green hole with
    # zero, then taking the maximum over horizontal/vertical mirror partners.
    full = [[0 if grid[r][c] == 3 else grid[r][c] for c in range(w)] for r in range(h)]
    for r in range(h):
        for c in range(w):
            vals = [
                full[r][c],
                full[h - 1 - r][c],
                full[r][w - 1 - c],
                full[h - 1 - r][w - 1 - c],
            ]
            full[r][c] = max(vals)

    return [row[c0:c1 + 1] for row in full[r0:r1 + 1]]
```

## Generator Constraints

ARC-GEN builds an `8x8` base bitmap from non-green colors, with diagonal
symmetry, then mirrors it into all four quadrants of a `16x16` grid. A `5x5`
cutout is copied to the output and the same area in the input is replaced with
green (`3`). Random generation chooses row and column in `0..3`, while the
fixed validation examples also place the green block in other quadrants and
near center-crossing positions. Colors are random non-green digits; green is
reserved only for the missing block marker.

## Reference Notes

The ARC-DSL solver finds all green cells, replaces green with black/zero, then combines the image with its vertical and horizontal mirrors using `maximum`. It finally extracts the subgrid at the green-cell bounding box. The Code Golf solution uses the same idea in a terse row/column reversal form: find rows with green, take the mirror partner row, and crop around the mirrored green-column position. The references agree that the output is the missing patch, not the whole repaired grid.
