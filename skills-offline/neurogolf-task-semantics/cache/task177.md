# task177 Semantics

## Sources

- Current champion builder: `solutions_py/task177.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task177.json`
- ARC-GEN task id: `7468f01a`
- ARC-DSL task id: `7468f01a`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_7468f01a.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_7468f01a.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task177.py`

## Pattern

The input is a black grid containing one nonzero rectangular object. The rectangle is filled with one foreground color, and a small connected/creature-like subset of cells inside the rectangle is recolored to a second foreground color. The output is the cropped rectangle, mirrored left-to-right.

## Readable Python Solver

```python
def solve(grid):
    cells = [(r, c) for r, row in enumerate(grid) for c, v in enumerate(row) if v != 0]
    r0 = min(r for r, _ in cells)
    r1 = max(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    c1 = max(c for _, c in cells)
    crop = [row[c0:c1 + 1] for row in grid[r0:r1 + 1]]
    return [row[::-1] for row in crop]
```

## Generator Constraints

Input width and height are random 10..20. The output rectangle width and height are random 4..8, placed strictly inside the input with at least a one-cell black border. The rectangle's base color and marker color are two distinct nonzero colors. The marker cells are generated from 1..3 small connected creatures of size 1..5, placed inside the rectangle. Fixed validation examples include rectangle sizes from 5x5 through 8x7.

## Reference Notes

The ARC-DSL solver finds foreground objects, takes the first object, extracts its subgrid from the input, and applies `vmirror`. The Code Golf solution recursively strips empty rows/columns and transposes/reverses to mirror the nonzero bounding box.
