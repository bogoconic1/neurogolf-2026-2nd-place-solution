# task091 Semantics

## Sources

- Current champion builder: `solutions_py/task091.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task091.json`
- ARC-GEN task id: `3f7978a0`
- ARC-DSL task id: `3f7978a0`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_3f7978a0.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_3f7978a0.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task091.py`

## Pattern

The input is a black grid with scattered cyan `8` pixels and a marked rectangular
window. The window has cyan `8` corner tips and gray `5` vertical side segments
between the tips on the left and right edges. The output is exactly the crop of
the input bounded by those markers: from the top cyan-corner row through the
bottom cyan-corner row, and from the left gray/cyan side through the right
gray/cyan side. Cyan random pixels inside that rectangle are preserved; pixels
outside the rectangle are discarded.

## Readable Python Solver

```python
def solve(grid):
    h = len(grid)
    w = len(grid[0])

    gray = [(r, c) for r in range(h) for c in range(w) if grid[r][c] == 5]
    r0 = min(r for r, c in gray) - 1
    r1 = max(r for r, c in gray) + 1
    c0 = min(c for r, c in gray)
    c1 = max(c for r, c in gray)

    return [row[c0:c1 + 1] for row in grid[r0:r1 + 1]]
```

## Generator Constraints

ARC-GEN chooses input width `9..15` and height `9..15`. It scatters cyan `8`
pixels at roughly 10 percent of cells, then chooses a crop width `3..width-1`,
crop height `3..height-1`, and a top-left crop origin that keeps the crop inside
the grid. The four crop corners are forced to cyan. The left and right crop
edges between the corners are forced to gray `5`, so there is always at least
one gray cell on each side because crop height is at least 3. The output size is
`zoom_height x zoom_width`, not the standard ARC input size.

## Reference Notes

The ARC-DSL solver extracts the gray object, takes its upper-left corner, moves one row up to include the top cyan tips, adds two rows to the gray bounding-box height, and crops the original input. This confirms that the gray side markers, not random cyan pixels, define the crop. The Code Golf solution repeatedly rotates/transposes until it finds the gray marker layout and returns the crop. All references agree that the output is a crop preserving interior pixels.
