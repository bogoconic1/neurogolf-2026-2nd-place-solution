# task031 Semantics

## Sources

- Current champion builder: `solutions_py/task031.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task031.json`
- ARC-GEN task id: `1cf80156`
- ARC-DSL task id: `1cf80156`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_1cf80156.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_1cf80156.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task031.py`

## Pattern

The input is a black rectangular grid of width 12 and height 10, 11, or 12 containing exactly one connected foreground object in a single nonzero color. The output is the tight bounding-box crop of that object: remove every all-black row above and below it and every all-black column to its left and right, preserving the object's color and internal black holes/gaps inside the bounding box.

## Readable Python Solver

```python
def solve(grid):
    rows = []
    cols = []
    for r, row in enumerate(grid):
        for c, value in enumerate(row):
            if value != 0:
                rows.append(r)
                cols.append(c)
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    return [row[c0:c1 + 1] for row in grid[r0:r1 + 1]]
```

## Generator Constraints

ARC-GEN fixes `width=12` and chooses `height` uniformly in `10..12`. It seeds a random-color object near the middle third of the grid, then grows a 4-connected tree-like object by random queue expansion. The object has 8 to 12 pixels when generation reaches the target count, avoids the outer border, and rejects candidate cells that would have more than one already-selected 4-neighbor. This produces a connected, mostly non-branch-overlapping foreground object with possible holes/gaps inside its bounding box. The output width and height are data-dependent bounding-box dimensions, generally much smaller than 12 but not fixed.

## Reference Notes

ARC-DSL simply finds the foreground object and returns `subgrid(object, input)`, confirming that the task is pure bounding-box cropping with no recoloring or shape transformation. The Code Golf solution recursively filters all-empty rows and columns by transposing, which is equivalent to trimming black borders until the tight object crop remains. There is no tie-breaking ambiguity because there is exactly one foreground object and one nonzero color.
