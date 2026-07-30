# task174 Semantics

## Sources

- Current champion builder: `solutions_py/task174.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task174.json`
- ARC-GEN task id: `72ca375d`
- ARC-DSL task id: `72ca375d`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_72ca375d.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_72ca375d.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task174.py`

## Pattern

The input is a 10x10 grid containing three non-overlapping same-color objects. Exactly one object is vertically mirror-symmetric within its own bounding box. The output is the cropped subgrid of that symmetric object only, preserving its color and shape.

## Readable Python Solver

The previous cache implementation did not survive replay. The compact executable oracle below is the bundled Code Golf 2025 reference; the Pattern section remains the readable specification.

```python
p = lambda g, i=7: [r * (r == r[::1 | i % -2]) for *r, in zip(*(g[70:] or p(g * 2, i - 1))) if i // 4 in r] or p(g, i + 4)


def solve(grid):
    result = p([list(row) for row in grid])
    return [list(row) for row in result]
```

## Generator Constraints

The grid size is fixed at 10x10 with exactly three placed objects. Each object has a unique nonzero color. Widths are 2..5 and heights are 2..(7-width), so cropped outputs fit within 5x5. Objects are separated by at least one cell. Object 0 is both vertically mirror-symmetric and rotationally symmetric; the other two are not vertically mirror-symmetric and not rotationally symmetric. The output is object 0 cropped to its bounding box.

## Reference Notes

The ARC-DSL solver extracts objects, computes each object subgrid, vertical-mirrors each subgrid, compares equality, and returns the one equal to its vertical mirror. The Code Golf solution performs a compact scan with symmetry checks. The sources agree that the selected object is the vertically symmetric one.
