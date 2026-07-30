# task395 Semantics

## Sources

- Current champion builder: `solutions_py/task395.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task395.json`
- ARC-GEN task id: `fafffa47`
- ARC-DSL task id: `fafffa47`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_fafffa47.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_fafffa47.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task395.py`

## Pattern

The input is a `6x3` logical grid made from two stacked `3x3` panels. The top panel uses maroon cells on black; the bottom panel uses blue cells on black. The output is a `3x3` grid. For each position, output red (`2`) exactly when the corresponding cell is black in both the top and bottom panels. Otherwise output black (`0`). In NeuroGolf the dense output is the standard `[1, 10, 30, 30]` tensor with the `3x3` answer at the origin.

## Readable Python Solver

```python
def solve(grid):
    top = grid[:3]
    bottom = grid[3:6]
    out = [[0 for _ in range(3)] for _ in range(3)]
    for r in range(3):
        for c in range(3):
            if top[r][c] == 0 and bottom[r][c] == 0:
                out[r][c] = 2
    return out
```

## Generator Constraints

ARC-GEN uses fixed `size=3`, so the input is always height `6` and width `3`, and the output is always `3x3`. It samples random colored pixels independently for the top and bottom panels until the union of sampled row indices and column indices covers all three rows and all three columns. Top-panel nonzero cells are maroon; bottom-panel nonzero cells are blue. Black cells are the only cells that matter for the output gate. There are no variable colors, variable sizes, rotations, or tie-breaks.

## Reference Notes

ARC-DSL splits the input with `tophalf` and `bottomhalf`, takes black-cell coordinates from both halves, intersects those coordinate sets, and fills a black canvas with red at the intersection. The Code Golf 2025 solution is a terse recursive/numpy expression but matches the same pairwise black-cell intersection rule. There is no disagreement between references.
