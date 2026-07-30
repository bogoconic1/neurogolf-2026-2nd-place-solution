# task059 Semantics

## Sources

- Current champion builder: `solutions_py/task059.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task059.json`
- ARC-GEN task id: `29623171`
- ARC-DSL task id: `29623171`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_29623171.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_29623171.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task059.py`

## Pattern

The input is an 11x11 grid composed of a 3x3 array of 3x3 mini-grids, separated by gray (`5`) grid lines at rows and columns `3` and `7`. The interior cells are black (`0`) except for sparse pixels of one non-gray color.

Count that non-gray, non-black color inside each 3x3 mini-grid. Let `m` be the maximum count over the nine mini-grids. The output preserves the gray separator lines, fills every mini-grid whose count equals `m` entirely with the colored pixel color, and clears every other mini-grid interior to black.

## Readable Python Solver

```python
def solve(grid):
    origins = (0, 4, 8)
    colors = [v for row in grid for v in row if v not in (0, 5)]
    color = colors[0] if colors else 0
    counts = {}
    for br in origins:
        for bc in origins:
            counts[(br, bc)] = sum(
                1
                for r in range(br, br + 3)
                for c in range(bc, bc + 3)
                if grid[r][c] == color
            )
    max_count = max(counts.values())
    out = [row[:] for row in grid]
    for br in origins:
        for bc in origins:
            fill = color if counts[(br, bc)] == max_count else 0
            for r in range(br, br + 3):
                for c in range(bc, bc + 3):
                    out[r][c] = fill
    return out
```

## Generator Constraints

The generator always uses `minisize=3`, so the grid is fixed at 11x11. The gray separator lines are fixed. It samples a per-mini-grid number of colored pixels between `0` and a random `max_pixels` in `2..5`, then samples that many distinct positions inside the 3x3 mini-grid. All colored pixels share one color chosen from the non-gray colors. Ties for maximum count are intentional: all tied mini-grids are filled in the output.

## Reference Notes

The ARC-DSL solver finds the least frequent color as the sparse pixel color, enumerates the nine origins `(0, 4, 8) x (0, 4, 8)`, counts that color in each 3x3 object, fills maximum-count mini-grids with the color, and fills the others with black. The Code Golf solution recursively lowers a threshold until at least one mini-grid exceeds it, then emits a filled 3x3 block for every maximum mini-grid. The references agree on fixed geometry and tie handling.
