# task321 Semantics

## Sources

- Current champion builder: `solutions_py/task321.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task321.json`
- ARC-GEN task id: `cf98881b`
- ARC-DSL task id: `cf98881b`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_cf98881b.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_cf98881b.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task321.py`

## Pattern

The input is a 4x14 grid containing three 4x4 panels separated by red (`2`)
vertical separator columns. The left panel contains color `4`, the middle panel
contains color `9`, and the right panel contains color `1`, each on background
0. The output is a 4x4 overlay of the three panels in their shared local
coordinates. Color `4` has highest priority, color `9` has second priority, and
color `1` has lowest priority; background remains 0 where none of the panels has
a colored cell.

## Readable Python Solver

```python
def solve(grid):
    out = [row[10:14] for row in grid[:4]]  # start from the right/color-1 panel
    for r in range(4):
        for c in range(4):
            if grid[r][5 + c] == 9:
                out[r][c] = 9
            if grid[r][c] == 4:
                out[r][c] = 4
    return out
```

## Generator Constraints

ARC-GEN uses `size=4` and colors `(4, 9, 1)`. It creates a 4x14 input and a 4x4
output. Red separator cells are placed at columns 4 and 9 in every row. For each
of the three color layers, `common.random_pixels(4,4)` chooses an arbitrary set
of local panel coordinates. The same local coordinate may appear in multiple
layers; priority is resolved by applying layer 1, then 9, then 4, so color 4
wins ties, then 9, then 1. No dimensions or colors vary in the bundled
generator.

## Reference Notes

The ARC-DSL solver splits the input into three panels, starts from the last panel, fills positions from the middle panel with 9, then fills positions from the first panel with 4. The Code Golf solution destructively pops row segments and ORs the layer masks in priority order, confirming that the red separators are only layout delimiters and do not affect the output.
