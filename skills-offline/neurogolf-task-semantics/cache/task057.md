# task057 Semantics

## Sources

- Current champion builder: `solutions_py/task057.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task057.json`
- ARC-GEN task id: `28bf18c6`
- ARC-DSL task id: `28bf18c6`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_28bf18c6.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_28bf18c6.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task057.py`

## Pattern

The input is an 8x8 grid with background `0` and one nonzero color. The nonzero cells form a diagonally connected Conway-style 3x3 sprite placed at a variable row and column. The output is the sprite's 3x3 bounding box duplicated horizontally, producing a 3x6 grid. The sprite color is preserved and background remains `0`.

## Readable Python Solver

```python
def solve(grid):
    cells = [(r, c) for r, row in enumerate(grid) for c, v in enumerate(row) if v != 0]
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    crop = [row[c0:c0 + 3] for row in grid[r0:r0 + 3]]
    return [row + row for row in crop]
```

## Generator Constraints

ARC-GEN uses fixed input size `8x8` and fixed sprite size `3x3`. It samples a diagonally connected sprite pattern, places it with `row` and `col` in `0..4`, and chooses a random nonzero color. Validation examples include 4 to 6 colored cells and placements near different sides. The output size is always `3x6` before NeuroGolf padding to `[1,10,30,30]`.

## Reference Notes

The ARC-DSL solver finds the object, takes its `subgrid`, and horizontally concatenates the subgrid with itself. The Code Golf solution recursively filters nonempty columns/rows and doubles the cropped sprite. The references agree that only the object's 3x3 bounding crop matters; no color remapping or rotation is involved.
