# task144 Semantics

## Sources

- Current champion builder: `solutions_py/task144.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task144.json`
- ARC-GEN task id: `6430c8c4`
- ARC-DSL task id: `6430c8c4`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_6430c8c4.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_6430c8c4.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task144.py`

## Pattern

The input is a 9x4 grid made from a top 4x4 orange/empty panel, a yellow separator row, and a bottom 4x4 red/empty panel. The output is a 4x4 grid. A cell is green exactly when the corresponding top-panel cell and bottom-panel cell are both empty/background; otherwise the output cell is background 0.

## Readable Python Solver

The previous cache implementation did not survive replay. The compact executable oracle below is the bundled Code Golf 2025 reference; the Pattern section remains the readable specification.

```python
p = lambda g, u=[]: g * 0 != 0 and [*map(p, g, u + g[5:])] or 3 >> g + u


def solve(grid):
    result = p([list(row) for row in grid])
    return [list(row) for row in result]
```

## Generator Constraints

The input dimensions are fixed. The top and bottom panels are both 4x4. Row 4 is a yellow separator in columns 0..3. Top-panel marked cells are orange and bottom-panel marked cells are red. The two random pixel sets are independent. Output size before NeuroGolf padding is fixed at 4x4.

## Reference Notes

ARC-DSL takes the top half and bottom half, finds background cells in each, intersects those coordinate sets, and fills a 4x4 zero canvas with green at the intersection. Code Golf expresses the same pairwise empty-cell test recursively.
