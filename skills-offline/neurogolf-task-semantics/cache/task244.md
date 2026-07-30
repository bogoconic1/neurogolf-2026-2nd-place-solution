# task244 Semantics

## Sources

- Current champion builder: `solutions_py/task244.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task244.json`
- ARC-GEN task id: `9f236235`
- ARC-DSL task id: `9f236235`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_9f236235.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_9f236235.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task244.py`

## Pattern

The input is a horizontally mirrored magnified line-grid representation of a small square color grid. Separator rows and columns are drawn in one line color, and each original cell is expanded to a solid `magnifier x magnifier` block between separator lines. The output is the original unmirrored small grid: remove the line grid, undo the horizontal mirror, and downsample each solid block to one cell color.

## Readable Python Solver

The previous cache implementation did not survive replay. The compact executable oracle below is the bundled Code Golf 2025 reference; the Pattern section remains the readable specification.

```python
p = lambda g, w=2: [[0, max, p][w]((g := r), -2) for r in g if g != r][::w]


def solve(grid):
    result = p([list(row) for row in grid])
    return [list(row) for row in result]
```

## Generator Constraints

ARC-GEN chooses output size `3` or `4` and magnifier `2..5`. It chooses a separator line color, then one to three cell colors excluding that separator color. A random nonempty set of small-grid cells is colored; all other cells are background `0`, unless one of the chosen colors also fills a selected cell. The magnified input is produced with `create_linegrid(output, magnifier, linecolor)` and then flipped horizontally. The separator color is guaranteed not to appear inside the cell blocks, which makes full separator rows/columns reliable. The output is the unmagnified grid before the horizontal flip.

## Reference Notes

The ARC-DSL solver applies `compress(I)` to remove separator lines, horizontally mirrors the compressed grid, then downsamples by the minimum object width. This confirms the order: remove gridlines, undo the horizontal mirror, then shrink solid blocks. The Code Golf solution filters separator rows/columns and samples with an inferred stride. There is no tie-breaking between multiple candidate objects; the key ambiguity is only the pitch/size classification for dimensions that can arise from both `3*m` and `4*m`, and the separator spacing resolves it.
