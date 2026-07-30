# task122 Semantics

## Sources

- Current champion builder: `solutions_py/task122.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task122.json`
- ARC-GEN task id: `5168d44c`
- ARC-DSL task id: `5168d44c`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_5168d44c.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_5168d44c.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task122.py`

## Pattern

The input has a horizontal or vertical green guide line with every other cell green (`3`). A 3x3 red block (`2`) is centered on one of those guide positions. The output moves the entire red 3x3 block one guide step forward, i.e. two cells along the guide direction, while the green guide remains fixed.

If the guide is horizontal, the red block moves two columns to the right. If the guide is vertical, the red block moves two rows down. The transformation preserves the input grid size, clears the old red block cells back to background except where the green guide remains, and paints the red block at the shifted location.

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])
    out = [row[:] for row in grid]

    greens = [(r, c) for r in range(h) for c in range(w) if grid[r][c] == 3]
    horizontal = len({r for r, _ in greens}) == 1
    dr, dc = (0, 2) if horizontal else (2, 0)

    red_cells = [(r, c) for r in range(h) for c in range(w) if grid[r][c] == 2]
    for r, c in red_cells:
        out[r][c] = 0
    for r, c in greens:
        out[r][c] = 3
    for r, c in red_cells:
        out[r + dr][c + dc] = 2
    return out
```

## Generator Constraints

Generated widths are in `[7, 21]`; height is in `[7, width]`. The red block center row is sampled from `1..height-2`, so the 3x3 red block fits vertically. The guide start column is `0` or `1`, and green cells occupy every other coordinate along the line. The selected red block index is chosen so that shifting the red block one guide step forward by two cells still fits. A transpose branch swaps horizontal and vertical cases.

The validate examples include both orientations and small widths. The output grid has the same dimensions as the input ARC grid; NeuroGolf still uses a `[1,10,30,30]` one-hot tensor with active content in the top-left grid area.

## Reference Notes

The ARC-DSL solver finds green cells, uses the height of the green object to detect orientation, chooses offset `(0,2)` or `(2,0)`, recolors red cells, and moves them by that offset. This confirms the task is only a red-object translation controlled by green-line orientation.

The Code Golf solution recursively transposes until it sees the horizontal case, then shifts the red pattern by two positions while preserving the rest of the grid.
