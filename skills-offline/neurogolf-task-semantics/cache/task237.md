# task237 Semantics

## Sources

- Current champion builder: `solutions_py/task237.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task237.json`
- ARC-GEN task id: `99fa7670`
- ARC-DSL task id: `99fa7670`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_99fa7670.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_99fa7670.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task237.py`

## Pattern

The input is a small black grid with one colored seed in each of several separated rows. For every seed at `(r, c)` with color `k`, the output draws a horizontal ray of color `k` from `(r, c)` through the right edge. The rightmost column also carries color `k` downward from row `r` until the next lower seed begins its own color run. Rows before the first seed stay black, and non-seed rows have only the rightmost carried color.

Equivalently, scan rows from top to bottom while remembering the most recent seed color. Within a row, if a seed appears, fill from that seed column to the right with its color and update the remembered color. If no seed appears, only the final column receives the remembered color.

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])
    out = [[0 for _ in range(w)] for _ in range(h)]
    carry = 0
    for r, row in enumerate(grid):
        seed_col = None
        seed_color = 0
        for c, x in enumerate(row[:-1]):  # generator never places a seed in the last column
            if x != 0:
                seed_col = c
                seed_color = x
                break
        if seed_col is not None:
            carry = seed_color
            for c in range(seed_col, w):
                out[r][c] = seed_color
        elif carry:
            out[r][w - 1] = carry
    return out
```

## Generator Constraints

- Input width and height are each 3 to 9.
- Seed rows start at row 0 or 1, then advance by 2 or 3, so seeds are vertically separated and no two seeds are adjacent.
- The loop condition is `r + 1 < height`, so every seed row is at most `height - 2`; the last row never contains a new seed but may contain the carried right-column color in the output.
- Each seed column is sampled from `0..width-2`, so no seed starts in the rightmost column.
- `common.random_colors(len(rows))` supplies distinct nonzero colors for the seeds.
- There are no distractor colors or objects; all nonzero input cells are seeds.

## Reference Notes

- ARC-DSL first paints rightward rays from each seed, then orders the resulting objects by row and underpaints vertical connectors between consecutive right-edge endpoints. A synthetic black object below the grid closes the final carried segment.
- The Code Golf 2025 one-liner implements the same row scan: carry the last seen seed color, fill right within seed rows, and emit the carried color in the row's final column.
- There is no ambiguous tie-breaking because seed rows are separated and each seed row contains exactly one nonzero cell.
