# task345 Semantics

## Sources

- Current champion builder: `solutions_py/task345.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task345.json`
- ARC-GEN task id: `d9f24cd1`
- ARC-DSL task id: `d9f24cd1`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_d9f24cd1.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_d9f24cd1.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task345.py`

## Pattern

The grid is always `10x10` with background `0`, red `2`, and gray `5`. The input contains gray obstacle dots in rows `2..6` and red start pixels on the bottom row. The output preserves the input and draws red paths upward from each bottom red start. A path normally moves straight up in its current column. If the cell immediately above the current red head is gray, the path first steps one cell to the right on the same row, then resumes moving upward from that new column. The gray obstacle remains gray; the red path routes around it on the right.

The final output is the full `10x10` grid, later represented in NeuroGolf as the dense `[1,10,30,30]` tensor with the visible result in the upper-left corner.

## Readable Python Solver

```python
def solve(grid):
    out = [row[:] for row in grid]
    n = len(grid)

    starts = [c for c, value in enumerate(grid[n - 1]) if value == 2]
    for start_col in starts:
        r, c = n - 1, start_col
        out[r][c] = 2
        while r > 0:
            if out[r - 1][c] == 5:
                c += 1
            else:
                r -= 1
            out[r][c] = 2
    return out
```

## Generator Constraints

- Size is fixed at `10x10` for all generated, train, and test examples.
- Colors are fixed: background `0`, red `2`, gray `5`.
- Red starts are placed on the bottom row. Candidate start columns begin at `1`, then advance by either `2`, `3`, or `4`; therefore starts are separated and never at column `0`.
- Optional gray dots are placed for some starts at rows sampled from `2..6` and columns `start-1`, `start`, or `start+1`.
- The generator rerolls until at least one red path encounters a gray cell above it and must step right. Not every start necessarily has an obstacle.
- The path may step right, but generator spacing keeps paths and obstacles within the `10x10` grid.
- The output keeps gray dots and bottom red starts, then adds the routed red paths.

## Reference Notes

- ARC-GEN is the clearest description: `draw` starts from each red bottom pixel, repeatedly checks the cell above, moves right if that cell is gray, otherwise moves up, and writes red at the new position.
- ARC-DSL expresses the same behavior through red/gray connections, vertical-line filters, underfilling red, identifying two-color objects, shooting upward from shifted upper-right corners, and filling vertical frontiers.
- The Code Golf 2025 solution uses a regex over the stringified grid to repeatedly insert red marks above red pixels and to the right of gray-blocked red pixels. It confirms the transformation is an iterative upward propagation/routing rule rather than a one-shot static recolor.
