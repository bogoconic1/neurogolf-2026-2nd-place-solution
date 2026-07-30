# task127 Semantics

## Sources

- Current champion builder: `solutions_py/task127.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task127.json`
- ARC-GEN task id: `54d9e175`
- ARC-DSL task id: `54d9e175`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_54d9e175.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_54d9e175.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task127.py`

## Pattern

The input is a fixed 11-column marker grid with either one band (`3 x 11`) or two bands (`7 x 11`). Gray (`5`) separator columns appear at columns 3 and 7 in every row, and the two-band case also has a full gray separator row at row 3. Each band has three marker cells at row `4*b + 1` and columns `1`, `5`, and `9`. Marker colors are in `1..4`.

The output expands every marker into its surrounding `3 x 3` block and shifts marker colors by `+5`: `1->6`, `2->7`, `3->8`, `4->9`. Gray separators stay gray. The rest of the standard canvas is zero/black.

## Readable Python Solver

```python
def solve(grid):
    H, W = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    for r in range(H):
        for c in range(W):
            color = grid[r][c]
            if color in (1, 2, 3, 4):
                for rr in range(r - 1, r + 2):
                    for cc in range(c - 1, c + 2):
                        if 0 <= rr < H and 0 <= cc < W:
                            out[rr][cc] = color + 5
    return out
```

## Generator Constraints

ARC-GEN samples either three marker colors (one band, height `3`) or six marker colors (two bands, height `7`). Width is always `11`. Gray separators are deterministic: columns `3` and `7` in all rows, plus row `3` in the two-band case. Marker colors are sampled independently from `1..4`, so repeats are allowed. Only marker colors `1..4` are expanded; gray remains gray.

## Reference Notes

The ARC-DSL solver finds singleton marker objects, paints each marker's neighboring cells with the same marker color, then applies the color replacements `1->6`, `2->7`, `3->8`, and `4->9`. The Code Golf recursive solution encodes the same band expansion: a marker row becomes three identical expanded rows, separator rows are preserved, and marker colors are shifted by five.
