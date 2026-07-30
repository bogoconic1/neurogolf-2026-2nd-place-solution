# task085 Semantics

## Sources

- Current champion builder: `solutions_py/task085.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task085.json`
- ARC-GEN task id: `3bdb4ada`
- ARC-DSL task id: `3bdb4ada`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_3bdb4ada.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_3bdb4ada.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task085.py`

## Pattern

The input contains one or more colored 3-row horizontal rectangles, called punchcards here, on a black background. Each rectangle has odd width. The output keeps the top and bottom rows of every rectangle unchanged, and in the middle row keeps only every other cell starting from the rectangle left edge. The middle-row cells at odd offsets from the rectangle left edge are turned black.

There is no tie-breaking: every 3-row colored rectangle is processed independently using its own left edge and width.

## Readable Python Solver

```python
def solve(grid):
    out = [row[:] for row in grid]
    h, w = len(grid), len(grid[0])
    seen = [[False] * w for _ in range(h)]
    for r in range(h):
        for c in range(w):
            color = grid[r][c]
            if color == 0 or seen[r][c]:
                continue
            # Find the monochrome connected rectangle containing this cell.
            cc = c
            while cc < w and grid[r][cc] == color:
                cc += 1
            width = cc - c
            # The generator makes exact 3-row rectangles, so r is their top row.
            for rr in range(r, min(r + 3, h)):
                for x in range(c, c + width):
                    if 0 <= x < w and grid[rr][x] == color:
                        seen[rr][x] = True
                        if rr == r + 1 and (x - c) % 2 == 1:
                            out[rr][x] = 0
            
    return out
```

## Generator Constraints

ARC-GEN chooses width `20` or `30` and height from `8..16`. It places punchcards from top to bottom, starting at row 1 and then advancing by 3 or 4 rows while room remains. Each punchcard is exactly 3 rows tall, has an odd width `2*k-1`, can start at any column that keeps it in bounds, and receives a random non-background color. Different punchcards are vertically separated enough to remain distinct.

For each punchcard, the input fills all cells in its 3xW rectangle with the card color. The output fills the top and bottom rows completely and fills only even offsets in the middle row; odd offsets in the middle row are black.

## Reference Notes

ARC-DSL finds foreground objects, connects the inner upper-left and lower-right corners of each object to identify the middle row segment, then filters cells by parity relative to the object left edge and fills the rejected middle-row cells black.

The Code Golf 2025 solution recursively transforms rows: repeated identical rows identify a 3-row card, and the middle row is replaced with a parity-punched version while the top/bottom rows are preserved.
