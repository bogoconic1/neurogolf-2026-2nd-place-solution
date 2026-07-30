# task136 Semantics

## Sources

- Current champion builder: `solutions_py/task136.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task136.json`
- ARC-GEN task id: `5c0a986e`
- ARC-DSL task id: `5c0a986e`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_5c0a986e.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_5c0a986e.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task136.py`

## Pattern

The input is a 10x10 grid containing two solid 2x2 squares on a black background. Color 1 is red and color 2 is blue. The output keeps both 2x2 squares and extends a red diagonal ray up-left from the red square's upper-left cell, and a blue diagonal ray down-right from the blue square's lower-right cell. All other cells stay black.

Equivalently, if the red square has upper-left `(rr, rc)` and the blue square has upper-left `(br, bc)`, then output starts as the input, sets every `(rr-k, rc-k)` in bounds to 1 for `k >= 0`, and sets every `(br+1+k, bc+1+k)` in bounds to 2 for `k >= 0`.

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])
    out = [row[:] for row in grid]

    red = [(r, c) for r in range(h) for c in range(w) if grid[r][c] == 1]
    blue = [(r, c) for r in range(h) for c in range(w) if grid[r][c] == 2]

    rr = min(r for r, c in red)
    rc = min(c for r, c in red)
    br = max(r for r, c in blue)
    bc = max(c for r, c in blue)

    r, c = rr, rc
    while r >= 0 and c >= 0:
        out[r][c] = 1
        r -= 1
        c -= 1

    r, c = br, bc
    while r < h and c < w:
        out[r][c] = 2
        r += 1
        c += 1

    return out
```

## Generator Constraints

ARC-GEN fixes the canvas size at 10x10. The red square upper-left row is chosen from `1..8` and column from `1..8`, so its upper-left is never on the top or left border but the 2x2 square can touch the bottom or right border. The blue square upper-left row is chosen from `0..7` and column from `0..7`, so its lower-right is never outside the grid but its upper-left can touch the top or left border.

The generator rejects placements where the two 2x2 boxes overlap or are too close in both row and column: `abs(rows[0] - rows[1]) >= 3` or `abs(cols[0] - cols[1]) >= 3` must hold. It also rejects cases where the red upper-left diagonal and blue upper-left diagonal are closer than three cells in diagonal offset: `abs((rows[0] - cols[0]) - (rows[1] - cols[1])) >= 3`. These constraints separate the two diagonal rays enough that simple per-color anchor scoring is reliable.

Colors are fixed: red is 1 and blue is 2. There are exactly four input pixels of each color, arranged as 2x2 blocks.

## Reference Notes

ARC-DSL computes the lower-right corner of the blue object and shoots along `UNITY` `(1, 1)`, then computes the upper-left corner of the red object and shoots along `NEG_UNITY` `(-1, -1)`. It fills blue first and red second, but the generator's diagonal separation means the rays do not create ambiguous overlap in normal cases.

The Code Golf 2025 solution uses repeated regex substitutions on the reversed string representation, alternating colors `2` and `1`; it is a compact trick for extending color-specific diagonal patterns, not a clearer algorithm than the DSL. It agrees with the two diagonal-ray interpretation.
