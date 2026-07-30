# task316 Semantics

## Sources

- Current champion builder: `solutions_py/task316.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task316.json`
- ARC-GEN task id: `cdecee7f`
- ARC-DSL task id: `cdecee7f`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_cdecee7f.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_cdecee7f.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task316.py`

## Pattern

The input is a 10x10 grid containing 6 to 9 isolated nonzero colored pixels. Their columns are unique and sorted left-to-right by the generator; row positions are irrelevant except for locating the color in each column. The output is a 3x3 grid containing the nonzero colors in left-to-right column order, padded with zeros to 9 entries. The middle output row is reversed, producing a snake order: positions 0,1,2 go left-to-right in row 0; positions 3,4,5 appear right-to-left in row 1; positions 6,7,8 go left-to-right in row 2.

## Readable Python Solver

```python
def solve(grid):
    colors = []
    for c in range(10):
        col_color = 0
        for r in range(10):
            if grid[r][c] != 0:
                col_color = grid[r][c]
                break
        if col_color:
            colors.append(col_color)
    colors = colors + [0] * (9 - len(colors))
    return [
        colors[0:3],
        colors[3:6][::-1],
        colors[6:9],
    ]
```

## Generator Constraints

ARC-GEN fixes input size `10` and output size `3`. It samples between 6 and 9 colors from the non-background palette, assigns each a random row, and samples the same number of unique columns from `0..9`, then sorts columns. There is at most one colored pixel per column and no column contains two nonzero colors. The colors may repeat. Missing output slots after the final marker are zero.

## Reference Notes

The ARC-DSL solver extracts objects, orders them by `leftmost`, converts their colors to 1x1 canvases, pads to nine positions, splits into three rows, and vertically mirrors the middle row before concatenating. The Code Golf solution uses column maxima (`map(max, *g)`) to recover each occupied column color, filters zeros, pads with zeros, then emits `[S[:3], S[5:2:-1], S[6:9]]`, confirming the snake order and the irrelevance of row coordinates.
