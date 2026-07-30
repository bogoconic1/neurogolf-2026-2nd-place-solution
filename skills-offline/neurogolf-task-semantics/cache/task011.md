# task011 Semantics

## Sources

- Current champion builder: `solutions_py/task011.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task011.json`
- ARC-GEN task id: `09629e4f`
- ARC-DSL task id: `09629e4f`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_09629e4f.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_09629e4f.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task011.py`

## Pattern

The input is an 11x11 Hollywood-squares grid: a 3x3 array of 3x3 mini-grids separated by gray (`5`) rows and columns at indices 3 and 7. Each mini-grid contains black (`0`) background and colored pixels from `{2, 3, 4, 6, 8}`. Exactly one mini-grid is the target because it has only four colored pixels and therefore only four non-black/non-gray colors; every other mini-grid has five colored pixels.

Copy the target mini-grid 3x3 color pattern to the whole output layout. For each target cell `(r, c)`, fill the corresponding 3x3 output block with that cell color; if the target cell is black, the output block is black. The gray separator rows and columns remain gray. The output size stays 11x11.

## Readable Python Solver

```python
def solve(grid):
    n = 3
    stride = 4

    best_block = None
    best_count = None
    for br in range(n):
        for bc in range(n):
            r0 = br * stride
            c0 = bc * stride
            block = [row[c0:c0 + n] for row in grid[r0:r0 + n]]
            count = sum(1 for row in block for v in row if v not in (0, 5))
            if best_count is None or count < best_count:
                best_count = count
                best_block = block

    out = [[0 for _ in range(11)] for _ in range(11)]
    for i in (3, 7):
        for j in range(11):
            out[i][j] = 5
            out[j][i] = 5

    for sr in range(n):
        for sc in range(n):
            color = best_block[sr][sc]
            for dr in range(n):
                for dc in range(n):
                    out[sr * stride + dr][sc * stride + dc] = color
    return out
```

## Generator Constraints

- Dimensions are fixed at 11x11, made from nine 3x3 mini-grids separated by gray `5` grid lines. Mini-grid top-left corners are `(0,0)`, `(0,4)`, `(0,8)`, `(4,0)`, ..., `(8,8)`.
- One mini-grid is chosen uniformly as the target. It receives exactly four colored cells using the first four rainbow colors `[2, 3, 4, 6]`.
- The other eight mini-grids receive exactly five colored cells using `[2, 3, 4, 6, 8]`.
- Colored cell positions inside each mini-grid are random distinct 3x3 coordinates. Empty cells are black `0`.
- The target is unique by colored-cell count and by number of non-black, non-gray colors. There is no tie-breaking among target blocks in generated data.
- The output is always the selected 3x3 target pattern upscaled into 3x3 constant blocks with gray separators restored.

## Reference Notes

- ARC-DSL computes all non-background objects, chooses the object with the fewest colors, normalizes it, upscales it by four, paints it over the input, then refills all original gray cells as gray. In this task that is exactly the unique four-color mini-grid expanded to the full 11x11 layout.
- The Code Golf solution enumerates possible 3x3 mini-grid offsets and builds an expanded candidate from each; its lexicographic minimum selects the sparse target pattern because the target lacks color `8`.
- ARC-GEN and examples agree: the selected mini-grid is the one with four colored cells, not a shape chosen by position or by a particular color.
