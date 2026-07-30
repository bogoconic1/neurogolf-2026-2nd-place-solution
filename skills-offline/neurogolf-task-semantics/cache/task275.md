# task275 Semantics

## Sources

- Current champion builder: `solutions_py/task275.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task275.json`
- ARC-GEN task id: `b190f7f5`
- ARC-DSL task id: `b190f7f5`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_b190f7f5.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_b190f7f5.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task275.py`

## Pattern

The input contains two square `size x size` panels, where `size` is 3 or 4. One panel is a colored pattern using colors 1..4 on black background. The other panel is a cyan (`8`) binary mask. The panels may be arranged left/right or top/bottom, and either panel can be first depending on `pairwise`.

The output is a Kronecker/block expansion: for every colored cell `(row, col)` in the color panel and every cyan mask cell `(r, c)`, output cell `(row * size + r, col * size + c)` receives that color. All other output cells are black. Thus the output canvas is `size*size` by `size*size`, padded to NeuroGolf's standard `[1,10,30,30]` one-hot tensor.

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])
    size = min(h, w)
    if h > w:
        a = [row[:] for row in grid[:size]]
        b = [row[:] for row in grid[size:2*size]]
    else:
        a = [row[:size] for row in grid]
        b = [row[size:2*size] for row in grid]
    # The cyan-only panel is the mask; the other nonempty panel is the color grid.
    if any(8 in row for row in a):
        mask, colors = a, b
    else:
        colors, mask = a, b
    out = [[0 for _ in range(size * size)] for _ in range(size * size)]
    for gr in range(size):
        for gc in range(size):
            color = colors[gr][gc]
            if color == 0 or color == 8:
                continue
            for mr in range(size):
                for mc in range(size):
                    if mask[mr][mc] == 8:
                        out[gr * size + mr][gc * size + mc] = color
    return out
```

## Generator Constraints

The generated `size` is 3 or 4. The colored panel has a diagonally connected subset of cells with colors sampled from two or three of colors `1..4`, and at least two distinct colors appear. The mask panel contains a random subset of cyan cells. The paired panels are either horizontal or vertical; for `pairwise` 0/1 the colored panel is in the first side and the mask is in the other side, while for `pairwise` 2/3 the colored panel is offset and the mask is first. The output dimensions are `9x9` for size 3 and `16x16` for size 4 before standard padding.

## Reference Notes

ARC-DSL splits the input by portrait/landscape, selects the lower-color-count panel as the binary mask and the higher-color-count panel as the colored pattern, then repeats/upscales the color grid and fills zeros where the repeated mask has zeros. The Code Golf solution is a compact recursive expression for the same Kronecker product.
