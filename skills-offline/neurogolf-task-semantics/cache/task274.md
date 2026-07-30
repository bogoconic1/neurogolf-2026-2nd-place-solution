# task274 Semantics

## Sources

- Current champion builder: `solutions_py/task274.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task274.json`
- ARC-GEN task id: `b0c4d837`
- ARC-DSL task id: `b0c4d837`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_b0c4d837.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_b0c4d837.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task274.py`

## Pattern

The input is a gray cup outline (`5`) containing cyan water (`8`) and black background. The cup has a gray base on its bottom row and gray side walls. Above the cyan water there is an empty vertical space inside the cup; its height is the semantic value to recover.

The output is a fixed `3x3` glyph, padded by the NeuroGolf wrapper to the dense `[1,10,30,30]` output. For empty-space height `space`:

- `space >= 1`: output `(0,0)` is cyan.
- `space >= 2`: output `(0,1)` is cyan.
- `space >= 3`: output `(0,2)` is cyan.
- `space >= 4`: output `(1,2)` is cyan.
- all other cells are black.

There is no object tie-breaking: the generator creates one cup and one water region. The useful compact state is just `space in {1,2,3,4}`.

## Readable Python Solver

```python
def solve(grid):
    h = len(grid)
    w = len(grid[0])

    # Locate cyan water rows. In generated examples cyan appears only as water.
    cyan_rows = [r for r in range(h) if any(grid[r][c] == 8 for c in range(w))]
    top_water = min(cyan_rows)

    # The cup side walls extend through the empty space and water. The top side wall row is the first row containing gray wall cells above the base.
    wall_rows = []
    for r in range(h - 1):
        cols = [c for c, v in enumerate(grid[r]) if v == 5]
        if len(cols) >= 2:
            wall_rows.append(r)
    top_wall = min(wall_rows)

    space = top_water - top_wall
    out = [[0, 0, 0] for _ in range(3)]
    if space > 0:
        out[0][0] = 8
    if space > 1:
        out[0][1] = 8
    if space > 2:
        out[0][2] = 8
    if space > 3:
        out[1][2] = 8
    return out
```

## Generator Constraints

From ARC-GEN `task_b0c4d837.py`:

- `base` is 4..8.
- `water` is 1..7.
- `space` is 1..4.
- `row_gap` is 1..2.
- `col_gap` is 1..2.
- input width is `2 * col_gap + base`.
- input height is `water + space + row_gap + 1`.
- gray base occupies the full cup base row from `col_gap` to `col_gap + base - 1`.
- gray side walls occupy rows from `height - water - space - 1` through `height - 2` at the two cup edges.
- cyan water occupies the interior columns between the side walls on rows `height - water - 1` through `height - 2`.

The generated task always has one cup, gray is only the cup outline, cyan is only the water, and `space` is always positive.

## Reference Notes

ARC-DSL computes the height of gray cells, the height of cyan cells, derives `space = height(gray_object) - 1 - height(cyan_object)`, then builds the same 3x3 cyan glyph. The Code Golf solution uses a compact equivalent:

```python w = [8] * g.count(max(g, key=any)) + g[0] return [w[:3], w[5:2:-1], [0] * 3] ```

This works because the serialized grid row count and generated cup geometry encode the same `space` value. There is no disagreement between references; the generator gives stronger bounds than the golf solution exposes.
