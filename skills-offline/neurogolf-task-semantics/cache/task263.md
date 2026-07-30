# task263 Semantics

## Sources

- Current champion builder: `solutions_py/task263.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task263.json`
- ARC-GEN task id: `a87f7484`
- ARC-DSL task id: `a87f7484`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_a87f7484.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_a87f7484.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task263.py`

## Pattern

The input is made from `K` colored 3x3 sprite panels, where `K` is 3, 4, or 5. In the untransposed case the panels are stacked vertically in a `3K x 3` grid; in the transposed case they are laid out horizontally in a `3 x 3K` grid. Each panel uses one distinct nonzero color on black/background cells. All but one panel have the same binary sprite shape (`basic`), and exactly one panel has a different binary sprite shape (`weird`) with a different number of colored cells. The output is the weird 3x3 panel, preserving its color and the input orientation.

## Readable Python Solver

```python
def solve(grid):
    rows = len(grid)
    cols = len(grid[0])
    was_portrait = rows > cols

    def transpose(g):
        return [list(row) for row in zip(*g)]

    work = transpose(grid) if was_portrait else [row[:] for row in grid]
    # Now work is 3 rows by 3*K columns.
    k = len(work[0]) // 3
    panels = []
    for i in range(k):
        panel = [row[3 * i : 3 * i + 3] for row in work]
        panels.append(panel)

    def zero_signature(panel):
        return tuple((r, c) for r in range(3) for c in range(3) if panel[r][c] == 0)

    signatures = [zero_signature(panel) for panel in panels]
    counts = {sig: signatures.count(sig) for sig in set(signatures)}
    weird_index = min(range(k), key=lambda i: counts[signatures[i]])
    out = panels[weird_index]
    return transpose(out) if was_portrait else out
```

## Generator Constraints

- `K = len(colors)` is sampled uniformly from 3..5.
- Colors are distinct nonzero ARC colors selected by `common.random_colors(K)`.
- `basicrows/basiccols` and `weirdrows/weirdcols` are independently sampled Conway sprites.
- The generator rejects equal-size basic/weird sprites, so the weird panel has a different number of colored cells from the repeated basic panels.
- Exactly one color index is selected as `weird`; every other panel uses the basic sprite.
- `xpose` randomly transposes both input and output. Therefore valid input dimensions are either `3K x 3` or `3 x 3K`, and output is always `3 x 3` in the matching orientation.
- Background is color 0. The task identifies the odd panel by its zero/background mask, not by its foreground color value.

## Reference Notes

The ARC-DSL solver first checks whether the input is portrait. If it is, it applies `dmirror` to normalize the panels into a horizontal strip. It then uses `numcolors(I) - 1` to split the normalized strip into foreground-color panels, computes each panel's zero set, selects the least-common zero set, extracts that panel, and applies the same orientation transform back to the result.

The Code Golf solution is a compact recursive/transposed implementation of the same idea: normalize orientation, compare boolean support masks of 3x3 chunks, rotate through chunks until the unique mask is found, then return that 3x3 chunk.
