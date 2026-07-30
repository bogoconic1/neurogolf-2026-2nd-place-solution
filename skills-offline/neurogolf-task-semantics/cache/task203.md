# task203 Semantics

## Sources

- Current champion builder: `solutions_py/task203.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task203.json`
- ARC-GEN task id: `85c4e7cd`
- ARC-DSL task id: `85c4e7cd`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_85c4e7cd.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_85c4e7cd.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task203.py`

## Pattern

The input is an even square grid of size `2n x 2n`, where `n` is the number of colors. The grid is made of concentric square layers. Layer `d = min(r, c, H-1-r, W-1-c)` has color `colors[d]`, so the outer border is `colors[0]` and the central `2x2` layer is `colors[n-1]`. The output preserves the exact same nested-square geometry but reverses the palette by layer: layer `d` becomes `colors[n-1-d]`.

## Readable Python Solver

```python
def solve(grid):
    size = len(grid)
    n = size // 2
    colors = [grid[i][i] for i in range(n)]
    out = []
    for r in range(size):
        row = []
        for c in range(size):
            depth = min(r, c, size - 1 - r, size - 1 - c)
            row.append(colors[n - 1 - depth])
        out.append(row)
    return out
```

Equivalent color-map view:

```python
def solve_by_map(grid):
    n = len(grid) // 2
    colors = [grid[i][i] for i in range(n)]
    repl = {colors[i]: colors[n - 1 - i] for i in range(n)}
    return [[repl[v] for v in row] for row in grid]
```

## Generator Constraints

ARC-GEN chooses `3..9` distinct non-black colors. The grid size is exactly `2 * len(colors)`, so dimensions are one of `6, 8, 10, 12, 14, 16, 18`. There is no background and no noise. Every chosen color appears in exactly one concentric layer, and all layers are present. The four quadrants are filled symmetrically with `colors[min(r,c)]`, which is the same as distance-to-nearest-edge layering over the full square.

## Reference Notes

The ARC-DSL solver partitions the grid into monochrome objects/layers, orders objects by increasing size and by decreasing size, extracts colors from the decreasing-size order, recolors the increasing-size order with those reversed colors, and paints the recolored objects back onto the grid. The Code Golf solution uses the center row as a lookup row to map each input color to its reversed counterpart.
