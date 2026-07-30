# task340 Semantics

## Sources

- Current champion builder: `solutions_py/task340.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task340.json`
- ARC-GEN task id: `d687bc17`
- ARC-DSL task id: `d687bc17`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_d687bc17.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_d687bc17.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task340.py`

## Pattern

The input is a rectangular grid, size `10..20` by `10..20`, with a colored frame on the four sides. The top, right, bottom, and left borders each have a distinct nonzero color. Interior cells contain isolated single pixels. Some interior pixels use one of the four frame colors; other interior pixels are garbage colors not used by the frame.

The output keeps the same frame and removes all original interior singletons. Each interior singleton whose color matches a frame side is projected to the cell just inside that matching side, preserving the singleton's orthogonal coordinate:

- top-frame color at `(r,c)` moves to `(1,c)`
- right-frame color at `(r,c)` moves to `(r,width-2)`
- bottom-frame color at `(r,c)` moves to `(height-2,c)`
- left-frame color at `(r,c)` moves to `(r,1)`

Interior singletons whose color is not one of the four frame colors disappear.

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])
    top = grid[0][1]
    right = grid[1][w - 1]
    bottom = grid[h - 1][1]
    left = grid[1][0]

    out = [[0 for _ in range(w)] for _ in range(h)]
    for c in range(1, w - 1):
        out[0][c] = top
        out[h - 1][c] = bottom
    for r in range(1, h - 1):
        out[r][0] = left
        out[r][w - 1] = right

    for r in range(1, h - 1):
        for c in range(1, w - 1):
            v = grid[r][c]
            if v == top:
                out[1][c] = v
            elif v == right:
                out[r][w - 2] = v
            elif v == bottom:
                out[h - 2][c] = v
            elif v == left:
                out[r][1] = v
    return out
```

## Generator Constraints

- Width and height are independently sampled from `10..20`.
- Four distinct edge colors are sampled for top, right, bottom, and left.
- For each edge color, the generator places `1..3` interior singleton pixels.
  Top and bottom colors get sampled columns with random rows; left and right colors get sampled rows with random columns.
- The generator avoids duplicate same-color cells sharing the same row/column in the natural projection direction.
- It also places `1..3` garbage singleton pixels whose colors are excluded from the four edge colors; these disappear in the output.
- Interior generated rows/columns stay at least one cell away from the frame, so projected marks land in row `1`, row `h-2`, column `1`, or column `w-2`.

## Reference Notes

The ARC-DSL solver separates size-one objects from the frame objects, keeps only singleton objects whose color is contained in a frame color set, computes each singleton's gravity vector toward the matching frame object, shifts the singleton by that vector, covers/removes all original singletons, and paints the shifted projections. The Code Golf solution recursively rotates/transposes the grid so the same edge-projection rule can be applied for each side.
