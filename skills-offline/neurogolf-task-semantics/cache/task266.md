# task266 Semantics

## Sources

- Current champion builder: `solutions_py/task266.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task266.json`
- ARC-GEN task id: `a9f96cdd`
- ARC-DSL task id: `a9f96cdd`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_a9f96cdd.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_a9f96cdd.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task266.py`

## Pattern

The input is a fixed `3x5` grid containing exactly one red (`2`) pixel and otherwise black. The output removes the red pixel and places up to four colored pixels on its diagonal neighbors: green (`3`) at up-left, magenta/pink (`6`) at up-right, cyan (`8`) at down-left, and orange (`7`) at down-right. Diagonal positions falling outside the grid are ignored.

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])
    out = [[0 for _ in range(w)] for _ in range(h)]
    src = None
    for r in range(h):
        for c in range(w):
            if grid[r][c] == 2:
                src = (r, c)
                break
        if src is not None:
            break
    if src is None:
        return out
    r, c = src
    for dr, dc, color in [(-1, -1, 3), (-1, 1, 6), (1, -1, 8), (1, 1, 7)]:
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            out[rr][cc] = color
    return out
```

## Generator Constraints

The generator always uses width `5` and height `3`. The red source coordinate is uniformly any row `0..2` and column `0..4`. There are no other nonblack cells. The output grid has the same `3x5` size. Edge and corner cases simply omit off-grid diagonal outputs.

## Reference Notes

ARC-DSL finds the set of red cells, clears red to black, then fills the four diagonal shifts with fixed colors. The code-golf solution uses a compact convolution-like expression over neighboring red pixels, confirming the rule is purely local and color-fixed.
