# task386 Semantics

## Sources

- Current champion builder: `solutions_py/task386.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task386.json`
- ARC-GEN task id: `f2829549`
- ARC-DSL task id: `f2829549`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_f2829549.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_f2829549.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task386.py`

## Pattern

The input is always a `4x7` grid. Column `3` is a vertical blue separator. The left `4x3` panel contains orange marks and the right `4x3` panel contains gray marks. The output is a `4x3` grid with green (`3`) exactly where both the left and right panels are empty at the same row and column; all occupied positions remain black.

## Readable Python Solver

```python
def solve(grid):
    h = len(grid)
    w = (len(grid[0]) - 1) // 2
    out = [[0 for _ in range(w)] for _ in range(h)]
    for r in range(h):
        for c in range(w):
            if grid[r][c] == 0 and grid[r][w + 1 + c] == 0:
                out[r][c] = 3
    return out
```

## Generator Constraints

ARC-GEN uses fixed panel size `height=4`, `width=3`, so the input is `4x7` and the output is `4x3`. It independently samples random occupied pixels for the left and right panels. Left marks are orange, right marks are gray, and the separator is blue. Only emptiness matters; the specific nonzero side colors are reliable but do not need to be preserved.

## Reference Notes

ARC-DSL takes the left and right halves, finds zero cells in each, intersects those coordinate sets, creates a black canvas with the left-half shape, and fills the intersection with green. The Code Golf solution expresses the same row-wise emptiness test across corresponding left/right cells.
