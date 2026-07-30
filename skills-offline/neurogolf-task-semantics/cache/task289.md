# task289 Semantics

## Sources

- Current champion builder: `solutions_py/task289.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task289.json`
- ARC-GEN task id: `b91ae062`
- ARC-DSL task id: `b91ae062`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_b91ae062.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_b91ae062.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task289.py`

## Pattern

The input is a 3x3 grid with black background and 3 to 5 colored cells. The colored cells use `k` distinct non-black colors, where `k` can be 1 through the number of colored cells and every chosen non-black color appears at least once. The output is the entire input grid uniformly upscaled by factor `k`, so each input cell becomes a `k x k` solid block of the same color. Output side length is `3*k` and therefore ranges from 3 to 15.

## Readable Python Solver

```python
def solve(grid):
    colors = {v for row in grid for v in row if v != 0}
    scale = len(colors)
    out = []
    for row in grid:
        expanded_row = []
        for v in row:
            expanded_row.extend([v] * scale)
        for _ in range(scale):
            out.append(expanded_row[:])
    return out
```

## Generator Constraints

ARC-GEN fixes the input size at 3x3. It samples 3 to 5 distinct cell positions, then chooses 1 through that many non-black colors. The index list is initialized so every chosen color is used at least once, then remaining colored cells reuse a random chosen color. All non-sampled cells are black. The output is produced by `common.grid_enhance(size, len(colors), ...)`, matching an upscale by the number of non-black colors.

## Reference Notes

ARC-DSL computes `numcolors(I)`, decrements it, and calls `upscale(I, factor)`. Because black is always present in the 3x3 input, this factor is exactly the number of non-black colors. The Code Golf solution is a compact expression that repeats rows and cells according to the number of distinct colors represented in the input.
