# task399 Semantics

## Sources

- Current champion builder: `solutions_py/task399.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task399.json`
- ARC-GEN task id: `ff28f65a`
- ARC-DSL task id: `ff28f65a`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_ff28f65a.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_ff28f65a.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task399.py`

## Pattern

The input is a small square grid, size 3 through 7, containing one to five red 2x2 boxes on a black background. The boxes may touch or partially overlap according to the generator's special overlap rule, but the intended object count is the number of generated red 2x2 placements. The output is always a 3x3 grid. It places blue cells in a fixed order according to the number of red boxes: first top-left, second top-right, third center, fourth bottom-left, fifth bottom-right. All other output cells are black.

Equivalently, if `n` is the number of red 2x2 objects, output positions `(0,0), (0,2), (1,1), (2,0), (2,2)` are blue for the first `n` positions and black afterwards.

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])
    red = 2
    blue = 1

    # Count top-left corners of red 2x2 blocks. The generator prevents ambiguous
    # placements where a different interpretation would be needed.
    count = 0
    for r in range(h - 1):
        for c in range(w - 1):
            if all(grid[r + dr][c + dc] == red for dr in (0, 1) for dc in (0, 1)):
                count += 1

    out = [[0 for _ in range(3)] for _ in range(3)]
    order = [(0, 0), (0, 2), (1, 1), (2, 0), (2, 2)]
    for r, c in order[:count]:
        out[r][c] = blue
    return out
```

## Generator Constraints

- Input size is a square from 3 to 7.
- Red is color `2`; output blue is color `1`; background is black `0`.
- Number of generated boxes depends on size: size 3 or 4 has one box, size 5 has one or two, size 6 has two or three, and size 7 has three to five.
- Each box is a 2x2 red square with top-left row/col in `0..size-2`.
- The generator rejects placements under a custom overlap predicate. It allows some touching/near placements, but rejects placements that would make the set of generated 2x2 objects ambiguous under the intended object extraction.
- Output is always exactly 3x3 and uses only blue dots at the fixed count-code positions.

## Reference Notes

- ARC-DSL uses `objects(I, T, F, T)` and `size` to count the red objects, doubles the count, fills a 1x9 row at even columns `0,2,...,2*(n-1)`, splits that row into three chunks, and merges them to a 3x3 grid. This confirms the fixed row-major-dot count encoding.
- ARC-GEN clarifies the input size/count ranges and the exact output dot order for counts one through five.
- Code Golf 2025 uses the total grid sum as a proxy for count. Since every 2x2 red box contributes four cells of color 2, the total red sum separates the allowed counts even when boxes touch in generator-valid ways.
