# task038 Semantics

## Sources

- Current champion builder: `solutions_py/task038.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task038.json`
- ARC-GEN task id: `1fad071e`
- ARC-DSL task id: `1fad071e`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_1fad071e.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_1fad071e.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task038.py`

## Pattern

The input is a `9x9` grid containing non-overlapping red/blue `2x2` blocks and red/blue singleton pixels. The output is a `1x5` row. It contains one color-1 cell for each full `2x2` object of color 1, followed by background zeros.

In the local ARC-GEN code, `common.blue()` is value `1` and `common.red()` is value `2`; the DSL solver confirms color `ONE` is the counted color. The task is therefore count the number of 2x2 color-1 objects and emit that count as a unary row of ones.

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])
    seen = set()
    count = 0
    for r in range(h - 1):
        for c in range(w - 1):
            cells = [(r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)]
            if all(grid[rr][cc] == 1 for rr, cc in cells):
                if not any(cell in seen for cell in cells):
                    count += 1
                    seen.update(cells)
    return [[1 if c < count else 0 for c in range(5)]]
```

## Generator Constraints

ARC-GEN uses `size=9`. It samples `5..7` non-overlapping `2x2` boxes, then colors the first `1..num_big_boxes-2` of them color 1 and the rest color 2. It may add singleton color-1 or color-2 pixels, but only when the singleton does not touch the same color in its 8-neighborhood. This prevents singletons from forming or extending a same-color `2x2` object.

The number of counted color-1 `2x2` boxes is therefore between 1 and 5. The output width is always 5.

## Reference Notes

The ARC-DSL solver finds connected objects, filters to color `ONE`, filters to size four, counts them, and emits that many ones followed by zeros to width five. The Code Golf solution counts the substring pattern `"1, 1"` in the grid string and maps that count into every other slot of a padded unary row; this works because each full `2x2` color-1 box contributes a fixed string pattern while singleton constraints avoid false positives.
