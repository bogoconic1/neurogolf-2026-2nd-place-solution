# task180 Semantics

## Sources

- Current champion builder: `solutions_py/task180.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task180.json`
- ARC-GEN task id: `75b8110e`
- ARC-DSL task id: `75b8110e`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_75b8110e.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_75b8110e.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task180.py`

## Pattern

The input is an 8x8 grid split into four 4x4 quadrants. The top-left quadrant contains color-4 mask pixels, top-right contains color-5 mask pixels, bottom-left contains color-6 mask pixels, and bottom-right contains color-9 mask pixels. The output is a 4x4 overlay of those four masks at corresponding coordinates. If multiple masks are present at a cell, priority is color 5 first, then 6, then 9, then 4, otherwise background 0.

## Readable Python Solver

```python
def solve(grid):
    size = len(grid) // 2
    out = [[0 for _ in range(size)] for _ in range(size)]
    # Paint in low-to-high priority so later colors overwrite earlier ones.
    quadrants = [
        (4, 0, 0),
        (9, size, size),
        (6, size, 0),
        (5, 0, size),
    ]
    for color, ro, co in quadrants:
        for r in range(size):
            for c in range(size):
                if grid[ro + r][co + c] == color:
                    out[r][c] = color
    return out
```

## Generator Constraints

ARC-GEN defaults to `size=4`, so the input is 8x8 and output is 4x4. It samples arbitrary pixel subsets inside a 4x4 coordinate frame for each of four colors `(4,5,6,9)`, then places each subset in its assigned quadrant using offsets `(0,0)`, `(0,1)`, `(1,0)`, `(1,1)`. The same output coordinate can be present in multiple quadrants, and priority is resolved by painting in order `[4,9,6,5]`, leaving color 5 highest. Empty coordinates remain background 0.

## Reference Notes

ARC-DSL splits the grid into left/right and top/bottom halves, extracts nonzero objects from three quadrants, and paints them over the top-left quadrant. The paint order gives the same priority overlay as the generator. The Code Golf solution zips the four quadrant rows and takes the first truthy/nonzero value after arranging the priority order, confirming that this is a coordinate-wise quadrant overlay rather than object motion. No reference disagreement was found.
