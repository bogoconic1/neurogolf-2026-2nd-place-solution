# task397 Semantics

## Sources

- Current champion builder: `solutions_py/task397.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task397.json`
- ARC-GEN task id: `fcc82909`
- ARC-DSL task id: `fcc82909`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_fcc82909.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_fcc82909.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task397.py`

## Pattern

The input is a 10x10 black grid containing two or three separated non-green 2x2 colored boxes. Each 2x2 box may contain repeated colors; it has between two and four distinct colors. The output starts as a copy of the input and adds a green vertical shadow directly below each 2x2 box. The shadow is two columns wide, aligned with the box, starts immediately below the 2x2 box, and its height equals the number of distinct colors inside that 2x2 box. Existing colored box cells remain unchanged.

For a box with top-left `(r, c)`, bottom row `r+1`, and distinct color count `k`, fill color `3` in rows `r+2 .. r+1+k`, columns `c` and `c+1`.

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    green = 3

    # Detect generator-style 2x2 non-black, non-green boxes by their top-left.
    # Boxes are separated enough that each top-left is the only 2x2 all-colored
    # window for that object.
    for r in range(h - 1):
        for c in range(w - 1):
            vals = [grid[r + dr][c + dc] for dr in (0, 1) for dc in (0, 1)]
            if all(v not in (0, green) for v in vals):
                # Avoid counting overlapping shifted windows inside/near a box.
                if r > 0 and any(grid[r - 1][c + dc] not in (0, green) for dc in (0, 1)):
                    continue
                if c > 0 and any(grid[r + dr][c - 1] not in (0, green) for dr in (0, 1)):
                    continue
                k = len(set(vals))
                for rr in range(r + 2, min(h, r + 2 + k)):
                    for cc in (c, c + 1):
                        out[rr][cc] = green
    return out
```

## Generator Constraints

- Grid size is fixed at 10x10.
- There are two or three boxes.
- Each box is exactly 2x2 and placed with top-left row/column in `0..8`.
- Box columns are sorted and non-overlapping in width 2 during random generation; hand-authored validation examples still follow separated 2x2 object intent.
- Box colors exclude green (`3`) and are sampled from a color list of size 2 through 4; the 2x2 cells may repeat colors.
- The shadow height is the number of distinct colors in that box, so it is 2, 3, or 4.
- Random generation rejects boxes whose top rows are too close vertically for adjacent boxes; this prevents ambiguous touching shadows/objects in generated data.

## Reference Notes

- ARC-DSL uses `objects(I, F, T, T)` to get multicolor objects, `numcolors` for each object's distinct color count, and fills the box from one row below the object's lower-left corner to `numcolors` rows below its lower-right corner with green. This confirms shadow height is distinct-color count, not total colored cells or connected-component area.
- ARC-GEN clarifies the exact random constraints: 10x10 grid, two or three 2x2 boxes, no green inside boxes, and shadow color green.
- Code Golf 2025 checks every possible 2x2 window, derives the distinct-color count, and writes green under the window, matching the generator rule in compact form.
