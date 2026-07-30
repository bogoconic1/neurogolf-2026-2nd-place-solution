# task190 Semantics

## Sources

- Current champion builder: `solutions_py/task190.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task190.json`
- ARC-GEN task id: `7ddcd7ec`
- ARC-DSL task id: `7ddcd7ec`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_7ddcd7ec.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_7ddcd7ec.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task190.py`

## Pattern

The input is a fixed 10x10 grid with one nonzero color. That color forms a solid 2x2 square plus one to three singleton pixels placed diagonally just outside corners of the square. Each singleton marks a diagonal direction. The output preserves the input and extends every marked singleton along its same diagonal direction until it leaves the 10x10 grid. All other cells remain black. There is no color tie: every nonzero cell has the same color.

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])
    pts = [(r, c) for r, row in enumerate(grid) for c, v in enumerate(row) if v]
    color = grid[pts[0][0]][pts[0][1]]

    # Find the unique 2x2 square.
    cells = set(pts)
    top = left = None
    for r, c in pts:
        if {(r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)} <= cells:
            top, left = r, c
            break

    out = [row[:] for row in grid]
    block = {(top, left), (top + 1, left), (top, left + 1), (top + 1, left + 1)}
    for r, c in pts:
        if (r, c) in block:
            continue
        dr = -1 if r < top else 1
        dc = -1 if c < left else 1
        rr, cc = r, c
        while 0 <= rr < h and 0 <= cc < w:
            out[rr][cc] = color
            rr += dr
            cc += dc
    return out
```

## Generator Constraints

ARC-GEN uses `size=10`, so input and output are always 10x10. The 2x2 square top-left is sampled with row and column in `2..6`, leaving room for one diagonal marker cell around every corner. The color is any random nonzero ARC color. `dirs` is a random subset of one to three diagonal directions from `[0, 1, 2, 3]`, where the directions correspond to up-left, up-right, down-right, and down-left. The input contains the 2x2 square plus only the first marker cell for each selected diagonal; the output draws the full ray from that marker to the edge. The 2x2 square remains unchanged. There are no multiple colors, no overlapping rays with different colors, and no output-size change.

## Reference Notes

The ARC-DSL solver finds all objects, separates singleton objects from the non-singleton 2x2 square, gets the square color, and for each singleton shoots a ray from the singleton center in the direction determined by its position relative to the square. The Code Golf solution implements the same operation through repeated regex/rotation: extend a diagonal seed away from a 2x2 block, then rotate and repeat to cover all four diagonal orientations. The references agree with the generator.
