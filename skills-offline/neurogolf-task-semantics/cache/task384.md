# task384 Semantics

## Sources

- Current champion builder: `solutions_py/task384.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task384.json`
- ARC-GEN task id: `f25fbde4`
- ARC-DSL task id: `f25fbde4`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_f25fbde4.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_f25fbde4.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task384.py`

## Pattern

The input is a 9x9 black grid containing one yellow (`4`) object. The output is the tight bounding box around all yellow cells, upscaled by a factor of 2 in both height and width. Each yellow source cell becomes a 2x2 yellow block and each black cell inside the bounding box becomes a 2x2 black block. The output dimensions are therefore `2 * bbox_height` by `2 * bbox_width`, placed in the top-left of the NeuroGolf output canvas with all remaining cells no-color/black.

## Readable Python Solver

```python
def solve(grid):
    pts = [(r, c) for r, row in enumerate(grid) for c, v in enumerate(row) if v == 4]
    r0 = min(r for r, _ in pts)
    r1 = max(r for r, _ in pts) + 1
    c0 = min(c for _, c in pts)
    c1 = max(c for _, c in pts) + 1
    crop = [row[c0:c1] for row in grid[r0:r1]]
    out = []
    for row in crop:
        doubled = []
        for v in row:
            doubled.extend([v, v])
        out.append(doubled[:])
        out.append(doubled[:])
    return out
```

## Generator Constraints

ARC-GEN starts from a random object bounding box with width 3..5 and height 3..4. It samples between one quarter and three quarters of the bounding-box cells and requires the sampled yellow pixels to be diagonally connected. The sampled coordinates are normalized so the object touches the top and left of its own bounding box. The object is then placed on a 9x9 black grid at a positive row/column offset, with offsets bounded by the generated object height/width. There is exactly one foreground color, yellow (`4`), and the output is always the normalized object crop scaled to width `2*bbox_width` and height `2*bbox_height`.

## Reference Notes

The ARC-DSL solver computes `objects(I, T, T, T)`, takes the first object, extracts its `subgrid`, and applies `upscale(..., TWO)`. The Code Golf 2025 solution is a compact recursive/transposed form that effectively removes all-empty outer rows/columns and duplicates every remaining row and column. All three references agree: the crop is the tight yellow-object bounding box, not a fixed-size crop, and the scale factor is exactly 2.
