# task201 Semantics

## Sources

- Current champion builder: `solutions_py/task201.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task201.json`
- ARC-GEN task id: `846bdb03`
- ARC-DSL task id: `846bdb03`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_846bdb03.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_846bdb03.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task201.py`

## Pattern

The input contains two separated objects in a 13x13 canvas: an empty rectangular frame and a two-colour sprite. The frame has yellow (`4`) corner pixels, a left vertical side in one non-yellow colour, and a right vertical side in another non-yellow colour. The separate sprite uses exactly those two side colours and may be horizontally flipped.

The output is the framed rectangle cropped to its own bounding box, with the sprite inserted inside the frame at offset `(1,1)`. If the sprite's left half colour already matches the frame's left side colour, it is inserted as-is; otherwise the sprite is mirrored horizontally before insertion.

## Readable Python Solver

```python
def solve(grid):
    # Find the frame by yellow corners, crop it, find the separate non-yellow sprite,
    # orient the sprite so its left half colour matches the frame left side, then
    # paint it one cell inside the cropped frame.
    h, w = len(grid), len(grid[0])
    yellow = [(r, c) for r in range(h) for c in range(w) if grid[r][c] == 4]
    fr0, fr1 = min(r for r, _ in yellow), max(r for r, _ in yellow)
    fc0, fc1 = min(c for _, c in yellow), max(c for _, c in yellow)
    out = [row[fc0:fc1 + 1] for row in grid[fr0:fr1 + 1]]

    frame_cells = {(r, c) for r in range(fr0, fr1 + 1) for c in range(fc0, fc1 + 1)}
    sprite = [(r, c, grid[r][c]) for r in range(h) for c in range(w)
              if grid[r][c] not in (0, 4) and (r, c) not in frame_cells]
    sr0, sr1 = min(r for r, _, _ in sprite), max(r for r, _, _ in sprite)
    sc0, sc1 = min(c for _, c, _ in sprite), max(c for _, c, _ in sprite)
    sh, sw = sr1 - sr0 + 1, sc1 - sc0 + 1

    left_frame = out[1][0]
    left_sprite_colors = {v for r, c, v in sprite if c - sc0 < sw / 2}
    mirror = left_frame not in left_sprite_colors
    for r, c, v in sprite:
        rr = r - sr0 + 1
        cc0 = c - sc0
        cc = (sw - 1 - cc0 if mirror else cc0) + 1
        out[rr][cc] = v
    return out
```

## Generator Constraints

- Input canvas is fixed `13x13`.
- Sprite base width is `2..3`; sprite height is `2..4`.
- Output/frame width is `2 * (sprite_width + 1)`, so `6` or `8`.
- Output/frame height is `sprite_height + 2`, so `4..6`.
- The frame and separate sprite do not overlap in the input.
- The two sprite colours are random non-yellow colours and match the frame side colours.
- The separate sprite is optionally mirrored horizontally before being placed in the input.

## Reference Notes

The ARC-DSL solver identifies the no-yellow object as the separate sprite and the remaining object as the frame. It compares the frame left-side colour with the colour appearing in the left half of the sprite crop; if they differ, it applies a horizontal mirror before normalizing and shifting the sprite into the frame.

The Code Golf 2025 solution is a dense column-scan formulation of the same rule: recover frame columns and colours, orient the sprite based on the side-colour order, and emit the cropped frame with the sprite inserted.
