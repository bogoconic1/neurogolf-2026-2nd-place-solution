# task133 Semantics

## Sources

- Current champion builder: `solutions_py/task133.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task133.json`
- ARC-GEN task id: `57aa92db`
- ARC-DSL task id: `57aa92db`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_57aa92db.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_57aa92db.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task133.py`

## Pattern

The input contains two to four copies of the same small sprite pattern. One copy is a complete unscaled prototype. The other copies are partial scaled copies: only the signature pixel block and one adjacent colored pixel block are shown. The output completes every partial copy by painting the whole prototype shape at that copy's scale and color. The signature pixel/block keeps its special `pcolor`; all other prototype cells are recolored to the copy color.

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])
    out = [row[:] for row in grid]

    # Connected components of nonzero cells, 4-neighbor.
    seen = set()
    comps = []
    for r in range(h):
        for c in range(w):
            if grid[r][c] == 0 or (r, c) in seen:
                continue
            stack = [(r, c)]
            seen.add((r, c))
            cells = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc, grid[rr][cc]))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and grid[nr][nc] != 0 and (nr, nc) not in seen:
                        seen.add((nr, nc))
                        stack.append((nr, nc))
            comps.append(cells)

    # The prototype has one signature-color cell and multiple cells of another color.
    def color_counts(comp):
        counts = {}
        for _, _, color in comp:
            counts[color] = counts.get(color, 0) + 1
        return counts

    proto = max(comps, key=lambda comp: max(color_counts(comp).values()) - min(color_counts(comp).values()))
    counts = color_counts(proto)
    pcolor = min(counts, key=counts.get)
    proto_color = max(counts, key=counts.get)
    pr0 = min(r for r, _, _ in proto)
    pc0 = min(c for _, c, _ in proto)
    proto_cells = [(r - pr0, c - pc0, color) for r, c, color in proto]
    sig_cells = [(r, c) for r, c, color in proto_cells if color == pcolor]
    sig_r, sig_c = min(sig_cells)

    for comp in comps:
        counts = color_counts(comp)
        if pcolor not in counts:
            continue
        p_cells = [(r, c) for r, c, color in comp if color == pcolor]
        scale = int(round(len(p_cells) ** 0.5))
        if scale < 1:
            continue
        colors = [color for color in counts if color not in (0, pcolor)]
        if not colors:
            continue
        copy_color = colors[0]
        anchor_r = min(r for r, _ in p_cells) - sig_r * scale
        anchor_c = min(c for _, c in p_cells) - sig_c * scale
        for rr, cc, color in proto_cells:
            paint = pcolor if color == pcolor else copy_color
            for dr in range(scale):
                for dc in range(scale):
                    r = anchor_r + rr * scale + dr
                    c = anchor_c + cc * scale + dc
                    if 0 <= r < h and 0 <= c < w:
                        out[r][c] = paint
    return out
```

## Generator Constraints

ARC-GEN samples grid width and height from `10..30` and creates `2..4` sprite copies. The base sprite lives in a `3x3` box and contains `3..5` connected cells from `continuous_creature`; the first base pixel is the signature location. Copy magnifiers are `1..4`, with the first copy forced to magnifier `1` and fully visible. Other copies can have any magnifier `1..4` and show only the signature block plus one colored block adjacent to the signature. Copies are placed without overlap using a two-cell margin. The signature color `pcolor` is distinct from all copy colors.

## Reference Notes

The ARC-DSL solver identifies the complete prototype as the object whose per-color counts have the largest range, takes its least frequent color as the signature color, normalizes that prototype, and records the signature offset. It then finds all objects containing the signature color, infers each object's scale from its bounding-box width, shifts/upscales the normalized prototype to that anchor, recolors non-signature cells to the observed copy color, and paints the original visible pixels on top. The Code Golf solution follows the same idea with connected-component growth, signature-color detection, scale from repeated counts, and coordinate arithmetic.
