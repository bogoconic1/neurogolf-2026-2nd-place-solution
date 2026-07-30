# task089 Semantics

## Sources

- Current champion builder: `solutions_py/task089.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task089.json`
- ARC-GEN task id: `3e980e27`
- ARC-DSL task id: `3e980e27`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_3e980e27.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_3e980e27.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task089.py`

## Pattern

The 13x13 input contains several non-overlapping 3x3 sprite placements. There are up to two sprite families. The red-marked family has one full exemplar containing red plus its ordinary color, and later placements may show only the red marker. The output fills every red marker-only placement with the horizontally mirrored version of the full red exemplar. The green-marked family has one full exemplar containing green plus its ordinary color, and later placements may show only the green marker. The output fills every green marker-only placement with an unmirrored copy of the full green exemplar.

All original pixels remain, and missing pixels are painted into the marker-only 3x3 placements. Red and green marker pixels are part of the copied sprite pattern and keep their marker colors at the corresponding relative positions.

## Readable Python Solver

```python
def solve(grid):
    out = [row[:] for row in grid]
    h, w = len(grid), len(grid[0])

    def components():
        seen = [[False] * w for _ in range(h)]
        comps = []
        for r in range(h):
            for c in range(w):
                if grid[r][c] == 0 or seen[r][c]:
                    continue
                stack = [(r, c)]
                seen[r][c] = True
                cells = []
                while stack:
                    rr, cc = stack.pop()
                    cells.append((rr, cc, grid[rr][cc]))
                    for nr in range(rr - 1, rr + 2):
                        for nc in range(cc - 1, cc + 2):
                            if 0 <= nr < h and 0 <= nc < w and not seen[nr][nc] and grid[nr][nc] != 0:
                                seen[nr][nc] = True
                                stack.append((nr, nc))
                comps.append(cells)
        return comps

    comps = components()
    for marker, mirror in ((2, True), (3, False)):
        family = [cells for cells in comps if any(v == marker for _, _, v in cells)]
        if not family:
            continue
        template = max(family, key=len)
        tr0 = min(r for r, _, _ in template)
        tc0 = min(c for _, c, _ in template)
        pattern = [(r - tr0, c - tc0, v) for r, c, v in template]
        if mirror:
            pattern = [(dr, 2 - dc, v) for dr, dc, v in pattern]
        for cells in family:
            if cells is template:
                continue
            # Marker-only components locate the target 3x3 placement. Align by
            # the marker's relative position in the pattern.
            mr, mc, _ = next(cell for cell in cells if cell[2] == marker)
            pdr, pdc, _ = next(cell for cell in pattern if cell[2] == marker)
            r0, c0 = mr - pdr, mc - pdc
            for dr, dc, v in pattern:
                if 0 <= r0 + dr < h and 0 <= c0 + dc < w:
                    out[r0 + dr][c0 + dc] = v
    return out
```

## Generator Constraints

The grid size is always 13x13. Each sprite family is a diagonally connected subset of a 3x3 cell block with 4 to 6 pixels. The ordinary colors exclude red and green; the final sampled pixel of family 0 is red and the final sampled pixel of family 1 is green. The generator chooses one or both families, then places two or three 3x3 occurrences of each chosen family without overlap. For the first occurrence of a family, the full sprite is visible. For later occurrences, only the marker pixel is visible in the input; the output contains the full sprite. Later red-family occurrences are horizontally mirrored, while green-family occurrences are not mirrored.

## Reference Notes

ARC-DSL finds connected foreground objects, partitions red-containing and green-containing objects, chooses the largest object in each partition as the full exemplar, removes it from the target set, mirrors the red exemplar with `vmirror`, and shifts the chosen exemplar to each marker-only object's center. Code Golf expresses the same rule with local marker/template matching.
