# task018 Semantics

## Sources

- Current champion builder: `solutions_py/task018.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task018.json`
- ARC-GEN task id: `0e206a2e`
- ARC-DSL task id: `0e206a2e`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_0e206a2e.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_0e206a2e.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task018.py`

## Pattern

The input contains one or two complete source sprites and one transformed clone
for each source. A sprite is a small connected multicolor object whose most
common color is a filler/background-within-object color and whose three rarer
colors are visible marker cells. In the clone, the filler-color cells are hidden
as black; only the rare marker cells are shown. The output removes the complete
source sprites and draws each clone fully, using the unique rotation/reflection
of the matching source sprite that agrees with the visible marker cells.

The grid size is preserved. Background remains 0. Source objects do not appear
in the output; only completed clone objects are painted.

## Readable Python Solver

```python
def solve(grid):
    from collections import Counter, deque

    H, W = len(grid), len(grid[0])
    out = [[0 for _ in range(W)] for _ in range(H)]
    colors = [v for row in grid for v in row if v]
    filler = Counter(colors).most_common(1)[0][0]

    def neighbors(r, c):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr or dc:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W:
                        yield nr, nc

    seen = set()
    objects = []
    for r in range(H):
        for c in range(W):
            if grid[r][c] == 0 or (r, c) in seen:
                continue
            q = deque([(r, c)])
            seen.add((r, c))
            cells = []
            while q:
                x, y = q.popleft()
                cells.append((x, y, grid[x][y]))
                for nx, ny in neighbors(x, y):
                    if grid[nx][ny] and (nx, ny) not in seen:
                        seen.add((nx, ny))
                        q.append((nx, ny))
            objects.append(cells)

    # Full sources contain the filler color and more than one nonzero color.
    sources = []
    markers = []
    for obj in objects:
        obj_colors = {v for _, _, v in obj}
        if filler in obj_colors and len(obj_colors) > 1:
            sources.append(obj)
        else:
            markers.extend(obj)

    marker_by_pos = {(r, c): v for r, c, v in markers if v != filler}

    def normalize(obj):
        r0 = min(r for r, _, _ in obj)
        c0 = min(c for _, c, _ in obj)
        return [(r - r0, c - c0, v) for r, c, v in obj]

    def transforms(cells):
        pts = [(r, c, v) for r, c, v in cells]
        variants = []
        for kind in range(8):
            cur = []
            for r, c, v in pts:
                if kind == 0:
                    x, y = r, c
                elif kind == 1:
                    x, y = r, -c
                elif kind == 2:
                    x, y = -r, c
                elif kind == 3:
                    x, y = -r, -c
                elif kind == 4:
                    x, y = c, r
                elif kind == 5:
                    x, y = c, -r
                elif kind == 6:
                    x, y = -c, r
                else:
                    x, y = -c, -r
                cur.append((x, y, v))
            r0 = min(r for r, _, _ in cur)
            c0 = min(c for _, c, _ in cur)
            norm = sorted((r - r0, c - c0, v) for r, c, v in cur)
            if norm not in variants:
                variants.append(norm)
        return variants

    used_marker_positions = set()
    for source in sources:
        src = normalize(source)
        best = None
        for shape in transforms(src):
            rare = [(r, c, v) for r, c, v in shape if v != filler]
            for mr, mc, mv in markers:
                for rr, rc, rv in rare:
                    if mv != rv:
                        continue
                    dr, dc = mr - rr, mc - rc
                    score = 0
                    ok = True
                    for sr, sc, sv in rare:
                        pos = (sr + dr, sc + dc)
                        if pos in marker_by_pos:
                            if marker_by_pos[pos] != sv:
                                ok = False
                                break
                            score += 1
                    if ok and score >= 3:
                        best = (score, dr, dc, shape)
                        break
                if best:
                    break
            if best:
                break
        if best is None:
            continue
        _, dr, dc, shape = best
        for sr, sc, sv in shape:
            r, c = sr + dr, sc + dc
            if 0 <= r < H and 0 <= c < W:
                out[r][c] = sv
                if sv != filler:
                    used_marker_positions.add((r, c))

    return out
```

## Generator Constraints

- Grid width and height are independently chosen from 12 through 24.
- There are one or two sprites. Each source sprite and its clone are placed in
  non-overlapping square bounding regions with a margin of at least 3 cells.
- Each sprite has width 3 through 6 and height `9 - width`; its generated
  connected body has 6 through 12 cells.
- Four nonzero colors are sampled. The fourth color is assigned to most sprite
  cells and is therefore the global/common filler color. Three selected sprite
  cells receive three rare marker colors; they are guaranteed not to all lie in
  one row or one column.
- The input contains each full source sprite. The corresponding clone is
  transformed and placed elsewhere, but only the rare marker-color cells are
  visible in the input. Filler-color clone cells are black.
- The output contains the full transformed clone sprite and does not contain the
  original source sprite.
- Transform choices are restricted to the generator's four rotate/mirror cases,
  while the ARC-DSL reference searches the dihedral closure; using all eight
  normalized square symmetries is a safe readable solver description.

## Reference Notes

- ARC-DSL identifies multicolor objects as complete sources, finds the global most frequent nonzero color as the filler, filters source cells to the rare marker colors, tries mirror/rotation compositions, paints matching shifted source cells, then covers/removes the original sources.
- The Code Golf solution performs a broad local search: it enumerates connected colored neighborhoods as candidate marker sets, builds transformed coordinate tuples, and paints a transformed source when enough cells overlap the visible marker evidence.
- The generator explicitly creates only four transform cases, but the DSL/code golf references use a larger transform family. This is not a contradiction: the larger family is a compact way to cover the generator cases and symmetric object degeneracies.
