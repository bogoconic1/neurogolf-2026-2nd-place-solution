# task268 Semantics

## Sources

- Current champion builder: `solutions_py/task268.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task268.json`
- ARC-GEN task id: `aba27056`
- ARC-DSL task id: `aba27056`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_aba27056.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_aba27056.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task268.py`

## Pattern

The input contains one non-black colored hollow rectangular vessel, possibly flipped and/or transposed. The vessel is near one side of a square grid. Its outline is one arbitrary non-black color, and its border has a centered opening on the side facing the empty space. The output keeps the colored vessel and fills a yellow (`4`) fountain: all interior holes become yellow, the opening extends outward as a straight yellow stream, and two diagonal yellow rays project outward from the opening corners. Cells outside the grid are ignored.

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    obj = {(r, c) for r in range(h) for c in range(w) if grid[r][c] != 0}
    r0, r1 = min(r for r, _ in obj), max(r for r, _ in obj)
    c0, c1 = min(c for _, c in obj), max(c for _, c in obj)

    rect = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
    border = {
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if r in (r0, r1) or c in (c0, c1)
    }
    holes = rect - obj
    opening = border - obj

    def center(cells):
        return (
            sum(r for r, _ in cells) / len(cells),
            sum(c for _, c in cells) / len(cells),
        )

    hr, hc = center(holes)
    gr, gc = center(opening)
    dr = (gr > hr) - (gr < hr)
    dc = (gc > hc) - (gc < hc)

    def put(r, c):
        if 0 <= r < h and 0 <= c < w:
            out[r][c] = 4

    for r, c in holes:
        put(r, c)
    for step in range(10):
        for r, c in opening:
            put(r + step * dr, c + step * dc)

    pr, pc = dc, -dr
    ends = sorted(opening, key=lambda rc: rc[0] * pr + rc[1] * pc)
    if ends:
        for (sr, sc), (rr, cc) in ((ends[0], (dr - pr, dc - pc)), (ends[-1], (dr + pr, dc + pc))):
            r, c = sr, sc
            while 0 <= r < h and 0 <= c < w:
                put(r, c)
                r += rr
                c += cc
    return out
```

## Generator Constraints

The generator uses square grids with side length `5..10`. The vessel width is `5..size`, height is `3..size-2`, the left coordinate is chosen so the vessel fits, and a `row` offset of `0` or `1` can lift it one cell from the bottom before optional transformations. The vessel color is a random non-black color, possibly yellow. The top/open side has a centered gap at least one cell wide because width is at least five. The generated pair may be vertically flipped and/or transposed, giving four orientations for the same rule.

## Reference Notes

ARC-DSL treats the nonzero vessel as one object, forms its bounding-box border, finds missing border cells as the opening, fills all holes in the bounding rectangle yellow, repeatedly shifts the opening outward, then shoots yellow rays from opening corners toward selected adjacent background cells. The Code Golf solution repeatedly rotates/transposes and uses regex substitutions, confirming that the transformation is a directional fountain fill rather than a generic rectangle fill.
