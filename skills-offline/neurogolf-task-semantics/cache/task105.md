# task105 Semantics

## Sources

- Current champion builder: `solutions_py/task105.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task105.json`
- ARC-GEN task id: `4612dd53`
- ARC-DSL task id: `4612dd53`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_4612dd53.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_4612dd53.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task105.py`

## Pattern

The input is a sparse blue (`1`) subset of a hidden red/blue (`2`/`1`) line
drawing. The drawing is always the perimeter of an axis-aligned rectangle, and
it may additionally contain one complete internal horizontal or vertical cutline.
Only blue cells are visible in the input; hidden cells belonging to the same
rectangle/cutline drawing must be filled red in the output. Existing blue cells
remain blue, all cells outside the drawing remain black, and the output grid has
the same size as the input.

The rectangle is recovered as the bounding box of the visible blue cells. Any
visible blue cells strictly inside that bounding box identify the optional
cutline: a shared interior row means a horizontal cutline, and a shared interior
column means a vertical cutline. If there are no interior blue cells, only the
rectangle perimeter is completed.

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])
    blues = [(r, c) for r in range(h) for c in range(w) if grid[r][c] == 1]
    r0 = min(r for r, _ in blues)
    r1 = max(r for r, _ in blues)
    c0 = min(c for _, c in blues)
    c1 = max(c for _, c in blues)

    target = set()
    for c in range(c0, c1 + 1):
        target.add((r0, c))
        target.add((r1, c))
    for r in range(r0, r1 + 1):
        target.add((r, c0))
        target.add((r, c1))

    inner = [(r, c) for r, c in blues if r0 < r < r1 and c0 < c < c1]
    if inner:
        rows = {r for r, _ in inner}
        cols = {c for _, c in inner}
        if len(rows) == 1:
            rr = next(iter(rows))
            for c in range(c0, c1 + 1):
                target.add((rr, c))
        elif len(cols) == 1:
            cc = next(iter(cols))
            for r in range(r0, r1 + 1):
                target.add((r, cc))

    out = [row[:] for row in grid]
    for r, c in target:
        if out[r][c] == 0:
            out[r][c] = 2
    return out
```

## Generator Constraints

The grid width is fixed at `13`; generated height is `tall + 2..5`, where
`tall` is `7..9`. The hidden rectangle width is `7..9`, placed at columns
`2..wide+1`; its top row is random but leaves at least one row above and below.
The optional cutline branch is one of: no cutline, one internal horizontal line,
or one internal vertical line. Horizontal cutlines are not adjacent to the top or
bottom border; vertical cutlines are internal and away from the left border.
Every drawing cell is independently blue with probability about `3/4` or red
with probability about `1/4`; the input keeps only blue cells, while the output
contains both colors.

## Reference Notes

ARC-DSL computes the blue bounding box, fills that box red as a local work grid, then uses the visible blue cells' horizontal and vertical frontiers to infer which optional line family is present before underfilling red back into the original input. The Code Golf 2025 solution is a compact recursive transpose/line-completion program with the same idea: complete the hidden rectangle and the dominant internal line implied by blue support.

The only semantic ambiguity is an extremely sparse cutline with too few visible blue cells to reveal its orientation. The public references infer from visible blue support, so optimization should preserve that behavior rather than trying to hallucinate a cutline with no evidence.
