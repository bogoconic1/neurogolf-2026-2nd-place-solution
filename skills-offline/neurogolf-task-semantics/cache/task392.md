# task392 Semantics

## Sources

- Current champion builder: `solutions_py/task392.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task392.json`
- ARC-GEN task id: `f8c80d96`
- ARC-DSL task id: `f8c80d96`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_f8c80d96.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_f8c80d96.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task392.py`

## Pattern

The input is a `10x10` black canvas with a non-gray colored partial set of repeated square/rectangular mats drawn from an anchor on either the top edge or left edge. Only the first few rings are shown. The output is the completed pattern clipped to the `10x10` canvas, with every non-pattern cell filled gray (`5`) and every pattern segment colored with the same non-gray input color.

## Readable Python Solver

```python
def solve(grid):
    n = 10
    color = max((v for row in grid for v in row if v not in (0, 5)), default=0)

    # Infer the visible colored coordinates, then infer anchor/thickness by
    # trying the small generator parameter space and matching the input prefix.
    colored = {(r, c) for r, row in enumerate(grid) for c, v in enumerate(row) if v == color}

    def draw(row, col, thick, show):
        partial = set()
        full = set()
        for i in range(n):
            radius = (thick + 1) * i
            for r in range(row - radius + thick, row + radius):
                for c in (col - radius + thick, col + radius - 1):
                    if 0 <= r < n and 0 <= c < n:
                        full.add((r, c))
                        if i <= show:
                            partial.add((r, c))
            for c in range(col - radius + thick, col + radius):
                for r in (row - radius + thick, row + radius - 1):
                    if 0 <= r < n and 0 <= c < n:
                        full.add((r, c))
                        if i <= show:
                            partial.add((r, c))
        return partial, full

    best_full = set()
    for row in range(n):
        for col in range(n):
            if not (row == 0 or col == 0):
                continue
            for thick in (1, 2):
                for show in (2, 3):
                    partial, full = draw(row, col, thick, show)
                    if partial == colored:
                        best_full = full
                        break

    out = [[5 for _ in range(n)] for _ in range(n)]
    for r, c in best_full:
        out[r][c] = color
    return out
```

## Generator Constraints

ARC-GEN fixes `size=10`. The anchor is either `(row=random 0..9, col=0)` or `(row=0, col=random 0..9)`. Ring thickness is `1` or `2`, and the input shows rings through `show=2` or `show=3`; the output draws all rings in range. The color is one random non-gray color. The input background is black (`0`), while the output background is gray (`5`). Drawing is clipped to the 10x10 canvas by `common.draw`.

## Reference Notes

ARC-DSL identifies the least color, finds the largest least-color object, expands its outer boxes multiple times according to the inferred mat thickness, paints those boxes with the least color, then replaces black with gray. The generator is more explicit: the repeated pattern is a family of square frames/rings around an edge anchor, with spacing determined by `thick + 1`. The Code Golf 2025 solution recursively rotates/extends the pattern and fills remaining background with gray.
