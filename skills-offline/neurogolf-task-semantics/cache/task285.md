# task285 Semantics

## Sources

- Current champion builder: `solutions_py/task285.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task285.json`
- ARC-GEN task id: `b775ac94`
- ARC-DSL task id: `b775ac94`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_b775ac94.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_b775ac94.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task285.py`

## Pattern

The grid contains one to three separated colored sprite objects. Each sprite is generated from a small connected pixel set in one quadrant around an implicit center. The input shows one full quadrant/orientation of each sprite, plus the corner/anchor pixel for the other nonzero orientations. The output completes the missing mirrored orientations around the same center. Each orientation has its own color; a zero color means that orientation is absent and should not be drawn.

## Readable Python Solver

The previous cache implementation did not survive replay. The compact executable oracle below is the bundled Code Golf 2025 reference; the Pattern section remains the readable specification.

```python
def p(h):
    for e in range(8):
        h[::-1] = zip(*h)
        for e in range(len(h)):
            for i in range(len(h)):
                for q, g in (f := [(e, i)]):
                    *h[q], = h[q]
                    h[q][i + i - g - 1] = h[e][i - 1]
                    f += [(p + q, g + n) for p in range(-1, 2) for n in range(-1, 2) if 0 < h[e][i - 1] != (2 * (2 * h)[p + q])[g + n] == h[e][i] > 0 == h[p + q][i + i - g - n - 1]]
    return h


def solve(grid):
    result = p([list(row) for row in grid])
    return [list(row) for row in result]
```

## Generator Constraints

Grid size is square, usually 12 through 30. There are 1 to 3 sprites. Each sprite has 4 to 8 connected source pixels in a 5x5 local coordinate frame, and sprite centers are chosen far enough from the border that all four mirrored orientations fit. Sprites are rejected if any output pixels overlap or become adjacent to another sprite. Four distinct random colors are chosen per sprite, then one non-shown orientation may be set to zero. The input displays the selected orientation for every source pixel and displays the source anchor `(row=0, col=0)` in every nonzero orientation, which exposes the orientation colors.

## Reference Notes

ARC-DSL groups nonzero objects, separates the most-color visible body from the anchor/corner cells, infers the relative position of the body and anchors, creates horizontal, vertical, and both-axis mirrored copies, samples the color at the intersecting anchor location for each copy, and paints those recolored copies back into the grid. The Code Golf solution repeatedly rotates/mirrors the grid and propagates matching colored neighbors to fill the hidden orientations.
