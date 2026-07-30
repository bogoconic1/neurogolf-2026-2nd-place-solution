# task080 Semantics

## Sources

- Current champion builder: `solutions_py/task080.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task080.json`
- ARC-GEN task id: `39e1d7f9`
- ARC-DSL task id: `39e1d7f9`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_39e1d7f9.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_39e1d7f9.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task080.py`

## Pattern

The input is a line-grid rendering of a small bitmap. Several bitmap cells contain the same center color. One center cell is fully decorated: its four orthogonal neighbors have an edge color, and optionally its four diagonal neighbors have a corner color. The other center cells are present but undecorated. The output copies the complete decoration pattern from the example center to every center cell while preserving the line grid and all original pixels.

The bitmap size is 5..10, and the line-grid spacing is derived from that size, so the rendered grid can have periods 3, 4, or 5 in NeuroGolf's 30x30 canvas. The graph must infer the period/shape, center color, edge color, optional corner color, and all target center locations.

## Readable Python Solver

The previous cache implementation did not survive replay. The compact executable oracle below is the bundled Code Golf 2025 reference; the Pattern section remains the readable specification.

```python
import re

def p(g):
    l = sum(g, [])
    w = len(g)
    a, b, c, d = max((({*(q := (l[i:i + 3] + l[i - 2 * ~w::w * w])), 0}, i, q) for i in range(w * w)))[2]
    return [(g := eval(re.sub(f"(?={-~l.index(b) * x * '.'}{a})0", 'y', f'{(*zip(*g[::-1]),)}'))) for x, y in [(3, c)] * 4 + [(3 * w + 5, d)] * 4][7]


def solve(grid):
    result = p([list(row) for row in grid])
    return [list(row) for row in result]
```

## Generator Constraints

- `size` is randomly 5..10.
- The line-grid spacing is `6 - (size - 1) // 2`, producing period families handled locally as period 3, 4, or 5.
- There are `(size - 1) // 2` selected bitmap centers, each separated enough that 3x3 motifs do not overlap in bitmap coordinates.
- The first selected center is not on the bitmap border and is fully decorated in the input.
- Other selected centers have only the center color in the input; their decorations appear only in the output.
- There is always an orthogonal edge color. There may be a distinct corner color, or ARC-GEN may reuse the edge color as the corner color in two-color cases.
- The line color is distinct from the center/decorator colors and is the large grid-line structure.

## Reference Notes

ARC-DSL computes a prototype crop around the best center-color object, then shifts and paints that prototype onto all objects of the same center color. It chooses the exemplar by maximizing the palette size around an outbox, after removing the most common line/background color.

ARC-GEN first draws all centers, then only decorates the first center for the input. For the output it decorates every center. Code Golf performs a compact regex-like repeated substitution over rotated/transposed grid views, effectively copying the complete local pattern from the exemplar to all matching center occurrences.
