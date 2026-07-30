# task096 Semantics

## Sources

- Current champion builder: `solutions_py/task096.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task096.json`
- ARC-GEN task id: `4290ef0e`
- ARC-DSL task id: `4290ef0e`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_4290ef0e.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_4290ef0e.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task096.py`

## Pattern

The input is a 13..19 by 13..19 grid with a random background color and 4..6
foreground colors. Each foreground color represents one radius of a centered
canonical motif. In the input, that color appears as one or more partial L-shaped
corner fragments at offsets `(+-radius, +-radius)` around some hidden center;
fragments may be clipped by the input boundary, but the generator requires at
least two quadrants to leave visible evidence for each nonzero radius. The
output is a compact square of size `2*n-1`, where `n` is the number of foreground
colors/radii. It uses the input background color and redraws the inferred L
corner fragment for each color in all four quadrants around the center, producing
the canonical completed motif.

## Readable Python Solver

The previous cache implementation did not survive replay. The compact executable oracle below is the bundled Code Golf 2025 reference; the Pattern section remains the readable specification.

```python
import re

def p(a):
    s = re.sub(', ', '', str(a + [*zip(*a)]))
    f = int(max(s, key=s.count))
    r = {0: (0, f)}
    for l in range(10):
        if (l != f) * re.findall(f'{l}', s):
            e = len(max(re.findall(f'{l}{l}([^]){l}]+){l}|', s + s[::-1])))
            r[len(max(re.findall(f'{l}+', s))) * ((e > 0) + 1) + e >> 1] = (1 + e >> 1, l)
    return [[[r[max(abs(l), abs(s))][1], f][r[max(abs(l), abs(s))][0] > min(abs(l), abs(s))] for l in range(-max(r), max(r) + 1)] for s in range(-max(r), max(r) + 1)]


def solve(grid):
    result = p([list(row) for row in grid])
    return [list(row) for row in result]
```

## Generator Constraints

Input width and height are independently sampled from `13..19`. The background
color is random, and `4..6` foreground colors are sampled excluding the
background. For foreground index `i`, the corner radius is `i`; arm length is
random with bounds `min(i+1,2)..i` for `i>1` and special small bounds for the
first colors. The generator repeatedly samples hidden centers until fragments
do not overlap other colors/background constraints and every nonzero radius has
visible evidence in at least two quadrants. Output size is always
`2*len(colors)-1`, so it ranges from 7x7 to 11x11.

## Reference Notes

The ARC-DSL solver partitions foreground objects, orders colors by a derived measure combining object width, pairwise Manhattan distances, and run/radius evidence, normalizes each color's object under the four mirrors, then paints the normalized fragments into a background canvas and rotates/paints them into all four quadrants. The Code Golf solution uses regex over rows plus transposed rows to recover the same background, ring order, color, and arm-length map.
