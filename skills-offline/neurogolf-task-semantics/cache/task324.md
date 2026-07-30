# task324 Semantics

## Sources

- Current champion builder: `solutions_py/task324.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task324.json`
- ARC-GEN task id: `d07ae81c`
- ARC-DSL task id: `d07ae81c`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_d07ae81c.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_d07ae81c.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task324.py`

## Pattern

The grid has two background colors arranged as a base color plus horizontal and vertical stripe bands. It also has two dot/diagonal colors. The input contains two or three isolated colored dots. Each dot sits on one of the two background colors, and the legal generator requires that the chosen dots collectively see both background colors.

The output keeps the original grid and draws both diagonals through every dot: the `r-c` diagonal and the `r+c` anti-diagonal. Cells on those diagonals are recolored according to the underlying background at that cell. One output diagonal color is used over the base background color and the other output diagonal color is used over the stripe background color. The dot colors themselves determine which foreground color corresponds to which background by inspecting the neighborhood around a singleton dot, matching the ARC-DSL branch logic.

## Readable Python Solver

The previous cache implementation did not survive replay. The compact executable oracle below is the bundled Code Golf 2025 reference; the Pattern section remains the readable specification.

```python
def p(m):
    s = sum(m, (t := []))
    r = sorted(s, key=s.count)[:2]
    for d in range(len(m)):
        for o in range(len(m[d])):
            if m[d][o] in r:
                s[m[d][o]] = s[max(s, key=[m[d][o - 1], m[d][o - len(m[d]) + 1], m[d < 1][o]].count)] = m[d][o]
                r += (-d - o,)
                t += (d - o,)
    for d in range(len(m)):
        for o in range(len(m[d])):
            if d - o in t or -d - o in r:
                m[d][o] = s[m[d][o]]
    return m


def solve(grid):
    result = p([list(row) for row in grid])
    return [list(row) for row in result]
```

## Generator Constraints

ARC-GEN chooses width and height independently from `10..20`. It samples two background colors and two foreground/dot colors. The second background color is drawn as several horizontal stripe bands and vertical stripe bands with spacing gaps. It then chooses two or three dot locations after removing diagonal neighbors, so dots do not share immediate diagonal adjacency. The draw loop rejects samples unless the dot locations land on both background colors, which lets the output-color/background-color pairing be inferred.

For every dot, the output draws both infinite diagonals clipped to the grid. If a diagonal cell is on the base background, it receives one foreground color; if it is on the stripe background, it receives the other foreground color. The input dots are also colored using the same background-dependent rule.

## Reference Notes

The ARC-DSL solver identifies singleton objects as the dots and non-singleton objects as the stripe/background regions. It computes both diagonal directions through each dot with `shoot` in `UNITY`, `NEG_UNITY`, `DOWN_LEFT`, and `UP_RIGHT`, intersects those rays with each background color, infers the rare color/background pairing from the neighbors of one singleton dot, and fills the two diagonal cell sets with the corresponding colors.

The Code Golf solution takes a frequency-based view: rare colors are the dot colors, frequent colors are backgrounds, then diagonals through the rare-colored dots are recolored according to the inferred surrounding/background color. The references agree on the transformation.
