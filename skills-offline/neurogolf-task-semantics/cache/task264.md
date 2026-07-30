# task264 Semantics

## Sources

- Current champion builder: `solutions_py/task264.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task264.json`
- ARC-GEN task id: `a8c38be5`
- ARC-DSL task id: `a8c38be5`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_a8c38be5.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_a8c38be5.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task264.py`

## Pattern

The input contains nine non-overlapping `3x3` gray sprite blocks scattered in a `14..16` by `14..16` black canvas. Each sprite has a fixed shape associated with one of the nine positions in a `3x3` output catalog, and its colored cells use an arbitrary non-gray color. The output is always a `9x9` gray canvas divided into nine `3x3` cells. Each detected input sprite is normalized to its `3x3` local coordinates and painted into the matching output catalog cell using its own color.

## Readable Python Solver

The previous cache implementation did not survive replay. The compact executable oracle below is the bundled Code Golf 2025 reference; the Pattern section remains the readable specification.

```python
T = (3, 2, 1)
p = lambda g: [sum([eval(f"sorted([0in(S:=sum(g,T)),*[i!=5for i in S],*g]{'for*g,in map(zip,g,g[1:],g[2:])' * 2})#{g}")[42 >> Y & 7 * ~-X ^ Y][-x] for Y in T], ()) for X in T for x in T]


def solve(grid):
    result = p([list(row) for row in grid])
    return [list(row) for row in result]
```

## Generator Constraints

There are exactly nine sprites. Each sprite occupies a gray `3x3` block and then paints one of nine fixed colored patterns inside it. The sprite blocks are non-overlapping with at least one-cell separation. Input width and height are independently `14..16`. Sprite colors are random non-gray colors, so the same color can appear in multiple sprites, but shape identifies the catalog position.

## Reference Notes

ARC-DSL removes gray, extracts connected colored objects, normalizes their coordinates, matches those normalized shapes against a synthetic `9x9` catalog of the nine patterns, and shifts each object into its matched output position. The code-golf solution performs an equivalent pattern extraction over sliding `3x3` windows. The center catalog cell has no colored pattern in the generator, so the output center `3x3` remains gray.
