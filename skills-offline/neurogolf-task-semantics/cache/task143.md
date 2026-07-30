# task143 Semantics

## Sources

- Current champion builder: `solutions_py/task143.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task143.json`
- ARC-GEN task id: `63613498`
- ARC-DSL task id: `63613498`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_63613498.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_63613498.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task143.py`

## Pattern

The grid is 10x10. A fixed gray 4-cell corner frame marks the top-left 4x4 reference area: row 3 columns 0..3 and column 3 rows 0..3 are gray. Inside the top-left 3x3 area is a colored reference sprite. Elsewhere there are 3 to 5 non-overlapping 3x3 creature sprites in distinct colors. Exactly one external sprite has the same normalized shape as the top-left reference sprite. The output paints that matching external sprite gray and paints the original 3x3 reference sprite back into the top-left reference area with its boxed color. Other cells remain as in the input.

## Readable Python Solver

The previous cache implementation did not survive replay. The compact executable oracle below is the bundled Code Golf 2025 reference; the Pattern section remains the readable specification.

```python
def p(g, i=1):
    *R, = G = b'%r' % g
    for k in b'\x02\x05\x08"%(BEH':
        R[k + i] = (48 % G[k] or -5) % G[k + i] + 5
    return {*G} > {*R} and eval(bytes(R)) or p(g, i + 1)


def solve(grid):
    result = p([list(row) for row in grid])
    return [list(row) for row in result]
```

## Generator Constraints

The size is fixed at 10. The number of external sprites is 3 to 5. Every sprite is a connected creature of size 3 or 4 contained in a 3x3 box, and duplicate shapes are rejected during generation. External 3x3 boxes do not overlap and none starts in the top-left 5x5 region. The first generated external sprite is the one whose normalized shape is copied into the top-left reference area. The reference color bcolor excludes gray; external sprite colors exclude bcolor and gray. The matching external sprite is recolored gray in the output, while the top-left reference sprite remains bcolor.

## Reference Notes

ARC-DSL crops the top-left 3x3, computes its nonzero normalized shape, filters objects in the full grid whose normalized indices match that shape, fills those matching object cells with gray, and repaints the top-left reference object. Code Golf implements the same search by repeatedly trying positions until replacing the matching external sprite with gray and restoring the reference makes a changed grid.
