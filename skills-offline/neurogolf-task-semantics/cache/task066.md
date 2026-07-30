# task066 Semantics

## Sources

- Current champion builder: `solutions_py/task066.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task066.json`
- ARC-GEN task id: `2dd70a9a`
- ARC-DSL task id: `2dd70a9a`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_2dd70a9a.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_2dd70a9a.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task066.py`

## Pattern

The input is a square grid containing a partially hidden one-cell-wide orthogonal path. The visible endpoint structure has a red adjacent pair at one end and a green adjacent pair at the other end. Cyan cells are visible blockers, turn markers, and random distractor noise. The true path cells between the endpoint structure are black in the input. The output preserves all visible non-black cells and paints the hidden black path cells green, connecting the green endpoint side to the red endpoint side through the intended S-shaped or U-shaped corridor. The generator may flip and/or transpose the whole grid, so the same rule must handle both orientations.

## Readable Python Solver

The previous cache implementation did not survive replay. The compact executable oracle below is the bundled Code Golf 2025 reference; the Pattern section remains the readable specification.

```python
def p(e):

    def p(e, o, n, i, r, l=5):
        e = [o * 1 for o in e]
        while 0 < o < len(e) - 1 > n > 0 == (d := e[o + i][n + r]):
            o += i
            n += r
            e[o][n] = 3
        return p(e, o, n, r, i, l + 1) or p(e, o, n, -r, -i, l + 1) if d > 7 > l else (d == 2) * e
    return max((p(e, o, n, i, r) for o in range(len(e)) for n in range(len(e)) for i in range(-1, 2) for r in range(-1, 2) if 3 == e[o][n] == e[o][n - r]))


def solve(grid):
    result = p([list(row) for row in grid])
    return [list(row) for row in result]
```

## Generator Constraints

- Grid size is sampled from `10..20`, square.
- There are two structural families before optional flip/transpose: an S-like path and a U-like path.
- The hidden path is generated as blue cells, but blue cells are written as black in the input and green in the output.
- Red endpoint cells mark one side of the hidden path and green endpoint cells mark the other side. Cyan cells mark turns/blockers, and additional random cyan distractors may be placed before the true path overwrites them.
- The generator may flip vertically and/or transpose the final input/output pair.
- Output size is the same as input size, with all original visible colors preserved except hidden black path cells become green.

## Reference Notes

The ARC-DSL solver explicitly reasons about red and green endpoint orientation, shoots tentative rays from a selected green endpoint, removes the wrong adjacent spur, finds the farthest temporary path cell, and connects that cell to the red endpoint ray before replacing temporary path color with green. The Code Golf solution reveals a compact equivalent: recursively walk from a green pair through black cells, turn at cyan blockers, and stop when red is reached. Both references agree that cyan is a structural turn/blocker signal but may also appear as distractor noise.
