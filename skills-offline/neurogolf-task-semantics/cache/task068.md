# task068 Semantics

## Sources

- Current champion builder: `solutions_py/task068.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task068.json`
- ARC-GEN task id: `31aa019c`
- ARC-DSL task id: `31aa019c`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_31aa019c.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_31aa019c.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task068.py`

## Pattern

The input is a 10x10 sparse grid with several colors. Exactly one color appears once; every other present color appears multiple times. The output is a 10x10 black canvas with the unique-color pixel copied at its original coordinate and all eight neighboring cells around it painted red (`2`).

## Readable Python Solver

```python
def solve(grid):
    vals = [v for row in grid for v in row if v != 0]
    color = min(set(vals), key=vals.count)
    r = c = None
    for i, row in enumerate(grid):
        for j, value in enumerate(row):
            if value == color:
                r, c = i, j
                break
        if r is not None:
            break
    out = [[0 for _ in range(10)] for _ in range(10)]
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            out[r + dr][c + dc] = 2
    out[r][c] = color
    return out
```

## Generator Constraints

- Grid size is fixed at `10x10`.
- The unique color's coordinate is sampled away from the border: row and column are in `1..8`, so the 3x3 halo is always inside the grid.
- A random set of 6 to 9 colors is used. The first color has exactly one pixel; every other color has 2 to 7 pixels.
- Nonzero distractor pixels never overwrite an occupied cell.

## Reference Notes

The ARC-DSL solver finds the least frequent color, takes its only coordinate, fills that coordinate with the original color on a blank 10x10 canvas, then fills its Moore neighbors red. The Code Golf solution encodes the same least-frequency and 3x3-neighborhood rule.
