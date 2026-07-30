# task226 Semantics

## Sources

- Current champion builder: `solutions_py/task226.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task226.json`
- ARC-GEN task id: `941d9a10`
- ARC-DSL task id: `941d9a10`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_941d9a10.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_941d9a10.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task226.py`

## Pattern

The input is a 10x10 black grid containing full gray separator rows and columns. The separators partition the grid into an odd number of row bands and an odd number of column bands. The output preserves the gray separator grid and colors three diagonal cells/blocks: the top-left band intersection becomes blue, the center band intersection becomes red, and the bottom-right band intersection becomes green. All other non-separator cells remain black.

## Readable Python Solver

The previous cache implementation did not survive replay. The compact executable oracle below is the bundled Code Golf 2025 reference; the Pattern section remains the readable specification.

```python
p = lambda g, i=0: [[-j + (j := (j + x)) or (1 + i * 2 / (C := sum(c))) * (i * 2 % C + i * sum(r) == j * C) for *c, x in zip(*g, r)] for r in g if [(i := (i + r[(j := 0)]))]]


def solve(grid):
    result = p([list(row) for row in grid])
    return [list(row) for row in result]
```

## Generator Constraints

ARC-GEN samples odd-length lists of row band heights and column band widths. Width bands are each 1..4 and height bands are each 1..3; the band sizes plus one-cell separators sum to exactly 10 in each dimension. The input has only black and gray. The output has gray separators plus blue in the top-left band intersection, red in the middle band intersection, and green in the bottom-right band intersection.

## Reference Notes

The ARC-DSL solver finds black connected components, then selects the component containing `(0,0)`, the component containing `(height-1,width-1)`, and the component containing `(5,5)`, filling them blue, green, and red respectively. The Code Golf solution expresses the same separator-count logic with running sums over rows and columns.
