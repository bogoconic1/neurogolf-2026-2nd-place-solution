# task371 Semantics

## Sources

- Current champion builder: `solutions_py/task371.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task371.json`
- ARC-GEN task id: `e9614598`
- ARC-DSL task id: `e9614598`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_e9614598.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_e9614598.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task371.py`

## Pattern

The input contains exactly two blue (`1`) cells, either in the same row or in the same column. The output preserves the input and paints a green (`3`) plus centered at the midpoint between the two blue cells. The plus has five cells: the midpoint and its four direct neighbors. The blue markers remain blue. The grid size is unchanged.

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    pts = [(r, c) for r, row in enumerate(grid) for c, v in enumerate(row) if v == 1]
    (r0, c0), (r1, c1) = pts
    cr = (r0 + r1) // 2
    cc = (c0 + c1) // 2
    for dr, dc in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)):
        out[cr + dr][cc + dc] = 3
    return out
```

## Generator Constraints

ARC-GEN chooses width `2*randint(5, 7)` and height `2*randint(3, 7)`, then may transpose the whole task. The untransposed dots are on a single row at columns `col - space` and `col + space`, where `space` is between `3` and `(width - 1)//2`; therefore the marker distance is even and the midpoint is an integer. The center row is between `2` and `height - 3`, and the center column is chosen so both markers and the plus are in bounds. Colors are only black background (`0`), blue markers (`1`), and the green plus (`3`).

## Reference Notes

The ARC-DSL solver finds the two color-1 cells, averages their first and last coordinates, takes direct neighbors of the midpoint, inserts the center, and fills those cells with color `3`. The Code Golf 2025 solution expresses the same midpoint operation compactly over the grid or its transpose. There is no tie-breaking ambiguity because the generator always creates exactly two blue markers and an integral midpoint.
