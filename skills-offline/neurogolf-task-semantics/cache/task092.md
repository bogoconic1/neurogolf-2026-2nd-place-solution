# task092 Semantics

## Sources

- Current champion builder: `solutions_py/task092.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task092.json`
- ARC-GEN task id: `40853293`
- ARC-DSL task id: `40853293`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_40853293.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_40853293.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task092.py`

## Pattern

The input contains five colored endpoint pairs on a black grid. Each color is
unique and appears exactly twice. A pair either shares a row, defining a
horizontal stick, or shares a column, defining a vertical stick. The output
preserves the endpoints and fills the entire straight horizontal or vertical
segment between each same-color pair with that color.

## Readable Python Solver

The previous cache implementation did not survive replay. The compact executable oracle below is the bundled Code Golf 2025 reference; the Pattern section remains the readable specification.

```python
p = lambda g: [*map((F := (lambda *l, c=0: [l.count(e) > 1 and (c := (c ^ e)) or e for e in l])), *map(F, *g))]


def solve(grid):
    result = p([list(row) for row in grid])
    return [list(row) for row in result]
```

## Generator Constraints

ARC-GEN chooses width and height independently from `{10, 20, 30}`. It creates
exactly five sticks. Each stick is randomly horizontal or vertical, with unique
rows for horizontal sticks and unique columns for vertical sticks. Endpoints are
at least two cells apart. Horizontal endpoints do not lie on any vertical-stick
column, and vertical endpoints do not lie on any horizontal-stick row, avoiding
endpoint clobbering. Every stick has a unique nonzero color.

## Reference Notes

The ARC-DSL solver partitions colored objects, recolors each object's backdrop, keeps horizontal and vertical line backdrops, and paints them onto the input. The Code Golf solution applies a row pass and a column pass: in each line, when a nonzero color appears twice, fill the interval between its appearances. Both references confirm that matching color and shared row/column are the only relationship needed; there are no diagonal or bent paths.
