# task301 Semantics

## Sources

- Current champion builder: `solutions_py/task301.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task301.json`
- ARC-GEN task id: `beb8660c`
- ARC-DSL task id: `beb8660c`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_beb8660c.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_beb8660c.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task301.py`

## Pattern

The input contains `n` horizontal colored runs, where `n` is between `3` and `9`. The grid width is `n`; the height is `n + gap`, with `gap` in `0..3`. Each run has a unique length from `1` through `n`, appears on its own row at an arbitrary horizontal start position where it fits, and has one color. The longest run is cyan (`8`); other colors are random non-cyan colors.

The output keeps the same grid size, sorts the runs by length increasing from top to bottom after the empty gap rows, and right-aligns every run. The first `gap` rows are empty. Equivalently, sort each row's values so nonzero colors move to the right, then sort the rows lexicographically; because run lengths are unique, this orders empty rows first and then runs from shortest to longest.

## Readable Python Solver

```python
def solve(grid):
    return sorted(sorted(row) for row in grid)
```

A more explicit version:

```python
def solve(grid):
    h = len(grid)
    w = len(grid[0])
    runs = []
    for row in grid:
        vals = [v for v in row if v != 0]
        if vals:
            runs.append((len(vals), vals[0]))
    out = [[0 for _ in range(w)] for _ in range(h)]
    gap = h - len(runs)
    for i, (length, color) in enumerate(sorted(runs)):
        out[gap + i][w - length:w] = [color] * length
    return out
```

## Generator Constraints

- `num_colors` is sampled from `3..9`.
- Grid width is `num_colors`; height is `num_colors + gap`, where `gap` is sampled from `0..3`.
- There is exactly one run of every length `1..num_colors`.
- Run rows are sampled without replacement from `0..num_colors+gap-1`; the longest run is placed on the last row `num_colors+gap-1` in generated random cases.
- Run start columns satisfy `0 <= start <= width - length`.
- Colors are random from `1..9` excluding cyan, plus cyan (`8`) appended as the longest run color.
- There are no other nonzero objects.

## Reference Notes

The ARC-DSL solver extracts objects, orders them by decreasing size, normalizes them, shifts them into consecutive rows, paints a blank canvas, and rotates 180 degrees; this is equivalent to increasing length with right alignment. The Code Golf 2025 solution is simply `sorted(map(sorted, g))`, which relies on positive colors sorting after zeros and on unique run lengths.
