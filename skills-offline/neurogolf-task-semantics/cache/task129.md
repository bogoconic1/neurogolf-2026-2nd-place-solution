# task129 Semantics

## Sources

- Current champion builder: `solutions_py/task129.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task129.json`
- ARC-GEN task id: `5582e5ca`
- ARC-DSL task id: `5582e5ca`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_5582e5ca.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_5582e5ca.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task129.py`

## Pattern

The input is a `3 x 3` grid containing several colors. The output is a `3 x 3` grid filled entirely with the most frequent color in the input.

ARC-GEN makes the majority unambiguous: one sampled color appears exactly three times, another appears twice, one or two colors may appear twice depending on whether the final sampled pair is duplicated, and the remaining colors appear once. The most frequent color is therefore the color placed in the first three positions.

## Readable Python Solver

```python
def solve(grid):
    counts = {}
    for row in grid:
        for color in row:
            counts[color] = counts.get(color, 0) + 1
    majority = max(counts, key=counts.get)
    return [[majority for _ in range(3)] for _ in range(3)]
```

## Generator Constraints

The grid size is fixed at `3 x 3`. ARC-GEN samples six colors from `0..9`; positions are a random permutation of all nine cells. The first color is written to three cells, the second to two cells, colors three and four to one cell each, and the fifth/sixth colors to the last two cells, with the sixth equal to the fifth about 75% of the time. This never creates a tie with the first color.

## Reference Notes

The ARC-DSL solver is exactly `canvas(mostcolor(I), THREE_BY_THREE)`. The Code Golf solution flattens repeated rows and selects the value with maximum count, then fills a `3 x 3` grid with it.
