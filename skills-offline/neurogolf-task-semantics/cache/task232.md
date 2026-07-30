# task232 Semantics

## Sources

- Current champion builder: `solutions_py/task232.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task232.json`
- ARC-GEN task id: `97999447`
- ARC-DSL task id: `97999447`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_97999447.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_97999447.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task232.py`

## Pattern

The input is a black grid containing one to four isolated non-gray colored seed cells, with at most one seed in any row. The output keeps every non-seed row black. For a row with seed color `color` at column `col`, columns before `col` stay black; columns from `col` through the true grid width are filled as a horizontal ray that alternates `color, 5, color, 5, ...`, starting with `color` at the seed column. Cells outside the original grid are zero-hot padding in NeuroGolf and must remain non-positive in all channels.

## Readable Python Solver

```python
def solve(grid):
    h = len(grid)
    w = len(grid[0])
    out = [[0 for _ in range(w)] for _ in range(h)]
    for r, row in enumerate(grid):
        for c, color in enumerate(row):
            if color == 0 or color == 5:
                continue
            for cc in range(c, w):
                out[r][cc] = color if (cc - c) % 2 == 0 else 5
    return out
```

## Generator Constraints

- Width and height are each sampled from `7..14`.
- The number of seeds is sampled from `1..4`.
- Seed colors are distinct random colors excluding gray `5`.
- Seed rows are sampled without replacement, so there is at most one active seed per row.
- Seed columns are independently sampled from `0..width//2`.
- The output has the same height and width as the input. The ray always runs to the true right edge; padded columns beyond the grid must stay blank.

## Reference Notes

- ARC-GEN gives the fresh-sample rule: fill every seeded row to the right with alternating seed color and gray.
- The Code Golf 2025 solution uses a row accumulator modulo `sum(row) + 5`, which is a compact way to start the alternating sequence when the seed is encountered and leave all-zero rows unchanged.
- The ARC-DSL solver paints colored rightward rays and overlays gray at odd horizontal offsets. It reflects the same parity rule, but its explicit odd offsets are less general than ARC-GEN's full width range, so ARC-GEN is the constraint source for generated validation.
