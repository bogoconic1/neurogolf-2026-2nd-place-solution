# task357 Semantics

## Sources

- Current champion builder: `solutions_py/task357.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task357.json`
- ARC-GEN task id: `e179c5f4`
- ARC-DSL task id: `e179c5f4`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_e179c5f4.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_e179c5f4.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task357.py`

## Pattern

The input is a `10 x w` black grid, where `w` is 2..10, with a single blue (`1`) cell at the lower-left corner. The output has the same shape. Every background cell becomes cyan (`8`), and a single blue cell appears in each row. The blue cell follows a horizontal bounce/zigzag path from the bottom-left seed upward: moving up one row advances one column until the right edge, then reflects back left, repeating with period `2*w - 2`.

Equivalently, for output row `r` from top to bottom, the blue column is the reflected position of step `9-r` on a width-`w` line starting at column 0 on the bottom row.

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])
    period = 2 * w - 2
    out = []
    for r in range(h):
        step = h - 1 - r
        t = step % period
        c = min(t, period - t)
        row = [8] * w
        row[c] = 1
        out.append(row)
    return out
```

## Generator Constraints

- Height is always 10.
- Width is uniformly 2..10.
- Input has black background and one blue seed at row 9, column 0.
- Output uses only cyan background (`8`) and blue path (`1`).
- The bounce period is `2*w - 2`; widths 2 and 3 create short repeating cycles, while larger widths may not complete a full cycle within 10 rows.

## Reference Notes

The ARC-DSL solver shoots a diagonal up-right ray from the seed, reflects it from the right edge, repeats the cropped bounce tile, mirrors/crops it to the input height and width, and replaces remaining black with cyan. The Code Golf solution implements the same reflected modulo formula using `min(i, period-i)` for each row.
