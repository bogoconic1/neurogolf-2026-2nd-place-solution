# task265 Semantics

## Sources

- Current champion builder: `solutions_py/task265.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task265.json`
- ARC-GEN task id: `a8d7556c`
- ARC-DSL task id: `a8d7556c`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_a8d7556c.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_a8d7556c.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task265.py`

## Pattern

The input is a fixed `18x18` grid of black (`0`) and gray (`5`) static. Several black holes are inserted as `2x2` all-black patches. The output paints red (`2`) over cells belonging to fillable all-black `2x2` holes. The generator applies two stripe passes: first all vertically constrained/downstripe empty `2x2` blocks, then all horizontally constrained/sidestripe empty `2x2` blocks, and it only keeps generated examples where reversing the pass order gives the same output.

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])
    out = [row[:] for row in grid]

    def is_empty(r, c):
        return all(out[r + dr][c + dc] == 0 for dr in (0, 1) for dc in (0, 1))

    def paint(cells):
        for r, c in cells:
            for dr in (0, 1):
                for dc in (0, 1):
                    out[r + dr][c + dc] = 2

    for stripe in (0, 1):
        todo = []
        for r in range(h - 1):
            for c in range(w - 1):
                if not is_empty(r, c):
                    continue
                if stripe == 0:
                    if c > 0 and out[r][c - 1] == 0 and out[r + 1][c - 1] == 0:
                        continue
                    if c < w - 2 and out[r][c + 2] == 0 and out[r + 1][c + 2] == 0:
                        continue
                else:
                    if r > 0 and out[r - 1][c] == 0 and out[r - 1][c + 1] == 0:
                        continue
                    if r < h - 2 and out[r + 2][c] == 0 and out[r + 2][c + 1] == 0:
                        continue
                todo.append((r, c))
        paint(todo)
    return out
```

## Generator Constraints

The grid is always `18x18`. Inputs contain only black and gray. The random generator starts from dense gray static with about 80% gray pixels, then inserts `4..6` black `2x2` holes at random top-left coordinates. It rejects cases where the vertical-first and horizontal-first stripe passes disagree, so accepted examples are unambiguous under the stripe-fill rule.

## Reference Notes

ARC-DSL approximates the rule by finding occurrences of a `2x2` black object and painting their cells red, with a fixed two-cell correction around coordinate `(8, 12)` for the benchmark examples. The code-golf solution uses a large regex over the printed grid to identify the same local `2x2`-hole pattern.
