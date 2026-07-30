# task060 Semantics

## Sources

- Current champion builder: `solutions_py/task060.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task060.json`
- ARC-GEN task id: `29c11459`
- ARC-DSL task id: `29c11459`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_29c11459.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_29c11459.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task060.py`

## Pattern

The input is a fixed 5x11 black grid. One or two rows contain a colored pixel at the left endpoint column `0` and a different colored pixel at the right endpoint column `10`. The output expands each marked row into a horizontal band: columns `0..4` take the left endpoint color, column `5` is gray (`5`), and columns `6..10` take the right endpoint color. Unmarked rows remain black.

## Readable Python Solver

```python
def solve(grid):
    out = [[0 for _ in range(11)] for _ in range(5)]
    for r, row in enumerate(grid):
        left, right = row[0], row[10]
        if left or right:
            for c in range(5):
                out[r][c] = left
                out[r][10 - c] = right
            out[r][5] = 5
    return out
```

## Generator Constraints

The generator fixes `height=5` and `width=11`. It samples one or two distinct rows. For each sampled row it chooses distinct colors from `1..9` excluding gray (`5`), assigning one to the left endpoint and one to the right endpoint. All other cells are black. The output shape is unchanged before NeuroGolf padding.

## Reference Notes

The ARC-DSL solution splits the grid into left and right halves, turns each endpoint object into a horizontal frontier of its color, paints both halves, and fills the center column beside the left object with gray. The Code Golf solution is the same rule in compact form: for each row, repeat the left endpoint five times, place gray when the row has a right endpoint, then repeat the right endpoint five times.
