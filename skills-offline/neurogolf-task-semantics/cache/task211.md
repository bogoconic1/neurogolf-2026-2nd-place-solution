# task211 Semantics

## Sources

- Current champion builder: `solutions_py/task211.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task211.json`
- ARC-GEN task id: `8d5021e8`
- ARC-DSL task id: `8d5021e8`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_8d5021e8.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_8d5021e8.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task211.py`

## Pattern

The input is a fixed `3x2` grid containing one nonzero color. Any nonempty subset of the six cells may be colored with that same color; all other cells are black.

The output is a fixed `9x4` tile made by horizontally mirroring every row and vertically repeating a mirrored/original row pattern. Equivalently, for input rows `g[0], g[1], g[2]`, the output rows use source row order:

```text
2, 1, 0, 0, 1, 2, 2, 1, 0
```

and each output row is `input_row[::-1] + input_row`, so source columns are selected in order:

```text
1, 0, 0, 1
```

The result is placed in the top-left of the standard dense NeuroGolf output tensor.

## Readable Python Solver

```python
def solve(grid):
    rows = list(reversed(grid)) + list(grid)
    rows = (rows * 2)[:9]
    return [list(reversed(row)) + list(row) for row in rows]
```

## Generator Constraints

- Input width is fixed at `2`; input height is fixed at `3`.
- At least one input cell is colored.
- All colored cells use one randomly chosen color from the ARC palette; there are no multiple foreground colors in one example.
- Output width is `4`; output height is `9`.
- The standard NeuroGolf graph still emits dense `[1,10,30,30]`, with the real output occupying rows `0..8` and columns `0..3`.

## Reference Notes

The ARC-DSL solver performs `vmirror(I)`, horizontally concatenates the mirror with the original input, mirrors that 3x4 tile vertically, concatenates it under the first tile, appends the first tile again, and finally applies a horizontal mirror. The Code Golf 2025 solution expresses the same mapping directly as `[r[::-1]+r for r in (g[::-1]+g)*2][:9]`.
