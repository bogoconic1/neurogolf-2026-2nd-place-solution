# task026 Semantics

## Sources

- Current champion builder: `solutions_py/task026.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task026.json`
- ARC-GEN task id: `1b2d62fb`
- ARC-DSL task id: `1b2d62fb`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_1b2d62fb.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_1b2d62fb.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task026.py`

## Pattern

The input is a fixed 5 row by 7 column grid made from two 5x3 panels separated by a vertical blue divider in column 3. The left and right panels contain only black (`0`) and maroon (`9`) cells. The output is a 5x3 grid. A cell in the output is cyan (`8`) exactly when the corresponding cell in both the left and right input panels is black. Every other output cell is black.

Equivalently, split the input into `left = grid[:, 0:3]` and `right = grid[:, 4:7]`. The output is the elementwise intersection of black cells in the two panels, drawn as cyan on a black background.

## Readable Python Solver

```python
def solve(grid):
    h = len(grid)
    out = [[0 for _ in range(3)] for _ in range(h)]
    for r in range(h):
        for c in range(3):
            if grid[r][c] == 0 and grid[r][c + 4] == 0:
                out[r][c] = 8
    return out
```

## Generator Constraints

- Active input size is fixed at height `5`, width `7`.
- The left panel is columns `0..2`; the divider is column `3`; the right panel is columns `4..6`.
- The full input starts as maroon (`9`) in both panels; sampled pixels are changed to black (`0`). The divider column is always blue (`1`) after pixel placement.
- The output size is fixed at `5x3` and starts black (`0`). It is filled cyan (`8`) only at positions where both corresponding panel cells are black.
- ARC-GEN samples arbitrary black pixels over the 5x7 input before the divider overwrite, so black pixels sampled in divider column do not affect the final input or output.

## Reference Notes

- ARC-DSL computes `lefthalf(I)` and `righthalf(I)`, finds black cells in each half, intersects those coordinate sets, replaces maroon in the left half with black, then fills cyan at the intersection. Since the output background is all black, this is just the black-cell intersection.
- The Code Golf 2025 solution is a compact row expression equivalent to checking paired cells across the divider and emitting color 8 on the intersection.
- The blue divider is only structural and never appears in the output.
