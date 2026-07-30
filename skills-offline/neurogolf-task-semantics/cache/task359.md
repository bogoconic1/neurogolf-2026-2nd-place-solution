# task359 Semantics

## Sources

- Current champion builder: `solutions_py/task359.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task359.json`
- ARC-GEN task id: `e26a3af2`
- ARC-DSL task id: `e26a3af2`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_e26a3af2.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_e26a3af2.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task359.py`

## Pattern

The input is a rectangular grid made from broad same-color stripes, then corrupted by sparse random single-cell noise. Stripes may be vertical in the displayed input or horizontal if the generator transposes the grid. The output is the clean stripe background with the same shape as the input: every noisy pixel is replaced by its stripe color.

For each cell, the correct color is the dominant color supported by that cell's row/column stripe structure. The Code Golf solution uses the most common value in the concatenation of the cell's row and column, which is enough because the true stripe color dominates the sparse noise in at least one direction.

## Readable Python Solver

```python
from collections import Counter

def solve(grid):
    cols = list(zip(*grid))
    out = []
    for r, row in enumerate(grid):
        out_row = []
        for c, col in enumerate(cols):
            counts = Counter(row)
            counts.update(col)
            out_row.append(max(counts, key=counts.get))
        out.append(out_row)
    return out
```

A stripe-specific equivalent is: compute the most common color of each row and each column, decide whether the repeated-color sequence varies by rows or columns, then broadcast that sequence across the opposite axis.

## Generator Constraints

- Base, before optional transpose: height is 13..15.
- Width is the sum of 3..5 stripe widths, each width 2..6.
- Each stripe receives a random color; repeated stripe colors are allowed.
- Noise pixels are sampled over the whole grid with probability about 0.1 and assigned random colors.
- `xpose` randomly transposes both input and output, so the final grid can have vertical or horizontal stripes.
- Output dimensions exactly match input dimensions and contain only the clean stripe colors.

## Reference Notes

The ARC-DSL solver computes row modes and column modes, deduplicates those mode sequences to infer orientation, then repeats the chosen mode sequence vertically or horizontally to reconstruct the clean stripes. The Code Golf solution avoids an explicit orientation branch by taking the most common color over `row + column` for each cell. Both agree that the task is not object-based; it is row/column statistical denoising under sparse noise.
