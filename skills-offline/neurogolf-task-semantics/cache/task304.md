# task304 Semantics

## Sources

- Current champion builder: `solutions_py/task304.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task304.json`
- ARC-GEN task id: `c3e719e8`
- ARC-DSL task id: `c3e719e8`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_c3e719e8.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_c3e719e8.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task304.py`

## Pattern

The input is a `3x3` grid using 3 or 4 colors. One color is the unique most frequent color. The output is a `9x9` block grid: for every input cell whose color is the most frequent color, copy the entire `3x3` input pattern into the corresponding `3x3` output block. All blocks whose input cell is not the most frequent color remain black.

Equivalently, upscale the input by factor 3 as a mask of winner-color cells, and within every winner-color block place the original 3x3 input.

## Readable Python Solver

```python
def solve(grid):
    from collections import Counter

    flat = [v for row in grid for v in row]
    winner = Counter(flat).most_common(1)[0][0]
    out = [[0 for _ in range(9)] for _ in range(9)]
    for br in range(3):
        for bc in range(3):
            if grid[br][bc] != winner:
                continue
            for r in range(3):
                for c in range(3):
                    out[br * 3 + r][bc * 3 + c] = grid[r][c]
    return out
```

## Generator Constraints

- Input size is fixed at `3x3`; output size is fixed at `9x9`.
- The color list has 3 or 4 colors.
- `common.square_with_unique_max_color` guarantees a unique most frequent color in the 3x3 input.
- Output blocks are black unless the corresponding input cell is the unique max color.

## Reference Notes

The ARC-DSL solver finds `mostcolor(I)`, constructs a 3x upscaled mask of that color, tiles the original 3x3 input into a 9x9 canvas, and zero-fills cells outside the mask. The Code Golf solution performs the same nested loop: for each output block, use the input pattern only when the block's source cell equals the mode.
