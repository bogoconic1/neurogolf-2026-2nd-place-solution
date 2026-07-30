# task001 Semantics

## Sources

- Current champion builder: `solutions_py/task001.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task001.json`
- ARC-GEN task id: `007bbfb7`
- ARC-DSL task id: `007bbfb7`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_007bbfb7.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_007bbfb7.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task001.py`

## Pattern

The input is a fixed 3x3 grid containing zeros and one nonzero color. The
output is 9x9 and can be viewed as a 3x3 grid of 3x3 blocks. For every nonzero
cell in the input, the corresponding output block is a copy of the entire input
grid; for every zero input cell, the corresponding output block is all zeros.
Equivalently, the output is the elementwise product of a 3x upscaled foreground
mask and a 3x3 tiling of the input pattern. Because all foreground pixels share
one color, this is also a Kronecker-style self-product of the nonzero mask with
the original colored pattern.

## Readable Python Solver

```python
def solve(grid):
    h = len(grid)
    w = len(grid[0])
    out = [[0 for _ in range(w * w)] for _ in range(h * h)]
    for br in range(h):
        for bc in range(w):
            if grid[br][bc] == 0:
                continue
            for r in range(h):
                for c in range(w):
                    out[br * h + r][bc * w + c] = grid[r][c]
    return out
```

## Generator Constraints

ARC-GEN uses `size=3`, so inputs are 3x3 and outputs are 9x9. It samples
between 2 and 8 distinct foreground pixels from the 3x3 grid and paints every
sampled pixel with one random nonzero ARC color. There is exactly one foreground
color per generated instance. At least one zero remains because the sample count
is at most 8, and at least two colored cells are present. The block rule works
for any 3x3 one-color mask, including sparse, dense, disconnected, edge, and
corner cases.

## Reference Notes

The ARC-DSL solution horizontally and vertically upscales the input by 3, separately tiles the input 3x3, then uses `cellwise(..., ZERO)` to keep only positions where both tensors are nonzero. This matches the generator's nested loop over every pair of foreground cells. The Code Golf 2025 solution expresses the same operation as a compact nested list over rows and columns of the first and last grid arguments. There is no disagreement between the references.
