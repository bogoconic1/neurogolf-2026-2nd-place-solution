# task294 Semantics

## Sources

- Current champion builder: `solutions_py/task294.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task294.json`
- ARC-GEN task id: `bb43febb`
- ARC-DSL task id: `bb43febb`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_bb43febb.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_bb43febb.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task294.py`

## Pattern

The input is a 10x10 grid with two solid gray (`5`) rectangles on black
background. The rectangles are stacked in one of several vertical layouts and
then the whole grid may be gravity-rotated/flipped by the generator. The output
keeps the gray one-cell border of each rectangle and recolors every strict
interior gray cell to red (`2`). Background remains black.

Equivalently, for each cell: if it is gray and its north, south, east, and west
neighbors are also gray, output red; otherwise keep the input color.

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    for r in range(h):
        for c in range(w):
            if grid[r][c] != 5:
                continue
            if r == 0 or c == 0 or r == h - 1 or c == w - 1:
                continue
            if (grid[r - 1][c] == 5 and grid[r + 1][c] == 5 and
                    grid[r][c - 1] == 5 and grid[r][c + 1] == 5):
                out[r][c] = 2
    return out
```

## Generator Constraints

ARC-GEN creates two gray rectangles on a 10x10 background. Rectangle lengths are
random integers from 4 through 8, columns are independently chosen so each
rectangle fits, and heights are `[3, 5]` or `[3, 6]` depending on a layout mode.
The mode controls whether the stack has top gap, bottom gap, or no internal
space. A gravity value in `0..3` transforms both input and output together, so
interiors can appear under any orientation.

Important edge cases:

- rectangles can touch the raw grid border after gravity
- interiors are based on 4-neighbor gray support, not diagonal support
- background cells next to rectangles must stay background
- output is still a full dense NeuroGolf `[1,10,30,30]` tensor, with the Conv
  graph output carrying the complete 10-channel logits

## Reference Notes

ARC-DSL identifies gray objects, takes the `inbox` of each object, then fills those interior cells with red. The Code Golf solution uses a regex over the string form of the grid to find gray cells with gray cells in the same column above and below and gray cells left and right, then substitutes red.
