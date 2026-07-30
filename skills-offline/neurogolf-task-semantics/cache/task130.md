# task130 Semantics

## Sources

- Current champion builder: `solutions_py/task130.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task130.json`
- ARC-GEN task id: `5614dbcf`
- ARC-DSL task id: `5614dbcf`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_5614dbcf.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_5614dbcf.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task130.py`

## Pattern

The input is a `9 x 9` grid interpreted as a `3 x 3` array of `3 x 3` blocks. Some blocks are solid non-gray colors; empty blocks are black. Gray (`5`) noise pixels may appear inside blocks. The output is the `3 x 3` downscaled grid: each output cell is the block color, with gray noise ignored and empty blocks becoming black.

## Readable Python Solver

```python
def solve(grid):
    out = [[0 for _ in range(3)] for _ in range(3)]
    for br in range(3):
        for bc in range(3):
            counts = {}
            for dr in range(3):
                for dc in range(3):
                    color = grid[3 * br + dr][3 * bc + dc]
                    if color == 5:
                        color = 0
                    counts[color] = counts.get(color, 0) + 1
            out[br][bc] = max(counts, key=counts.get)
    return out
```

## Generator Constraints

ARC-GEN uses a fixed `3 x 3` block grid, so the input size is always `9 x 9` and output size is `3 x 3`. It samples 1 to 8 colored blocks from the nine possible block locations. Each colored block is filled with a random non-gray color. Separately, each block position may receive zero, one, or rarely two gray noise pixels at arbitrary offsets. Gray is excluded from block colors.

## Reference Notes

The ARC-DSL solver replaces gray with black, then downscales by factor three. The Code Golf solution recursively reduces rows/blocks and uses the maximum color after suppressing gray. The generator guarantees that at most two gray pixels can occur in a `3 x 3` block, so sampling any three cells in a row or column is enough to see at least one true block-color cell.
