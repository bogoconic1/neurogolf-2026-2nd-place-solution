# task165 Semantics

## Sources

- Current champion builder: `solutions_py/task165.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task165.json`
- ARC-GEN task id: `6d58a25d`
- ARC-DSL task id: `6d58a25d`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_6d58a25d.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_6d58a25d.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task165.py`

## Pattern

The input is a 20x20 grid with black background, one fixed kite-shaped object in one nonzero color, and sparse noise pixels/components in one other nonzero color. The kite is a 10-cell downward shape anchored at `(row, col)`:

- row `r`: `c`
- row `r+1`: `c-1, c, c+1`
- row `r+2`: `c-2, c-1, c+1, c+2`
- row `r+3`: `c-3, c+3`

The output preserves the input and extends the noise color upward from the bottom in any column where a noise pixel has a kite cell somewhere above it. For each selected column, background cells are filled with the noise color from the bottom upward until the first kite cell in that column is reached. Existing kite cells remain kite-colored; existing noise pixels remain noise-colored.

## Readable Python Solver

The previous cache implementation did not survive replay. The compact executable oracle below is the bundled Code Golf 2025 reference; the Pattern section remains the readable specification.

```python
p = lambda g: [*zip(*map(lambda *r: r[:-(k := r[::-1].index(max(r, key=sum(g[::-1], g).index)))] + k * (max(r[-k:]),) + r, *g))]


def solve(grid):
    result = p([list(row) for row in grid])
    return [list(row) for row in result]
```

## Generator Constraints

ARC-GEN uses square 20x20 grids. Random noise pixels are sampled at roughly 2-5% density, all in the same non-kite color. The kite anchor row is in `[1, 10]`, and the anchor column is in `[5, size - 6]`, so all kite offsets remain in-bounds with margin. Exactly two non-background colors are chosen. The generator writes the same initial noise and kite cells into input and output, then performs the vertical bottom-up fills only for noise pixels with a kite cell above in the same column. It skips pixels that are inside the kite.

## Reference Notes

The ARC-DSL solution identifies the largest object as the kite, takes the merged remaining objects' color as the fill/noise color, filters objects that share a column with the kite and lie below the kite top, builds vertical frontiers through their centers, and underfills those cells with the noise color. The Code Golf 2025 solution is a compact transpose-oriented implementation of the same vertical-fill rule.

The direct generator is the clearest specification for edge behavior: selected columns fill from the bottom until the first kite cell encountered from below. This preserves lower kite endpoints and prevents filling above them in sparse outer kite columns.
