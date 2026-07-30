# task041 Semantics

## Sources

- Current champion builder: `solutions_py/task041.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task041.json`
- ARC-GEN task id: `22168020`
- ARC-DSL task id: `22168020`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_22168020.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_22168020.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task041.py`

## Pattern

The visible task grid is 10x10. The input contains up to six non-overlapping colored roof/zigzag outlines on a zero background. Each object uses one foreground color and has an even horizontal length 4, 6, or 8. Its outline has two same-color pixels on each roof row, narrowing toward the center, plus the two center pixels on the bottom row. The output keeps the outline and fills every horizontal span between each same-color pair on a row, producing the solid triangular/roof interior for every object. Multiple objects can appear in the same grid but their bounding boxes do not overlap.

Equivalently, each row can be decoded left-to-right by toggling a color state when a nonzero pixel is encountered and emitting the current state through the interval. Because generated objects contribute paired endpoints on each row, the state returns to zero after each span.

## Readable Python Solver

```python
def solve(grid):
    out = []
    for row in grid:
        state = 0
        out_row = []
        for value in row:
            if value:
                state ^= value
            out_row.append(value | state)
        out.append(out_row)
    return out
```

## Generator Constraints

ARC-GEN samples square 10x10 grids. For each of up to six placement attempts it chooses length in {4, 6, 8}, a top row from 0 through `size - length // 2 - 1`, and a left column from 0 through `size - length`. A candidate is skipped if its bounding box overlaps a previously accepted object; vertical contact at one boundary row is treated as overlap, while horizontal adjacency is allowed. Foreground colors are sampled with `random_colors(len(lengths))`, so accepted objects have distinct nonzero colors in normal generator output. The object top row contains the two outer endpoints, intermediate rows contain symmetric endpoint pairs, and the bottom row contains the two center pixels. Outputs are made by extending each input colored endpoint vertically upward to the object's top row, which is identical to filling the horizontal interval between paired endpoints on every affected row.

## Reference Notes

The ARC-DSL solution takes every foreground color, connects every pair of cells of that color, recolors the resulting connected cells, and paints them onto the input. For this generator, same-color cells are exactly one roof object, so the union of pairwise connections fills the interior. The Code Golf 2025 solution is `[[v | (a := a ^ v) for v in r] for r in g]`, a compact prefix-XOR row scan. It carries `a` syntactically across rows, but generated rows end with zero state because endpoints are paired, so it behaves as a per-row scan on valid examples.
