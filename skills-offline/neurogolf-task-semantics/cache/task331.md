# task331 Semantics

## Sources

- Current champion builder: `solutions_py/task331.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task331.json`
- ARC-GEN task id: `d364b489`
- ARC-DSL task id: `d364b489`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_d364b489.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_d364b489.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task331.py`

## Pattern

The input is a 10x10 black grid with sparse blue (`1`) pixels. The output has
the same 10x10 size. Each blue pixel remains blue at its original cell. Its
valid cardinal neighbors are recolored with a direction-specific color:

- the cell above the blue pixel becomes red (`2`)
- the cell below becomes cyan (`8`)
- the cell to the right becomes magenta/pink (`6`)
- the cell to the left becomes orange (`7`)

Neighbors that would fall outside the 10x10 grid are omitted. Black background
cells that are not a blue source or one of these valid cardinal neighbors stay
black (`0`). The generator separates blue pixels far enough that the colored
crosses do not conflict, so there is no tie-breaking between overlapping marks.

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    for r in range(h):
        for c in range(w):
            if grid[r][c] != 1:
                continue
            if r > 0:
                out[r - 1][c] = 2
            if r + 1 < h:
                out[r + 1][c] = 8
            if c > 0:
                out[r][c - 1] = 7
            if c + 1 < w:
                out[r][c + 1] = 6
            out[r][c] = 1
    return out
```

## Generator Constraints

ARC-GEN uses a square grid with `size=10` for the benchmark examples. It samples
up to 10 candidate positions and keeps a candidate only when, relative to every
already-kept blue pixel, `abs(row difference) >= 3` or `abs(column difference) >=
3`. This prevents overlap between the 3x3 neighborhoods relevant to the
cross-coloring rule. Blue pixels may appear on edges or corners, so every
directional mark must be clipped to the original grid bounds. The only input
foreground color is blue (`1`); output adds red (`2`), magenta/pink (`6`), orange
(`7`), and cyan (`8`).

## Reference Notes

The ARC-DSL solver finds all blue cells, then fills shifted copies of that set: down-shifted blue cells become cyan (`8`), up-shifted cells become red (`2`), right-shifted cells become magenta (`6`), and left-shifted cells become orange (`7`). The Code Golf 2025 solution encodes the same four directional fills by rotating/transposing the grid and replacing shifted `1,0` patterns with the direction color. There is no disagreement between the references.
