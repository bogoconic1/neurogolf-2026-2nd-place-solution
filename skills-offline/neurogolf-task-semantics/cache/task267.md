# task267 Semantics

## Sources

- Current champion builder: `solutions_py/task267.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task267.json`
- ARC-GEN task id: `aabf363d`
- ARC-DSL task id: `aabf363d`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_aabf363d.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_aabf363d.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task267.py`

## Pattern

The input is a fixed `7x7` grid. A connected blob of one foreground color is drawn inside the central `5x5` window at rows `1..5`, columns `1..5`. A single marker pixel of the replacement color is placed at fixed coordinate `(6, 0)`. The output removes the marker pixel and recolors every blob cell from the original foreground color to the marker color. All other cells remain black.

## Readable Python Solver

```python
def solve(grid):
    out = [row[:] for row in grid]
    target = grid[6][0]
    source = None
    for r in range(7):
        for c in range(7):
            v = grid[r][c]
            if v != 0 and (r, c) != (6, 0):
                source = v
                break
        if source is not None:
            break
    for r in range(7):
        for c in range(7):
            if (r, c) == (6, 0):
                out[r][c] = 0
            elif out[r][c] == source:
                out[r][c] = target
    return out
```

## Generator Constraints

The generator always uses `size=7`. It samples a connected creature with `12..15` cells inside a `5x5` local box, then shifts those cells by `+1` in both row and column, so the recolored blob never touches the outer border. Two distinct random colors are chosen. The first color is used for every blob cell in both input and output; the second color is stored only at fixed marker coordinate `(6, 0)` in the input and replaces the blob color in the output. The marker is not part of the blob and is always removed in the output.

## Reference Notes

ARC-DSL identifies the least frequent nonblack color as the marker, replaces it with black, then identifies the remaining nonblack color and replaces it with the marker color. The code-golf solution uses the fixed marker coordinate `(6, 0)` and the fact that background should become black while every nonblack blob cell should become the marker color. The generator confirms the fixed-coordinate marker shortcut is safe for this task.
