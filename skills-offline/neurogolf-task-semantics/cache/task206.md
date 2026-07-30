# task206 Semantics

## Sources

- Current champion builder: `solutions_py/task206.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task206.json`
- ARC-GEN task id: `88a10436`
- ARC-DSL task id: `88a10436`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_88a10436.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_88a10436.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task206.py`

## Pattern

The input contains one connected colored sprite and one gray marker pixel. The
colored sprite uses colors sampled from `{1, 2, 3, 6}` and occupies cells inside
a 3x3 box whose center cell is occupied. The gray marker has color `5`.

The output preserves the original grid and paints a second copy of the colored
sprite at the gray marker location. More exactly, normalize the colored sprite
to its own 3x3 local coordinates, then place local coordinate `(1, 1)` on the
gray marker cell. This overwrites the gray marker with the copied sprite color
at that local center cell. Output size equals input size.

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])

    marker = None
    colored = []
    for r in range(h):
        for c in range(w):
            v = grid[r][c]
            if v == 5:
                marker = (r, c)
            elif v in {1, 2, 3, 6}:
                colored.append((r, c, v))

    if marker is None:
        return [row[:] for row in grid]

    min_r = min(r for r, _, _ in colored)
    min_c = min(c for _, c, _ in colored)
    mr, mc = marker

    out = [row[:] for row in grid]
    for r, c, v in colored:
        rr = mr + (r - min_r) - 1
        cc = mc + (c - min_c) - 1
        out[rr][cc] = v
    return out
```

## Generator Constraints

- Width and height are independently sampled from `7..12`.
- The colored object is a connected Conway sprite contained in a 3x3 local box.
  The local center `(1, 1)` is always occupied.
- Colored sprite cells use colors sampled from `{1, 2, 3, 6}`. The generator
  samples three colors from that set and assigns one sampled color to each
  sprite cell, so repeated colors are common and not semantically meaningful.
- There are two placement centers, both interior cells. The two centers are
  separated by at least four rows or at least four columns, preventing overlap
  between the source sprite and target marker/sprite.
- The input contains the first colored sprite and only a gray marker at the
  second center. The output contains colored copies at both centers.
- The gray marker is at the target sprite's local center and is overwritten in
  the output.

## Reference Notes

ARC-DSL extracts objects, separates the color-5 marker object from the remaining colored object, normalizes the colored object, shifts it to the marker center, then shifts by `(-1, -1)` before painting onto the original grid. This confirms that the marker denotes the center of the 3x3 local sprite frame rather than the top-left corner.

The Code Golf solution finds the marker's flat index, trims empty rows and columns after ignoring colors `0` and `5`, and writes the resulting sprite rows around the marker. This independently confirms that the transformation is a copy of the colored sprite only, not a recolor or reflection.

No disagreement was found among ARC-GEN, ARC-DSL, and Code Golf 2025.
