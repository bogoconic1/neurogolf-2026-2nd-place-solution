# task400 Semantics

## Sources

- Current champion builder: `solutions_py/task400.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task400.json`
- ARC-GEN task id: `ff805c23`
- ARC-DSL task id: `ff805c23`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_ff805c23.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_ff805c23.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task400.py`

## Pattern

The input is a 24x24 square image embedded in the standard NeuroGolf 30x30 one-hot tensor. A contiguous 5x5 block of color `1` marks a cutout whose original contents are hidden. The visible image was generated from an eight-way mirror-symmetric pattern around the square: each orbit under row, column, and diagonal/anti-diagonal reflection has one color. The output is the hidden 5x5 patch that belongs under the color-1 marker block.

The hidden patch can be recovered by taking a 5x5 crop from a mirrored location that is not covered by the marker. A direct recovery uses the 180-degree reflected location: if the marker top-left is `(r, c)`, the output patch is equivalent to `grid[23-r : 18-r : -1, 23-c : 18-c : -1]`. ARC-DSL expresses an alternate safe rule: crop the marker bounding box from a horizontal mirror, and if that crop still contains color `1`, use the vertical mirror crop instead.

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])

    marker = [(r, c) for r in range(h) for c in range(w) if grid[r][c] == 1]
    r0 = min(r for r, _ in marker)
    c0 = min(c for _, c in marker)
    size = 5

    # Use a mirrored copy of the occluded patch. The ARC-GEN generator makes the
    # whole 24x24 pattern symmetric enough that this crop equals the hidden area.
    return [
        [grid[h - 1 - (r0 + dr)][w - 1 - (c0 + dc)] for dc in range(size)]
        for dr in range(size)
    ]
```

## Generator Constraints

- `size` defaults to 24 and `cutout` defaults to 5; all bundled examples use those values.
- The marker is one contiguous 5x5 rectangle of color `1`, with top-left `(brow, bcol)` chosen anywhere from `0` through `19` in both dimensions.
- Color `1` is reserved for the marker. Pattern colors are black `0` plus three random non-blue colors. The outer/inner regions bias toward different chosen colors, but any orbit may be black.
- The generator fills each eight-way reflection orbit with a single color, then records the original 5x5 cutout as the output and replaces that region with color `1`.
- The generator rejects placements where the 5x5 marker would cover every cell in an orbit needed by the draw loop. This guarantees enough visible symmetric evidence to reconstruct the hidden patch.

## Reference Notes

- ARC-DSL: `ofcolor(I, ONE)` finds the marker cells. It tries `subgrid(marker, hmirror(I))`; if that crop still contains `1`, it instead uses `subgrid(marker, vmirror(I))`. This confirms that one reflected copy of the marker-aligned 5x5 patch is the answer, with a fallback when the chosen mirror overlaps the marker.
- Code Golf 2025: the concise Python pops rows from the bottom while scanning marker rows and reverses from the corresponding marker column, which is a compact 180-degree reflected crop.
