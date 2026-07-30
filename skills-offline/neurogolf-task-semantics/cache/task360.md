# task360 Semantics

## Sources

- Current champion builder: `solutions_py/task360.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task360.json`
- ARC-GEN task id: `e3497940`
- ARC-DSL task id: `e3497940`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_e3497940.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_e3497940.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task360.py`

## Pattern

The input is a `10 x 9` grid with a fixed gray separator column at column 4. The left half is columns 0..3 and the right half is columns 5..8. Sparse colored pixels appear in mirrored pairs around the separator: output column `c` corresponds to input columns `c` and `8-c`. A generated item can appear on the left side only, the right side only, or on both sides; when both sides are present for the same mirrored location, they have the same foreground color. Black is background, gray separator cells are ignored, and the output is a `10 x 4` grid containing the left half overlaid with the mirrored right half.

For every row `r` and output column `c in 0..3`, the output color is `max(input[r][c], input[r][8-c])`. This `max` is equivalent to an OR over matching one-hot foreground channels under the generator invariant that both nonzero sides, when present, share the same color.

## Readable Python Solver

```python
def solve(grid):
    return [[max(row[c], row[8 - c]) for c in range(4)] for row in grid]
```

## Generator Constraints

- Input height is always 10 and input width is always 9 (`2 * 4 + 1`).
- Column 4 is filled with gray (`5`) and is not copied to the output.
- Two foreground colors are chosen excluding black and gray.
- Generated rows occupy an interior vertical band, leaving at least one or two blank rows near the top and bottom depending on the random draw.
- For each active row, the generator creates one or more mirrored offsets from the separator. Each offset may be left-only, right-only, or both. If both sides are present, the color is identical on the two sides.
- Output width is exactly 4. Right-half input columns map as `8 -> 0`, `7 -> 1`, `6 -> 2`, and `5 -> 3`.

## Reference Notes

The ARC-DSL reference takes the left half, extracts the right half, mirrors the right half horizontally, and paints/merges the mirrored right objects onto the left-half canvas. The Code Golf 2025 solution is the compact row expression `[*map(max, r, r[:4:-1])]`, which zips row columns `0..3` with reversed columns `8..5` and applies `max`. The ARC-GEN generator confirms the fixed separator geometry and the same-color invariant for both-sided mirrored foreground pixels.
