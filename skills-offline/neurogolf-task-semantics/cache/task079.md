# task079 Semantics

## Sources

- Current champion builder: `solutions_py/task079.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task079.json`
- ARC-GEN task id: `39a8645d`
- ARC-DSL task id: `39a8645d`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_39a8645d.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_39a8645d.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task079.py`

## Pattern

The input is a square grid, normally 14x14 in ARC-GEN, containing non-overlapping copies of two or three different 3x3 sprites. Each sprite type has a unique nonzero color. Copies of the same type have the same 3x3 shape and color. The output is a 3x3 crop of the sprite type that appears the most times in the input, preserving its color and black cells.

ARC-GEN assigns the sprite types distinct copy counts sampled from 1..3 and sorted descending, so the target type is unique. The target is also the most common object color when objects are extracted as single-color connected components.

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])
    candidates = []
    for r in range(h - 2):
        for c in range(w - 2):
            patch = tuple(tuple(grid[r + dr][c + dc] for dc in range(3)) for dr in range(3))
            if all(v == 0 for row in patch for v in row):
                continue
            if any(all(patch[dr][dc] == 0 for dc in range(3)) for dr in range(3)):
                continue
            if any(all(patch[dr][dc] == 0 for dr in range(3)) for dc in range(3)):
                continue
            colors = {v for row in patch for v in row if v != 0}
            if len(colors) == 1:
                candidates.append(patch)
    return [list(row) for row in max(candidates, key=candidates.count)]
```

This matches the Code Golf idea: enumerate 3x3 windows with no all-zero row or column, then choose the most frequent window. The ARC-DSL solver reaches the same result by extracting objects, finding the most common object color, selecting an object of that color, and returning its subgrid.

## Generator Constraints

- Grid size is square, default `14x14` in ARC-GEN.
- There are two or three sprite types.
- Colors are distinct nonzero ARC colors.
- Each sprite is a 3x3 Conway-style diagonally connected pattern; its occupied cells have no all-zero row or all-zero column after cropping.
- Copy counts are distinct values sampled from `1..3` and sorted descending, so the most frequent sprite type is unique.
- Sprite copies are placed as non-overlapping 3x3 bounding boxes with at least one-cell separation according to `common.overlaps(..., 1)`.
- The output is exactly `3x3` and contains the most frequent sprite shape in its own color.

## Reference Notes

ARC-DSL:

```python objects(I, T, T, T) -> colors -> mostcommon -> extract matching object -> subgrid ```

This confirms that the semantic target is the object/color with the greatest number of copies, not the largest area or highest color id.

ARC-GEN explicitly constructs type 0 with the highest copy count and writes the output from type 0's sprite cells. Code Golf confirms that a purely local approach can scan every 3x3 window, discard windows with empty rows/columns, and take the most repeated 3x3 pattern.
