# task017 Semantics

## Sources

- Current champion builder: `solutions_py/task017.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task017.json`
- ARC-GEN task id: `0dfd9992`
- ARC-DSL task id: `0dfd9992`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_0dfd9992.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_0dfd9992.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task017.py`

## Pattern

The input is a 21x21 periodic quadratic-residue color pattern with five black rectangular cutouts. The output restores the complete pattern, filling the black cutouts with the color that would have been present in the underlying periodic grid.

The generated full pattern is defined by parameters `mod in 4..9`, `length in 4..mod`, and `offset in 1..length`. For each row/column, compute centered residues `r = (offset + row) % length - length // 2` and `c = (offset + col) % length - length // 2`; the color is `(r*r + c*c) % mod + 1`. The input copies this full pattern, then zeroes five random rectangles of width and height 2..5.

## Readable Python Solver

```python
def solve(grid):
    # Reference-style readable behavior: recover the row/column periods from
    # visible nonzero cells, then repaint black cutouts with the repeated
    # periodic pattern. A compact exact implementation is nontrivial; the
    # ARC-DSL solver does this by measuring vertical and horizontal periods
    # from border crops and painting shifted copies of the nonzero object.
    h, w = len(grid), len(grid[0])
    out = [row[:] for row in grid]

    # Brute-force readable solver over generator bounds.
    for mod in range(4, 10):
        for length in range(4, mod + 1):
            for offset in range(1, length + 1):
                pattern = []
                for row in range(h):
                    rr = (offset + row) % length - length // 2
                    prow = []
                    for col in range(w):
                        cc = (offset + col) % length - length // 2
                        prow.append((rr * rr + cc * cc) % mod + 1)
                    pattern.append(prow)
                if all(grid[r][c] in (0, pattern[r][c]) for r in range(h) for c in range(w)):
                    return pattern
    return out
```

## Generator Constraints

- Grid size is fixed at 21x21.
- `mod` is 4..9.
- `length` is 4..mod.
- `offset` is 1..length.
- Five black cutouts are placed at random. Each width and height is 2..5.
- Cutouts may overlap. Visible cells outside cutouts exactly match the hidden periodic quadratic pattern.
- Output contains colors 1..9 from the restored periodic pattern; black cutouts are fully filled.

## Reference Notes

- ARC-DSL recovers vertical and horizontal periods from the nonzero object, then paints shifted copies of visible nonzero cells into the black regions.
- ARC-GEN exposes the exact bounded formula `(r*r + c*c) % mod + 1`.
- The Code Golf solution uses row/column recursion and max/overlay behavior to fill zero cells from repeated visible rows/columns.
