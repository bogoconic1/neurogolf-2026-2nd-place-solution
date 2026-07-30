# task326 Semantics

## Sources

- Current champion builder: `solutions_py/task326.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task326.json`
- ARC-GEN task id: `d10ecb37`
- ARC-DSL task id: `d10ecb37`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_d10ecb37.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_d10ecb37.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task326.py`

## Pattern

The task returns the upper-left `2x2` crop of the input grid. If the input is a grid `grid`, the output is:

```text
[[grid[0][0], grid[0][1]],
 [grid[1][0], grid[1][1]]]
```

No recoloring, object detection, tie-breaking, or conditional logic is involved. All colors are copied exactly from the source cells.

## Readable Python Solver

```python
def solve(grid):
    return [row[:2] for row in grid[:2]]
```

## Generator Constraints

ARC-GEN chooses an even base size `size` from `{4, 6}`. Input width is `size` or `2*size`, and input height is `size` or `2*size`, so generated dimensions are in `{4, 6, 8, 12}` with both dimensions at least `4`. It samples a palette of three or four colors from `0..9`, fills every input cell from that palette, and sets the output to the top-left `2x2` cells. The validation examples include `6x6`, `8x8`, `6x12`, and `8x4` inputs.

Guaranteed invariants:

- the input always has at least two rows and two columns
- the output is exactly `2x2`
- all colors are ordinary ARC colors `0..9`
- every output cell is copied from the same coordinate in the input
- cells outside the `2x2` output are absent from the ARC output and must be inactive in the dense NeuroGolf tensor

## Reference Notes

The ARC-DSL solver is simply `crop(I, ORIGIN, TWO_BY_TWO)`. The Code Golf 2025 solution is the same rule in compact form: `p=lambda g:[g[0][:2],g[1][:2]]`. The ARC-GEN generator confirms that no hidden structure from the remaining input area matters.
