# task341 Semantics

## Sources

- Current champion builder: `solutions_py/task341.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task341.json`
- ARC-GEN task id: `d6ad076f`
- ARC-DSL task id: `d6ad076f`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_d6ad076f.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_d6ad076f.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task341.py`

## Pattern

The grid is `10x10`. The input has two non-cyan colored rectangular blocks. In canonical orientation before gravity is applied, a shorter top block sits above a longer bottom block. The output keeps both blocks and fills the vertical gap between them with cyan (`8`) over the interior columns of the shorter block. The whole input/output pair may then be rotated/flipped by one of four gravity directions, so the visible bridge may point down, up, right, or left.

Equivalently: identify the smaller rectangle and the larger rectangle. Shoot cyan rays from the interior cells of the smaller rectangle toward the larger rectangle along the axis connecting the rectangles. Fill only the empty cells between the two blocks; keep the original colored blocks unchanged.

## Readable Python Solver

The previous cache implementation did not survive replay. The compact executable oracle below is the bundled Code Golf 2025 reference; the Pattern section remains the readable specification.

```python
p = lambda g: [(g := [[g[i][j] or (9 > j >= 2 < len({*min(g[i - 1:][:3])})) * 8 for i in R] for j in R]) for R in [range(10)] * 2][1]


def solve(grid):
    result = p([list(row) for row in grid])
    return [list(row) for row in result]
```

## Generator Constraints

- Grid size is fixed at `10x10`.
- Two block thicknesses are each `2..4`; gaps at the outer edges are `0..1` except a thickness-4 block has no outer gap.
- Short length is `4..6`; long length is at least two cells longer and at most `9`.
- In canonical orientation, the long lower block horizontally contains the short upper block with at least one extra column on each side.
- The cyan bridge occupies the gap between the blocks and the interior columns of the short block, excluding the short block's left/right border columns.
- `common.apply_gravity` applies one of four orientations to both input and output.

## Reference Notes

ARC-DSL chooses the smallest object and largest object, infers the bridge direction from vertical matching, shoots rays from the smaller object's inbox/interior toward the larger object, underfills cyan, then removes any cyan component that borders the original input boundary. The Code Golf solution uses repeated transposition/rotation to canonicalize directions and fills cyan where three-row windows indicate the bridge gap.

The generator is authoritative for dimensions and edge gaps; the DSL clarifies that the fill is from the smaller block's interior toward the larger block and must not leak to boundary-bordering cyan components.
