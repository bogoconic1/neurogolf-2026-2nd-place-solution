# task106 Semantics

## Sources

- Current champion builder: `solutions_py/task106.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task106.json`
- ARC-GEN task id: `46442a0e`
- ARC-DSL task id: `46442a0e`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_46442a0e.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_46442a0e.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task106.py`

## Pattern

The input is a square `N x N` color grid where `N` is either `2` or `3`. The output is a `2N x 2N` square made from the input and its rotations:

```text
I          rot90(I)
rot270(I)  rot180(I)
```

The output preserves every input color exactly and introduces no new colors.

## Readable Python Solver

The previous cache implementation did not survive replay. The compact executable oracle below is the bundled Code Golf 2025 reference; the Pattern section remains the readable specification.

```python
p = lambda g: (w := (g + [*zip(*g)][::-1])) + [k + [*w.pop()][::-1] for *k, in w * 1]


def solve(grid):
    result = p([list(row) for row in grid])
    return [list(row) for row in result]
```

## Generator Constraints

ARC-GEN chooses `size` uniformly in `2..3`. It chooses three random colors and fills the `size*size` input cells by sampling from those colors, so repeated colors are common and all colors are ordinary ARC digits. There are no markers, background assumptions, or object ambiguities. The generated output is always `4x4` for size `2` or `6x6` for size `3`, embedded in the standard NeuroGolf `[1,10,30,30]` tensor.

## Reference Notes

The ARC-DSL solver directly builds `rot90(I)`, `rot180(I)`, `rot270(I)`, then concatenates horizontally and vertically. The Code Golf 2025 solution is a compressed expression of the same rotation-and-concatenation rule.
