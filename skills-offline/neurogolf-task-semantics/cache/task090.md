# task090 Semantics

## Sources

- Current champion builder: `solutions_py/task090.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task090.json`
- ARC-GEN task id: `3eda0437`
- ARC-DSL task id: `3eda0437`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_3eda0437.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_3eda0437.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task090.py`

## Pattern

The input is a short rectangular grid containing mostly blue `1`, occasional
gray `5`, and black `0`. There is one unique largest all-black axis-aligned
rectangle. The output has the same shape as the input and preserves every
original cell except that every cell in this largest black rectangle is
recolored pink `6`. Other black cells remain black unless they are inside the
selected rectangle.

## Readable Python Solver

The previous cache implementation did not survive replay. The compact executable oracle below is the bundled Code Golf 2025 reference; the Pattern section remains the readable specification.

```python
import re
p = lambda g: eval('6'.join(max([re.split((f"(..{{{len(g[0]) * 3 - i % 8 * 3}}})0{i % 8 * '(, )0'}" * (i % 5 + 2))[8:], str(g), 1) for i in range(40)], key=len)))


def solve(grid):
    result = p([list(row) for row in grid])
    return [list(row) for row in result]
```

## Generator Constraints

ARC-GEN creates grids with width `20..30` and height `2..5`. Background noise
is nonzero: mostly blue `1`, with rare gray `5`. A black rectangular cutout is
inserted with width `2..5` and height `2..height`. The generator repeats until
there is exactly one largest all-black rectangle in the whole input. The output
fills that rectangle with pink `6` and leaves all other cells unchanged.

## Reference Notes

The ARC-DSL solver enumerates possible rectangle sizes from `2..9`, finds occurrences of an all-zero canvas of each size, chooses the union with maximum cell count, and fills it with `6`. The Code Golf solution uses regular expressions over the serialized grid to identify the largest zero rectangle. All references agree that the selected object is a filled all-zero rectangle, not a border, component, or bounding box around scattered zeros.
