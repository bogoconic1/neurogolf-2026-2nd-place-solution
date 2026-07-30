# task162 Semantics

## Sources

- Current champion builder: `solutions_py/task162.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task162.json`
- ARC-GEN task id: `6cf79266`
- ARC-DSL task id: `6cf79266`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_6cf79266.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_6cf79266.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task162.py`

## Pattern

The input is a 20x20 grid filled mostly with one non-blue foreground color and black background/noise. The generator cuts one to three 3x3 all-black holes into the dense foreground. The output preserves the input everywhere except that every unambiguous 3x3 all-black hole is filled with blue color 1.

## Readable Python Solver

The previous cache implementation did not survive replay. The compact executable oracle below is the bundled Code Golf 2025 reference; the Pattern section remains the readable specification.

```python
import re
p = lambda g, k=0: eval('1,1,1'.join(re.split(('(.{55})0, 0, 0' * 3)[7:], str(k or p(g, g)))))


def solve(grid):
    result = p([list(row) for row in grid])
    return [list(row) for row in result]
```

## Generator Constraints

The grid is square, normally 20x20, and `minisize=3`. A non-blue foreground color is chosen and about 80% of cells are initially filled with that color. Then one to three 3x3 black cutouts are inserted at random anchors in `[0, size-3]`. The generator repeats until the transformation is unambiguous: a 3x3 all-black region in the input must correspond exactly to a fill region in the output, avoiding ambiguous overlapping/adjacent zero holes. Output color 1 is reserved for the fills.

## Reference Notes

The ARC-DSL solver enumerates black cells, tests 3x3 neighborhoods, rejects candidates whose shifted 3x3 block differs from the set of black cells, filters predecessor/overlap ambiguity with a shifted negative-unity test, merges the surviving 3x3 blocks, and fills them blue. The Code Golf solution uses a regex over the serialized grid to find repeated `0, 0, 0` 3x3 hole patterns and replace them with blue rows.
