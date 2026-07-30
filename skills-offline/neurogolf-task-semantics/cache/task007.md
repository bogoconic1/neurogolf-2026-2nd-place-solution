# task007 Semantics

## Sources

- Current champion builder: `solutions_py/task007.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task007.json`
- ARC-GEN task id: `05269061`
- ARC-DSL task id: `05269061`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_05269061.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_05269061.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task007.py`

## Pattern

The input is a fixed 7x7 grid containing three colored anti-diagonal stripes, one stripe for each residue class of `row + col mod 3`. Only one diagonal from each residue class is shown. The output fills the entire 7x7 grid with repeating diagonal stripes: every cell gets the color associated with its `(row + col) % 3` residue.

## Readable Python Solver

```python
def solve(grid):
    colors = [0, 0, 0]
    for r, row in enumerate(grid):
        for c, color in enumerate(row):
            if color:
                colors[(r + c) % 3] = color
    return [[colors[(r + c) % 3] for c in range(7)] for r in range(7)]
```

## Generator Constraints

ARC-GEN uses a fixed `7x7` grid and exactly three nonzero colors. For each residue class `s in {0,1,2}`, it chooses one diagonal index from `s, s+3, s+6, ...` up to `12`; only cells on those three chosen diagonals are visible in the input. The output colors every cell according to the same period-3 anti-diagonal residue pattern. Colors are distinct nonzero ARC colors.

## Reference Notes

The ARC-DSL solver finds the colored objects, shifts the merged colored cells by all neighbor-of-neighbor offsets multiplied by three, and paints those shifted copies plus two diagonal shifts to fill the full period-3 stripe lattice. The Code Golf solution extracts the maximum color seen in each residue class and repeats those three colors across all rows.
