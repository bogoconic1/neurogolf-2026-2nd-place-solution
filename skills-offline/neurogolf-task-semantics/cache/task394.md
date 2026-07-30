# task394 Semantics

## Sources

- Current champion builder: `solutions_py/task394.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task394.json`
- ARC-GEN task id: `f9012d9b`
- ARC-DSL task id: `f9012d9b`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_f9012d9b.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_f9012d9b.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task394.py`

## Pattern

The input is a square periodic color pattern with one black square bite removed. The original pattern repeats a `2x2` tile for grid sizes `4..6`, and a `3x3` tile for grid size `7`. The bite is a contiguous square of black cells. The output is exactly the missing bite content, reconstructed from the periodic pattern.

## Readable Python Solver

```python
def solve(grid):
    n = len(grid)
    period = 3 if n == 7 else 2
    black = [(r, c) for r, row in enumerate(grid) for c, v in enumerate(row) if v == 0]
    r0 = min(r for r, _ in black)
    c0 = min(c for _, c in black)
    b = int(len(black) ** 0.5)

    out = []
    for dr in range(b):
        row = []
        for dc in range(b):
            rr = (r0 + dr) % period
            cc = (c0 + dc) % period
            # Find any visible cell with the same phase in the periodic grid.
            val = None
            for r in range(n):
                for c in range(n):
                    if r % period == rr and c % period == cc and grid[r][c] != 0:
                        val = grid[r][c]
                        break
                if val is not None:
                    break
            row.append(val)
        out.append(row)
    return out
```

## Generator Constraints

ARC-GEN chooses `size` from `4..7`. For sizes `4`, `5`, and `6`, the repeating tile is `2x2`; for size `7`, the repeating tile is `3x3`. The tile colors are two randomly chosen non-black colors, assigned independently to every tile phase, so multiple phases can share a color. Bite size is `1` for size `4`, `2` for sizes `5` and `6`, and either `2` or `3` for size `7`. The bite top-left row and column are uniformly chosen so the square bite fits inside the grid. The input is the full periodic grid with the bite cells set to black; the output is the original colors in that bite square.

## Reference Notes

ARC-DSL finds the black object, infers vertical and horizontal periods from unbitten halves, paints periodic copies of the missing object, and returns the black-object subgrid after reconstruction. The Code Golf 2025 solution uses row filtering and modular index shifts to pull the missing periodic cells. The references agree that black cells are the bite, not ordinary pattern colors.
