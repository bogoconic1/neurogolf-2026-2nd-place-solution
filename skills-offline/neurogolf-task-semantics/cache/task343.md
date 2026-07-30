# task343 Semantics

## Sources

- Current champion builder: `solutions_py/task343.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task343.json`
- ARC-GEN task id: `d8c310e9`
- ARC-DSL task id: `d8c310e9`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_d8c310e9.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_d8c310e9.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task343.py`

## Pattern

The grid is always height `5` and width `15`. Nonzero cells form a bottom-anchored vertical-bar motif repeated horizontally. The motif width is `3` or `4`. Each motif column has height `1..3`; its colored cells occupy the bottom rows. Colors come from a small palette of two or three non-background colors and are assigned per cell in the motif.

The input shows only a left prefix of the repeated pattern. The output completes the same pattern across all 15 columns. There are two modes:

- normal mode: every period repeats the same motif left-to-right;
- flipped mode: every other period is horizontally mirrored.

The visible prefix is guaranteed long enough to infer the period and flip mode: at least two periods plus a partial column in normal mode, and at least three periods plus a partial column in flipped mode.

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])

    def col(c):
        return tuple(grid[r][c] for r in range(h))

    cols = [col(c) for c in range(w)]
    last_visible = max((c for c, v in enumerate(cols) if any(v)), default=-1)
    if last_visible < 0:
        return [row[:] for row in grid]

    # Try the generator-supported motif widths and flip modes. Choose the first
    # pattern that exactly explains the visible nonzero prefix.
    for p in (3, 4):
        for flip in (False, True):
            block0 = cols[:p]
            ok = True
            for c in range(last_visible + 1):
                block = c // p
                j = c % p
                src = p - 1 - j if (flip and block % 2) else j
                expected = block0[src]
                # A partially visible final block may omit still-hidden cells by
                # being all zero only beyond last_visible; within the visible
                # prefix the drawn column must match the repeated/mirrored motif.
                if cols[c] != expected:
                    ok = False
                    break
            if not ok:
                continue
            out_cols = []
            for c in range(w):
                block = c // p
                j = c % p
                src = p - 1 - j if (flip and block % 2) else j
                out_cols.append(block0[src])
            return [[out_cols[c][r] for c in range(w)] for r in range(h)]

    # Fallback matching the Code Golf observation: repeat the shortest visible
    # prefix that explains the row strings.
    out = [row[:] for row in grid]
    for r, row in enumerate(grid):
        for p in (3, 4, 6, 8):
            prefix = row[:p]
            if any(prefix):
                out[r] = (prefix * ((w + p - 1) // p))[:w]
                break
    return out
```

## Generator Constraints

- `height=5`, `width=15`.
- Motif width `len(lengths)` is `3` or `4`.
- Each motif column height is independently `1..3`, drawn upward from the bottom row.
- The color sequence length is `sum(lengths)`, using two or three randomly chosen colors. Colors are assigned bottom-up within each motif column.
- If `flip=False`, every motif block is identical.
- If `flip=True`, odd-numbered motif blocks are horizontally mirrored; colors stay attached to the original motif columns when mirrored.
- `visible=(flip+2)*len(lengths)+randint(0,len(lengths)-1)`, so the input contains enough complete periods to identify normal versus flipped repetition, followed by a partial period.
- Output is the full generated pattern, not a resized grid.

## Reference Notes

ARC-DSL identifies the first non-background object, computes its horizontal period with `hperiod`, and paints shifted copies at one and three periods to fill the hidden continuation. The Code Golf 2025 solution works row-wise: it decides whether to use a 6- or 8-column prefix based on whether the first three 4-column windows indicate flipped structure, repeats that prefix, and truncates to width 15.

The references agree that task343 is pattern continuation by horizontal period, with a possible alternating mirror mode. The ARC-GEN generator is the clearest source for period sizes, visible-prefix guarantees, and edge cases.
