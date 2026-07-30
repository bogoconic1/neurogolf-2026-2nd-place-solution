# task219 Semantics

## Sources

- Current champion builder: `solutions_py/task219.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task219.json`
- ARC-GEN task id: `90f3ed37`
- ARC-DSL task id: `90f3ed37`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_90f3ed37.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_90f3ed37.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task219.py`

## Pattern

The input is a `15x10` grid containing several horizontally structured bands of
cyan `8` cells on black. Each band has the same height, at most 3 rows. In the
topmost band, the full structure is visible: a left pattern `A` repeated from
column 0 up to a middle column, a middle pattern `B`, and a right pattern `C`
repeated from just after `B` to the right edge. In every lower band, only the
left `A` repeats and the middle `B` are visible in cyan; the right `C` repeats
are missing from the input.

The output preserves all input cyan cells and fills the missing right-side
copies of the top band's `C` pattern in blue `1` for every lower band. The
topmost full band remains cyan; only missing cells in lower bands are painted.

## Readable Python Solver

The previous cache implementation did not survive replay. The compact executable oracle below is the bundled Code Golf 2025 reference; the Pattern section remains the readable specification.

```python
def p(i):
    f = min((r for r in range(8) if any(i[r])))
    for d in range(8):
        for e in range((r := min((r for r in range(f, 9) if any(i[r]) - 1))), f - r + 16):
            for y in range(5):
                if min(((2 * i[e + r - f][:1] + i[e + r - f])[o + 4 - y] == i[r][o] * (o < 9 - d) for r in range(f, r) for o in range(8))):
                    i[e:e + r - f] = [[i[e + r - f][o] or (i[e + r - f][:2] + i[r] + i[r][-2:])[o + y] % 7 for o in range(10)] for r in range(f, r)]
    return i


def solve(grid):
    result = p([list(row) for row in grid])
    return [list(row) for row in result]
```

## Generator Constraints

- Grid size is fixed at width `10`, height `15`.
- All visible input pixels are cyan `8`; missing generated output pixels are
  blue `1`.
- Band height `tall` is `1`, `2`, or `3`.
- Pattern widths `awide`, `bwide`, and `cwide` are each `1` or `2`.
- The `A`, `B`, and `C` patterns are sampled nonempty subsets of their
  `width x tall` blocks.
- The first band contains `A` repeats on the left, `B` at the split column, and
  `C` repeats on the right, all cyan.
- Lower bands contain the same visible left and middle cyan structure; their
  right-side `C` repeats are omitted from the input and painted blue in the
  output.
- Band start rows are separated by at least one blank row and at most a few
  blank rows.

## Reference Notes

ARC-GEN is the clearest source: it explicitly draws the right-side `C` repeats as cyan only for the first band and as blue-only output for later bands. ARC-DSL finds connected cyan objects, takes the uppermost object as the complete template, then shifts that normalized template against every lower object and selects the shift with maximum intersection before underfilling blue. The Code Golf solution performs the same alignment and repeated right-side fill with compact row slicing.
