# task255 Semantics

## Sources

- Current champion builder: `solutions_py/task255.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task255.json`
- ARC-GEN task id: `a64e4611`
- ARC-DSL task id: `a64e4611`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_a64e4611.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_a64e4611.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task255.py`

## Pattern

The task is a 30x30 hidden-pipe restoration. The output contains a green (`3`) rectangular pipe network drawn over random black/non-green noise. The input is the same grid except every green cell is hidden as black (`0`); all non-green colored noise remains unchanged.

The green network consists of a thick main artery and optional side veins. In the generator's untransformed orientation, the artery is a vertical rectangular interior strip whose top either touches the top edge or starts lower at row 5. It has a black one-cell outline and a green interior. Optional left veins extend from the artery to the left edge, and optional right veins extend from the artery to the right edge. Veins are horizontal rectangular interiors with black outlines. The full pattern may be transposed, and may then be vertically flipped, so the observable orientation can be vertical or horizontal and can be mirrored.

The output restores exactly the hidden green interiors while preserving the original random non-green color and black background/outline cells.

## Readable Python Solver

The previous cache implementation did not survive replay. The compact executable oracle below is the bundled Code Golf 2025 reference; the Pattern section remains the readable specification.

```python
import re
p = lambda o: [(o := eval(re.sub(*p, f'{(*zip(*o[::-1]),)}'))) for p in [['0(?=(?<=[^0], 0).{,12}, [^0].{80}, [^0])', '12'[1 in o[0]]]] * 20 + [['0(?=(?<=[^0], 0).{91}[^0].{,5}0|, [^0].{85}0, [^0])', '12'[1 in o[0]]]] * 20 + [['0(?=, [^0])', '12'[1 in o[0]]]] * 4 + [['[30], ' * 12, '3,' * 12]] * 4 + [['12'[1 in o[0]], '0']] * 4][51]


def solve(grid):
    result = p([list(row) for row in grid])
    return [list(row) for row in result]
```

## Generator Constraints

The grid is always 30x30. The random foreground color is any color except green (`3`), and random foreground pixels are placed with probability about 0.5 before the pipe network is carved. The pipe rectangles are drawn by first clearing the full rectangle to black and then filling the strict interior green, leaving a one-cell black outline.

The main artery has column in `[5,10]`, width in `[6,12]`, and height `32`, with row `-1` or `4`, so its green interior spans most or all of the grid. Left veins, when present, start at column `-1` and extend to the artery's right side; they may be one tall vein of height `10..14` or two short veins of height `3..4`. Right veins, when present, start two columns before the artery's right edge and extend to the right edge; one may be high (`+6..9` rows from the artery) and one may be much lower (`+18..24`). After drawing, the whole bitmap may be transposed and may be vertically flipped.

Input colors never include green because output green cells are converted to black in the input. Output colors include green plus the original random non-green foreground color and black.

## Reference Notes

The ARC-DSL solver constructs candidate rectangular templates, finds occurrences in the input, shifts those templates, fills green in several passes, then uses local neighborhood predicates to complete black cells forced by surrounding green and border constraints. This confirms that the task is not a simple color replacement; the hidden green cells must be inferred from the pipe geometry and surrounding black corridors.

The Code Golf 2025 solution serializes the grid and repeatedly applies regex patterns that turn eligible zeros into temporary colors and then green. The repeated regex passes encode the same local completion process: seed long corridors, propagate through side arms, fill cells adjacent to established green structure, then normalize temporary marks.
