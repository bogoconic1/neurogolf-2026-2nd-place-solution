"""Inspect NeuroGolf task JSON files and print agent-friendly summaries."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_REPO_ROOT = SCRIPT_DIR if (SCRIPT_DIR / "mapping.csv").exists() else SCRIPT_DIR.parent
REPO_ROOT = Path(os.environ.get("NEUROGOLF_REPO_ROOT", DEFAULT_REPO_ROOT)).resolve()
DATA_DIR = REPO_ROOT / "neurogolf-2026"
SPLITS = ("train", "test", "arc-gen")


def print_json(value: Any) -> None:
    print(json.dumps(value, indent=2))


def posix_path(path: Path) -> str:
    return path.as_posix()


def normalize_task_id(value: str) -> str:
    text = value.strip().lower()
    suffix = text[4:] if text.startswith("task") else text
    if not suffix.isdigit():
        raise argparse.ArgumentTypeError(f"expected task id like task123 or 123: {value!r}")
    number = int(suffix)
    if not 1 <= number <= 400:
        raise argparse.ArgumentTypeError("expected task number from 1 to 400")
    return f"task{number:03d}"


def grid_shape(grid: list[list[int]]) -> list[int]:
    return [len(grid), len(grid[0]) if grid else 0]


def color_counts(grid: list[list[int]]) -> dict[str, int]:
    counts: Counter[int] = Counter()
    for row in grid:
        counts.update(row)
    return {str(color): counts[color] for color in sorted(counts)}


def bbox_for_cells(cells: list[tuple[int, int]]) -> list[int] | None:
    if not cells:
        return None
    rows = [row for row, _ in cells]
    cols = [col for _, col in cells]
    return [min(rows), min(cols), max(rows), max(cols)]


def bboxes_by_color(grid: list[list[int]]) -> dict[str, list[int]]:
    cells_by_color: dict[int, list[tuple[int, int]]] = {}
    for row_index, row in enumerate(grid):
        for col_index, color in enumerate(row):
            if color == 0:
                continue
            cells_by_color.setdefault(color, []).append((row_index, col_index))
    return {
        str(color): bbox
        for color in sorted(cells_by_color)
        if (bbox := bbox_for_cells(cells_by_color[color])) is not None
    }


def grid_summary(grid: list[list[int]]) -> dict[str, Any]:
    counts = color_counts(grid)
    nonzero_cells = [
        (row_index, col_index)
        for row_index, row in enumerate(grid)
        for col_index, color in enumerate(row)
        if color != 0
    ]
    return {
        "shape": grid_shape(grid),
        "colors": [int(color) for color in counts],
        "color_counts": counts,
        "nonzero_count": len(nonzero_cells),
        "nonzero_bbox": bbox_for_cells(nonzero_cells),
        "color_bboxes": bboxes_by_color(grid),
    }


def changed_cells(input_grid: list[list[int]], output_grid: list[list[int]]) -> int | None:
    if grid_shape(input_grid) != grid_shape(output_grid):
        return None
    return sum(
        1
        for row_index, row in enumerate(input_grid)
        for col_index, value in enumerate(row)
        if output_grid[row_index][col_index] != value
    )


def pair_summary(
    split: str,
    index: int,
    pair: dict[str, Any],
    include_grids: bool,
) -> dict[str, Any]:
    input_grid = pair["input"]
    output_grid = pair["output"]
    input_summary = grid_summary(input_grid)
    output_summary = grid_summary(output_grid)
    input_colors = set(input_summary["colors"])
    output_colors = set(output_summary["colors"])
    row: dict[str, Any] = {
        "split": split,
        "index": index,
        "input": input_summary,
        "output": output_summary,
        "shape_changed": input_summary["shape"] != output_summary["shape"],
        "colors_added": sorted(output_colors - input_colors),
        "colors_removed": sorted(input_colors - output_colors),
        "changed_cells": changed_cells(input_grid, output_grid),
    }
    if include_grids:
        row["input_grid"] = input_grid
        row["output_grid"] = output_grid
    return row


def split_names(value: str) -> list[str]:
    if value == "all":
        return list(SPLITS)
    names = [part.strip() for part in value.split(",") if part.strip()]
    invalid = [name for name in names if name not in SPLITS]
    if invalid:
        raise argparse.ArgumentTypeError(f"invalid split(s): {', '.join(invalid)}")
    return names


def task_path(args: argparse.Namespace) -> Path:
    if args.json_file:
        return args.json_file.resolve()
    return (args.data_dir / f"{args.task_id}.json").resolve()


def render_grid(grid: list[list[int]]) -> str:
    return "\n".join(" ".join(str(int(cell)) for cell in row) for row in grid)


def render_text_report(
    args: argparse.Namespace,
    path: Path,
    data: dict[str, Any],
    selected_splits: list[str],
    selected_pairs: list[tuple[str, int, dict[str, Any]]],
) -> str:
    lines: list[str] = []
    lines.append(f"task_id: {args.task_id}")
    lines.append(f"path: {posix_path(path)}")
    split_counts = ", ".join(f"{split}={len(data.get(split, []))}" for split in SPLITS)
    lines.append(f"split_counts: {split_counts}")
    lines.append(f"selected_splits: {','.join(selected_splits)}")
    lines.append(f"pairs_shown: {len(selected_pairs)}")
    lines.append("")
    for split, index, pair in selected_pairs:
        input_grid = pair["input"]
        output_grid = pair["output"]
        in_shape = grid_shape(input_grid)
        out_shape = grid_shape(output_grid)
        lines.append(f"## {split}[{index}]  {in_shape[0]}x{in_shape[1]} -> {out_shape[0]}x{out_shape[1]}")
        lines.append("input:")
        lines.append(render_grid(input_grid))
        lines.append("output:")
        lines.append(render_grid(output_grid))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def collect_selected_pairs(
    data: dict[str, Any],
    selected_splits: list[str],
    max_pairs: int | None,
) -> list[tuple[str, int, dict[str, Any]]]:
    collected: list[tuple[str, int, dict[str, Any]]] = []
    for split in selected_splits:
        for index, pair in enumerate(data.get(split, []), start=1):
            if max_pairs is not None and len(collected) >= max_pairs:
                return collected
            collected.append((split, index, pair))
    return collected


def command_inspect(args: argparse.Namespace) -> int:
    path = task_path(args)
    if not path.exists():
        raise FileNotFoundError(
            f"{posix_path(path)} does not exist; fetch it with the leaderboard skill task-files command"
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    selected_splits = split_names(args.splits)
    selected_pairs = collect_selected_pairs(data, selected_splits, args.max_pairs)

    if not args.json:
        sys.stdout.write(render_text_report(args, path, data, selected_splits, selected_pairs))
        return 0

    pairs = [pair_summary(split, index, pair, args.include_grids) for split, index, pair in selected_pairs]
    all_colors = sorted(
        {
            color
            for pair in pairs
            for side in ("input", "output")
            for color in pair[side]["colors"]
        }
    )
    result = {
        "task_id": args.task_id,
        "path": posix_path(path),
        "split_counts": {split: len(data.get(split, [])) for split in SPLITS},
        "selected_splits": selected_splits,
        "colors": all_colors,
        "input_shapes": sorted({tuple(pair["input"]["shape"]) for pair in pairs}),
        "output_shapes": sorted({tuple(pair["output"]["shape"]) for pair in pairs}),
        "pairs": pairs,
    }
    print_json(result)
    return 0


def existing_file(value: str) -> Path:
    path = Path(value)
    if not path.exists():
        raise argparse.ArgumentTypeError(f"file does not exist: {value}")
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"not a file: {value}")
    return path


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected an integer: {value!r}")
    if parsed < 1:
        raise argparse.ArgumentTypeError("expected an integer >= 1")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect a NeuroGolf task JSON file.")
    parser.add_argument("task_id", type=normalize_task_id, help="Task id like task123 or 123.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Directory containing taskXXX.json files.",
    )
    parser.add_argument(
        "--json-file",
        type=existing_file,
        help="Inspect this JSON file instead of --data-dir/taskXXX.json.",
    )
    parser.add_argument(
        "--splits",
        default="train,test",
        help="Splits to inspect: all, train, test, arc-gen, or comma-separated names.",
    )
    parser.add_argument(
        "--max-pairs",
        type=positive_int,
        help="Maximum number of pairs to include across selected splits.",
    )
    parser.add_argument(
        "--include-grids",
        action="store_true",
        help="JSON mode only: include raw input/output grids alongside summaries.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the legacy structured JSON summary instead of text grids.",
    )
    parser.set_defaults(func=command_inspect)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        return args.func(args)
    except Exception as exc:
        print_json({"error": str(exc), "type": type(exc).__name__})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
