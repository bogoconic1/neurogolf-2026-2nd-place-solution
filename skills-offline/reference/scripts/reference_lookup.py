from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MAPPING_PATH = ROOT / "mapping.csv"


def print_json(value: Any) -> None:
    print(json.dumps(value, indent=2))


def normalize_task_id(value: str) -> str:
    text = value.strip().lower()
    if text.startswith("task"):
        suffix = text[4:]
    else:
        suffix = text
    if not suffix.isdigit():
        raise argparse.ArgumentTypeError(f"expected task id like task123 or 123: {value!r}")
    number = int(suffix)
    if not 1 <= number <= 400:
        raise argparse.ArgumentTypeError("expected task number from 1 to 400")
    return f"task{number:03d}"


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def file_entry(path: Path) -> dict[str, Any]:
    return {
        "path": relative(path),
        "exists": path.exists(),
    }


def load_mapping() -> dict[str, dict[str, str]]:
    with MAPPING_PATH.open(newline="", encoding="utf-8") as handle:
        return {row["neurogolf_task_id"]: row for row in csv.DictReader(handle)}


def command_lookup(args: argparse.Namespace) -> int:
    mapping = load_mapping()
    row = mapping[args.task_id]
    arc_gen_id = row["arc_gen_task_id"]
    arc_dsl_id = row["arc_dsl_task_id"]
    result = {
        "neurogolf_task_id": row["neurogolf_task_id"],
        "arc_gen_task_id": arc_gen_id,
        "arc_dsl_task_id": arc_dsl_id,
        "references": {
            "arc_dsl_solver": file_entry(
                ROOT / "scripts" / "arc-dsl" / "solver_scripts" / f"solve_{arc_dsl_id}.py"
            ),
            "arc_gen_generator": file_entry(
                ROOT / "scripts" / "ARC-GEN" / "tasks" / f"task_{arc_gen_id}.py"
            ),
            "code_golf_2025_solution": file_entry(
                ROOT / "scripts" / "NeurIPS-Code-Golf-2025" / "solutions" / f"{row['neurogolf_task_id']}.py"
            ),
        },
    }
    print_json(result)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Look up NeuroGolf reference files for a task.")
    parser.add_argument("task_id", type=normalize_task_id, help="Task id like task123 or 123.")
    parser.set_defaults(func=command_lookup)
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
