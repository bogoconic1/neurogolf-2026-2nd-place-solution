#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

# Resolve paths relative to this script so the skill bundle is portable
# wherever it is unzipped (scripts/ -> skill dir -> skills bundle root).
SKILL_DIR = Path(__file__).resolve().parent.parent
SKILLS_ROOT = SKILL_DIR.parent
DEFAULT_CACHE_DIR = SKILL_DIR / 'cache'
REFERENCE_DIR = SKILLS_ROOT / 'reference'
MAPPING_PATH = REFERENCE_DIR / 'mapping.csv'
TASK_RE = re.compile(r'^task(\d{3})$', re.IGNORECASE)
REQUIRED_MARKERS = [
    '## Pattern',
    '## Readable Python Solver',
    'def solve',
    '## Generator Constraints',
    '## Reference Notes',
    '## Optimization Implications',
]


def normalize_task_id(value: str) -> str:
    value = (value or '').strip()
    if value.isdigit():
        number = int(value)
    else:
        match = TASK_RE.match(value)
        if not match:
            raise argparse.ArgumentTypeError(f'invalid task id: {value!r}')
        number = int(match.group(1))
    if not 1 <= number <= 400:
        raise argparse.ArgumentTypeError(f'task id out of range: {value!r}')
    return f'task{number:03d}'


def print_json(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def load_mapping() -> dict[str, dict[str, str]]:
    with MAPPING_PATH.open(newline='', encoding='utf-8') as handle:
        return {row['neurogolf_task_id']: row for row in csv.DictReader(handle)}


def relative_to_root(path: Path) -> str:
    try:
        return str(path.relative_to(SKILLS_ROOT)).replace('\\', '/')
    except ValueError:
        return str(path)


def file_entry(path: Path) -> dict[str, Any]:
    return {
        'path': str(path),
        'relative_path': relative_to_root(path),
        'exists': path.exists(),
    }


def reference_info(task_id: str) -> dict[str, Any]:
    row = load_mapping()[task_id]
    arc_gen_id = row['arc_gen_task_id']
    arc_dsl_id = row['arc_dsl_task_id']
    return {
        'neurogolf_task_id': task_id,
        'arc_gen_task_id': arc_gen_id,
        'arc_dsl_task_id': arc_dsl_id,
        'references': {
            'arc_dsl_solver': file_entry(REFERENCE_DIR / 'scripts' / 'arc-dsl' / 'solver_scripts' / f'solve_{arc_dsl_id}.py'),
            'arc_gen_generator': file_entry(REFERENCE_DIR / 'scripts' / 'ARC-GEN' / 'tasks' / f'task_{arc_gen_id}.py'),
            'code_golf_2025_solution': file_entry(REFERENCE_DIR / 'scripts' / 'NeurIPS-Code-Golf-2025' / 'solutions' / f'{task_id}.py'),
        },
    }


def cache_path(cache_dir: Path, task_id: str) -> Path:
    return cache_dir / f'{task_id}.md'


def template(task_id: str, info: dict[str, Any]) -> str:
    refs = info['references']
    return f"""# {task_id} Semantics

## Sources

- ARC-GEN task id: `{info['arc_gen_task_id']}`
- ARC-DSL task id: `{info['arc_dsl_task_id']}`
- ARC-GEN generator: `{refs['arc_gen_generator']['path']}`
- ARC-DSL solver: `{refs['arc_dsl_solver']['path']}`
- Code Golf 2025 solution: `{refs['code_golf_2025_solution']['path']}`

## Pattern

TODO: Describe the transformation in English, including colors, objects, geometry, output size, and tie-breaking.

## Readable Python Solver

```python
def solve(grid):
    # Readable reference implementation for the task semantics.
    # TODO: Replace with clear Python derived from ARC-GEN, ARC-DSL, and Code Golf 2025.
    raise NotImplementedError
```

## Generator Constraints

TODO: Summarize dimensions, color choices, object counts, random branches, guaranteed invariants, and edge cases from ARC-GEN.

## Reference Notes

TODO: Summarize what ARC-DSL and Code Golf 2025 reveal. Note disagreements or ambiguity explicitly.

## Optimization Implications

TODO: Identify compact semantic state, possible memory floors, promising ONNX families, and shortcuts that would likely be unsafe.
"""


def cmd_status(args: argparse.Namespace) -> int:
    task_id = args.task
    path = cache_path(args.cache_dir, task_id)
    info = reference_info(task_id)
    print_json({'task_id': task_id, 'cache_path': str(path), 'cache_exists': path.exists(), **info})
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    path = cache_path(args.cache_dir, args.task)
    if not path.exists():
        print_json({'ok': False, 'task_id': args.task, 'cache_path': str(path), 'error': 'semantic cache does not exist'})
        return 1
    print(path.read_text(encoding='utf-8'))
    return 0


def cmd_init(args: argparse.Namespace) -> int:
    path = cache_path(args.cache_dir, args.task)
    if path.exists() and not args.overwrite:
        print_json({'ok': True, 'task_id': args.task, 'cache_path': str(path), 'created': False, 'cache_exists': True})
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(template(args.task, reference_info(args.task)), encoding='utf-8')
    print_json({'ok': True, 'task_id': args.task, 'cache_path': str(path), 'created': True})
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    path = cache_path(args.cache_dir, args.task)
    if not path.exists():
        print_json({'ok': False, 'task_id': args.task, 'cache_path': str(path), 'error': 'semantic cache does not exist'})
        return 1
    text = path.read_text(encoding='utf-8')
    missing = [marker for marker in REQUIRED_MARKERS if marker not in text]
    todos = text.count('TODO')
    ok = not missing and todos == 0
    print_json({'ok': ok, 'task_id': args.task, 'cache_path': str(path), 'missing_markers': missing, 'todo_count': todos})
    return 0 if ok else 1


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('task', type=normalize_task_id)
    parser.add_argument('--cache-dir', type=Path, default=DEFAULT_CACHE_DIR)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Manage NeuroGolf per-task semantic cache files.')
    sub = parser.add_subparsers(dest='cmd', required=True)
    for name, func in [('status', cmd_status), ('show', cmd_show), ('validate', cmd_validate)]:
        child = sub.add_parser(name)
        add_common(child)
        child.set_defaults(func=func)
    init = sub.add_parser('init')
    add_common(init)
    init.add_argument('--overwrite', action='store_true')
    init.set_defaults(func=cmd_init)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == '__main__':
    raise SystemExit(main())
