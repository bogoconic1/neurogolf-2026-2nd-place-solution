"""Evaluate NeuroGolf zip, ONNX, or Python candidates against local task examples."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import importlib.util
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import zipfile
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "neurogolf-matplotlib-cache"))

import numpy as np
import onnx
import onnxruntime
from google.protobuf.message import DecodeError

onnxruntime.set_default_logger_severity(4)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_REPO_ROOT = SCRIPT_DIR if (SCRIPT_DIR / "mapping.csv").exists() else SCRIPT_DIR.parent
REPO_ROOT = Path(os.environ.get("NEUROGOLF_REPO_ROOT", DEFAULT_REPO_ROOT)).resolve()
DEFAULT_SUBMISSION = Path("submission.zip")
DATA_DIR = REPO_ROOT / "neurogolf-2026"
UTILS_PATH = DATA_DIR / "neurogolf_utils" / "neurogolf_utils.py"
ARC_GEN_DIR = REPO_ROOT / "ARC-GEN"
MAPPING_PATH = REPO_ROOT / "mapping.csv"
DEFAULT_WORKERS = 8
# Static (train+test+arc-gen) miss-rate at/above this marks a submission `wrong`; below it the
# model is kept `unsafe` (a tiny public miss can still pass on Kaggle). Matches the live website.
STATIC_WRONG_THRESHOLD = 0.05
_ARC_GEN_TASK_LIST = None
_TASK_ID_MAPPING: dict[str, str] | None = None
CONVERT_TIMEOUT_SECONDS = int(os.environ.get("NEUROGOLF_CONVERT_TIMEOUT", "180"))

CONVERTER_CODE = r"""
import importlib.util
import shutil
import sys
from pathlib import Path

import onnx

module_path = Path(sys.argv[1]).resolve()
out_path = Path(sys.argv[2]).resolve()
task_id = sys.argv[3]
work_root = Path(sys.argv[4]).resolve()
out_path.parent.mkdir(parents=True, exist_ok=True)

spec = importlib.util.spec_from_file_location("candidate_solution", module_path)
if spec is None or spec.loader is None:
    raise RuntimeError(f"unable to load {module_path}")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

if hasattr(module, "save_model"):
    module.save_model(out_path)
elif hasattr(module, "build_model"):
    onnx.save_model(module.build_model(), out_path)
else:
    candidates = []
    for path in [
        work_root / "best_onnx" / f"{task_id}.onnx",
        module_path.with_suffix(".onnx"),
    ]:
        if path.exists():
            candidates.append(path)
    candidates.extend(work_root.rglob(f"{task_id}.onnx"))
    candidates = sorted(set(candidates), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise RuntimeError(
            "candidate did not define save_model/build_model and did not write "
            f"{task_id}.onnx"
        )
    shutil.copy2(candidates[0], out_path)

if not out_path.exists():
    raise RuntimeError(f"conversion did not produce {out_path}")
onnx.checker.check_model(str(out_path))
print(out_path)
"""


@dataclass
class TaskResult:
    task_id: str
    status: str
    file_size_bytes: int | None = None
    arc_agi_pass: int = 0
    arc_agi_fail: int = 0
    arc_gen_pass: int = 0
    arc_gen_fail: int = 0
    generated_examples: int = 0
    new_samples_pass: int = 0
    new_samples_fail: int = 0
    skipped_examples: int = 0
    memory_bytes: int | None = None
    params: int | None = None
    points: float | None = None
    error: str = ""
    mismatch_sample: str = ""
    memory_breakdown: dict[str, int] | None = None

    @property
    def ready(self) -> bool:
        return (
            self.status == "ok"
            and self.kaggle_ready
            and self.new_samples_fail == 0
        )

    @property
    def static_fail_rate(self) -> float:
        total = (self.arc_agi_pass + self.arc_agi_fail
                 + self.arc_gen_pass + self.arc_gen_fail)
        return ((self.arc_agi_fail + self.arc_gen_fail) / total) if total else 0.0

    @property
    def kaggle_ready(self) -> bool:
        # A tiny static miss (< 5% of train+test+arc-gen) is tolerated — matches the live website.
        return (
            self.static_fail_rate < STATIC_WRONG_THRESHOLD
            and self.memory_bytes is not None
            and self.params is not None
            and self.memory_bytes >= 0
            and self.params >= 0
        )

    @property
    def unsafe(self) -> bool:
        return self.status == "unsafe"


def compact_json(data: dict[str, object]) -> str:
    return json.dumps(data, separators=(",", ":"), sort_keys=True)


def error_json(error_type: str, message: str, **fields: object) -> str:
    return compact_json(
        {
            "type": error_type,
            "message": message,
            **{key: value for key, value in fields.items() if value is not None},
        }
    )


def parse_json_or_none(value: str) -> object | None:
    if not value:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{sec:02d}"
    return f"{minutes:d}:{sec:02d}"


def tail_text(value: object, limit: int = 4000) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        text = value.decode(errors="replace")
    else:
        text = str(value)
    return text[-limit:]


class ProgressBar:
    def __init__(self, total: int, enabled: bool, label: str = "Evaluating") -> None:
        self.total = total
        self.enabled = enabled and total > 0
        self.label = label
        self.completed = 0
        self.started_at = time.perf_counter()
        self.last_message_length = 0
        if self.enabled:
            self.render()

    def advance(self, suffix: str = "", step: int = 1) -> None:
        if not self.enabled:
            return
        self.completed = min(self.total, self.completed + step)
        self.render(suffix)

    def update(self, result: TaskResult) -> None:
        self.advance(f"{result.task_id}:{result.status}")

    def render(self, suffix: str = "") -> None:
        elapsed = time.perf_counter() - self.started_at
        rate = self.completed / elapsed if elapsed else 0.0
        remaining = self.total - self.completed
        eta = remaining / rate if rate else 0.0
        width = 30
        filled = round(width * self.completed / self.total)
        bar = "#" * filled + "-" * (width - filled)
        message = (
            f"\r{self.label} [{bar}] {self.completed}/{self.total} "
            f"elapsed {format_duration(elapsed)} eta {format_duration(eta)}"
        )
        if suffix:
            message += f" {suffix}"
        padding = " " * max(0, self.last_message_length - len(message))
        sys.stderr.write("\r" + message + padding)
        sys.stderr.flush()
        self.last_message_length = len(message)

    def finish(self) -> None:
        if self.enabled:
            sys.stderr.write("\n")
            sys.stderr.flush()


def load_neurogolf_utils():
    spec = importlib.util.spec_from_file_location("neurogolf_utils", UTILS_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load neurogolf_utils from {UTILS_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


NG = load_neurogolf_utils()


def normalize_task_id(raw_task: str) -> str:
    task = raw_task.strip()
    if not task:
        raise ValueError("Empty task id")
    if task.endswith(".onnx") or task.endswith(".json"):
        task = Path(task).stem
    if task.startswith("task"):
        suffix = task[4:]
    else:
        suffix = task
    if not suffix.isdigit():
        raise ValueError(f"Invalid task id: {raw_task!r}")
    task_num = int(suffix)
    if task_num < 1 or task_num > 400:
        raise ValueError(f"Task id out of range 1..400: {raw_task!r}")
    return f"task{task_num:03d}"


def expand_task_token(raw_task: str) -> list[str]:
    token = raw_task.strip()
    if not token:
        return []
    range_match = re.fullmatch(r"(?:task)?(\d{1,3})-(?:task)?(\d{1,3})", token, flags=re.IGNORECASE)
    if not range_match:
        return [normalize_task_id(token)]
    start = int(range_match.group(1))
    end = int(range_match.group(2))
    step = 1 if end >= start else -1
    return [normalize_task_id(str(number)) for number in range(start, end + step, step)]


def parse_task_args(tasks: list[str] | None, task_file: Path | None, default_all: bool = True) -> list[str]:
    selected: list[str] = []
    if task_file:
        selected.extend(task_file.read_text().replace(",", "\n").split())
    if tasks:
        for token in tasks:
            selected.extend(token.replace(",", " ").split())

    if not selected and default_all:
        return [path.stem for path in sorted(DATA_DIR.glob("task*.json"))]
    if not selected:
        return []

    normalized = []
    seen = set()
    for task in selected:
        for task_id in expand_task_token(task):
            if task_id not in seen:
                seen.add(task_id)
                normalized.append(task_id)
    return normalized


def infer_task_id_from_path(path: Path) -> str | None:
    for text in (path.stem, path.parent.name):
        match = re.search(r"task\d{1,3}", text, flags=re.IGNORECASE)
        candidate = match.group(0) if match else text
        try:
            return normalize_task_id(candidate)
        except ValueError:
            continue
    return None


def classify_submission_input(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".zip":
        return "zip"
    if suffix == ".onnx":
        return "onnx"
    if suffix == ".py":
        return "python"
    if zipfile.is_zipfile(path):
        return "zip"
    raise ValueError(
        f"unsupported input type for {path}; expected .zip, .onnx, or .py"
    )


def resolve_tasks_for_input(
    tasks: list[str] | None,
    task_file: Path | None,
    input_path: Path,
    input_kind: str,
) -> list[str]:
    selected = parse_task_args(tasks, task_file, default_all=input_kind == "zip")
    if input_kind == "zip":
        return selected
    if len(selected) > 1:
        raise ValueError("single .onnx/.py inputs can be evaluated for exactly one task")
    if selected:
        return selected
    inferred = infer_task_id_from_path(input_path)
    if inferred is None:
        raise ValueError(
            "single .onnx/.py input requires --tasks taskXXX unless the file or parent "
            "directory name contains a task id"
        )
    return [inferred]


def load_examples(task_id: str) -> dict[str, list[dict[str, list[list[int]]]]]:
    with (DATA_DIR / f"{task_id}.json").open() as f:
        return json.load(f)


def load_task_id_mapping() -> dict[str, str]:
    global _TASK_ID_MAPPING
    if _TASK_ID_MAPPING is None:
        with MAPPING_PATH.open(newline="") as f:
            _TASK_ID_MAPPING = {
                row["neurogolf_task_id"]: row["arc_gen_task_id"]
                for row in csv.DictReader(f)
            }
    return _TASK_ID_MAPPING


def load_arc_gen_task_list():
    global _ARC_GEN_TASK_LIST
    if _ARC_GEN_TASK_LIST is None:
        if str(ARC_GEN_DIR) not in sys.path:
            sys.path.insert(0, str(ARC_GEN_DIR))
        import task_list

        _ARC_GEN_TASK_LIST = task_list.task_list()
    return _ARC_GEN_TASK_LIST


def canonical_example(example: dict) -> str:
    return json.dumps(example, sort_keys=True, separators=(",", ":"))


def all_task_examples(examples: dict[str, list[dict]]) -> list[dict]:
    return examples["train"] + examples["test"] + examples["arc-gen"]


def generation_seed(seed: int | None, task_id: str, attempt: int) -> int:
    if seed is None:
        return random.SystemRandom().randrange(2**63)
    digest = hashlib.sha256(f"{seed}:{task_id}:{attempt}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def generate_new_examples(
    task_id: str,
    count: int,
    existing_examples: list[dict],
    on_generated=None,
    seed: int | None = None,
) -> list[dict]:
    if count <= 0:
        return []

    task_mapping = load_task_id_mapping()
    arc_gen_task_id = task_mapping[task_id]
    generator, _ = load_arc_gen_task_list()[arc_gen_task_id]

    seen = {canonical_example(example) for example in existing_examples}
    generated = []
    max_attempts = max(100, count * 100)
    for attempt in range(max_attempts):
        # By default, keep generation fresh on every evaluator run. With --seed,
        # make each task/attempt draw deterministic for reproducible stress evals.
        random.seed(generation_seed(seed, task_id, attempt))
        try:
            example = generator()
        except ValueError:
            # Some ARC-GEN generators occasionally draw invalid random parameters.
            # Discard only that draw and keep trying to produce the requested count.
            continue
        key = canonical_example(example)
        if key in seen:
            continue
        seen.add(key)
        generated.append(example)
        if on_generated is not None:
            on_generated(task_id, len(generated))
        if len(generated) == count:
            return generated

    return generated


def generate_new_examples_for_tasks(
    tasks: list[str],
    count: int,
    progress: ProgressBar,
    seed: int | None = None,
) -> dict[str, list[dict]]:
    if count <= 0:
        return {}

    generated_by_task: dict[str, list[dict]] = {}

    def on_generated(task_id: str, generated_count: int) -> None:
        progress.advance(f"{task_id}:{generated_count}/{count}")

    try:
        for task_id in tasks:
            examples = load_examples(task_id)
            generated_by_task[task_id] = generate_new_examples(
                task_id,
                count,
                all_task_examples(examples),
                on_generated=on_generated,
                seed=seed,
            )
            missing = count - len(generated_by_task[task_id])
            if missing:
                progress.advance(f"{task_id}:skipped {missing}/{count}", step=missing)
    finally:
        progress.finish()

    return generated_by_task


def zip_entries_by_basename(submission: Path) -> dict[str, zipfile.ZipInfo]:
    with zipfile.ZipFile(submission) as archive:
        entries: dict[str, zipfile.ZipInfo] = {}
        for entry in archive.infolist():
            if entry.is_dir() or not entry.filename.endswith(".onnx"):
                continue
            name = Path(entry.filename).name
            if name in entries:
                raise ValueError(f"Duplicate ONNX filename in zip: {name}")
            entries[name] = entry
    return entries


def extract_model(
    archive: zipfile.ZipFile,
    entry: zipfile.ZipInfo,
    task_id: str,
    work_dir: Path,
) -> Path:
    model_path = work_dir / f"{task_id}.onnx"
    with archive.open(entry) as src, model_path.open("wb") as dst:
        dst.write(src.read())
    return model_path


def _stage_path(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        return
    if source.is_dir():
        try:
            destination.symlink_to(source, target_is_directory=True)
        except OSError:
            shutil.copytree(
                source,
                destination,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def stage_converter_context(work_root: Path, task_id: str) -> None:
    for filename in ("mapping.csv",):
        source = REPO_ROOT / filename
        if source.exists():
            _stage_path(source, work_root / filename)

    task_json = DATA_DIR / f"{task_id}.json"
    if task_json.exists():
        _stage_path(task_json, work_root / "neurogolf-2026" / task_json.name)

    utils_dir = DATA_DIR / "neurogolf_utils"
    if utils_dir.exists():
        _stage_path(utils_dir, work_root / "neurogolf-2026" / "neurogolf_utils")

    for dirname in (
        "ARC-GEN",
        "arc-dsl",
        "submission_code",
        "submission_helper_code",
        "submission_helper_code_test",
    ):
        source = REPO_ROOT / dirname
        if source.exists():
            _stage_path(source, work_root / dirname)


def convert_python_candidate(
    py_source: Path,
    task_id: str,
    candidate_dir: Path,
    onnx_path: Path,
) -> subprocess.CompletedProcess[str]:
    work_root = candidate_dir / "workroot"
    staged_dir = work_root / "src" / task_id
    staged_dir.mkdir(parents=True, exist_ok=True)
    staged_py = staged_dir / py_source.name
    shutil.copy2(py_source, staged_py)
    stage_converter_context(work_root, task_id)
    (work_root / "best_onnx").mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        str(work_root)
        + os.pathsep
        + str(REPO_ROOT)
        + os.pathsep
        + env.get("PYTHONPATH", "")
    )
    return subprocess.run(
        [sys.executable, "-c", CONVERTER_CODE, str(staged_py), str(onnx_path), task_id, str(work_root)],
        cwd=work_root,
        env=env,
        text=True,
        capture_output=True,
        timeout=CONVERT_TIMEOUT_SECONDS,
    )


def verify_subset(
    session: onnxruntime.InferenceSession, examples: list[dict], subset_name: str
) -> tuple[int, int, int, str, str]:
    right, wrong, skipped = 0, 0, 0
    first_error = ""
    first_mismatch = ""
    for index, example in enumerate(examples):
        benchmark = NG.convert_to_numpy(example)
        if benchmark is None:
            skipped += 1
            continue
        try:
            user_output = NG.run_network(session, benchmark["input"])
        except Exception:
            trace = traceback.format_exc(limit=3).strip()
            wrong += 1
            if not first_error:
                first_error = error_json(
                    "python_eval_error",
                    f"{subset_name}[{index}] runtime error",
                    subset=subset_name,
                    index=index,
                    traceback=trace,
                )
            if not first_mismatch:
                first_mismatch = compact_json(
                    {
                        "subset": subset_name,
                        "index": index,
                        "input": example["input"],
                        "expected": example["output"],
                        "actual": None,
                        "runtime_error": trace,
                    },
                )
            continue
        if np.array_equal(user_output, benchmark["output"]):
            right += 1
        else:
            wrong += 1
            if not first_error:
                first_error = error_json(
                    "wrong_sample",
                    f"{subset_name}[{index}] output mismatch",
                    subset=subset_name,
                    index=index,
                    expected_shape=list(benchmark["output"].shape),
                    actual_shape=list(user_output.shape),
                )
            if not first_mismatch:
                first_mismatch = compact_json(
                    {
                        "subset": subset_name,
                        "index": index,
                        "input": example["input"],
                        "expected": example["output"],
                        "actual": NG.convert_from_numpy(user_output),
                    },
                )
    return right, wrong, skipped, first_error, first_mismatch


def failure_distribution(result: TaskResult) -> dict[str, str]:
    """Per-subset fail/total, e.g. {"arc_gen": "1/261"} — the failed-test distribution."""
    dist: dict[str, str] = {}
    if result.arc_agi_fail:
        dist["arc_agi"] = f"{result.arc_agi_fail}/{result.arc_agi_pass + result.arc_agi_fail}"
    if result.arc_gen_fail:
        dist["arc_gen"] = f"{result.arc_gen_fail}/{result.arc_gen_pass + result.arc_gen_fail}"
    if result.new_samples_fail:
        dist["fresh"] = f"{result.new_samples_fail}/{result.new_samples_pass + result.new_samples_fail}"
    return dist


def wrong_result_error(result: TaskResult, mismatch_sample: str) -> str:
    failures: dict[str, int] = {}
    if result.arc_agi_fail:
        failures["arc_agi"] = result.arc_agi_fail
    if result.arc_gen_fail:
        failures["arc_gen"] = result.arc_gen_fail
    if result.new_samples_fail:
        failures["fresh"] = result.new_samples_fail
    return error_json(
        "wrong_sample",
        "Wrong results",
        failures=failures,
        failed=failure_distribution(result),
        sample=parse_json_or_none(mismatch_sample),
    )


def unsafe_result_error(result: TaskResult, mismatch_sample: str) -> str:
    return error_json(
        "unsafe_champion",
        "Kept as unsafe: static miss-rate < 5% (Kaggle may still pass) and/or fresh ARC-GEN failures",
        failures={"fresh": result.new_samples_fail},
        failed=failure_distribution(result),
        generated_examples=result.generated_examples,
        sample=parse_json_or_none(mismatch_sample),
    )


def prepare_session(model_path: Path, task_id: str, trace_dir: Path) -> tuple[onnx.ModelProto, onnxruntime.InferenceSession]:
    if model_path.stat().st_size > NG._FILESIZE_LIMIT_IN_BYTES:
        raise ValueError(
            f"File size {model_path.stat().st_size} exceeds "
            f"{int(NG._FILESIZE_LIMIT_IN_BYTES)} bytes"
        )

    sanitized = onnx.load(model_path)
    banned_ops = sorted(
        {node.op_type.upper() for node in sanitized.graph.node}
        & {op.upper() for op in NG._EXCLUDED_OP_TYPES}
    )
    if banned_ops:
        raise ValueError(f"Disallowed ONNX ops: {', '.join(banned_ops)}")

    for node in sanitized.graph.node:
        if not node.output:
            raise ValueError(f"Node {node.name!r} has no outputs")
    # Canonical sanitizer (renames node+tensor names to safe_name_N) so profiler-trace
    # events match nodes by name. The old node.name=output[0] left original tensor names,
    # which ORT normalizes in the trace -> data-dependent tensors' runtime memory was
    # dropped -> under-counted memory / inflated score vs Kaggle.
    sanitized = NG.sanitize_model(sanitized)
    if sanitized is None:
        raise ValueError("Disallowed tensor/node name: 'kernel_time'")

    options = onnxruntime.SessionOptions()
    options.enable_profiling = True
    options.log_severity_level = 4
    options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.profile_file_prefix = str(trace_dir / task_id)
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = onnxruntime.InferenceSession(
        sanitized.SerializeToString(),
        options,
        providers=["CPUExecutionProvider"],
    )
    return sanitized, session


def calculate_memory_breakdown(model: onnx.ModelProto, trace_path: str) -> dict[str, int] | None:
    onnx.checker.check_model(model, full_check=True)
    graph = onnx.shape_inference.infer_shapes(model, strict_mode=True).graph
    if len(graph.input) > 1 or len(graph.output) > 1:
        return None
    init_names = {init.name for init in graph.initializer}
    init_names.update(init.name for init in graph.sparse_initializer)
    io_names = {tensor.name for tensor in list(graph.input) + list(graph.output)}
    if io_names.intersection(init_names):
        return None
    if model.functions:
        return None
    for opset in model.opset_import:
        if opset.domain not in {"", "ai.onnx"}:
            return None

    node_outputs: dict[str, list[str]] = {}
    tensor_names: set[str] = set()
    for node in graph.node:
        for attr in node.attribute:
            if attr.type in [onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS]:
                return None
        node_outputs[node.name] = list(node.output)
        for output_name in node.output:
            if output_name:
                tensor_names.add(output_name)

    tensor_memory: dict[str, int] = {}
    tensor_dtypes: dict[str, np.dtype] = {}
    tensor_map = {
        tensor.name: tensor
        for tensor in list(graph.input) + list(graph.value_info) + list(graph.output)
    }
    tensor_names.update(tensor_map.keys())
    for tensor_name in tensor_names:
        item = tensor_map.get(tensor_name)
        if not item:
            return None
        if item.type.HasField("sequence_type"):
            return None
        if not item.type.HasField("tensor_type"):
            continue
        tensor_type = item.type.tensor_type
        if not tensor_type.HasField("shape"):
            return None
        num_elements = 1
        for dim in tensor_type.shape.dim:
            if dim.HasField("dim_param"):
                return None
            if not dim.HasField("dim_value"):
                return None
            if dim.dim_value <= 0:
                return None
            num_elements *= dim.dim_value
        if tensor_name in ["input", "output"]:
            continue
        np_dtype = onnx.helper.tensor_dtype_to_np_dtype(tensor_type.elem_type)
        tensor_memory[tensor_name] = num_elements * np.dtype(np_dtype).itemsize
        tensor_dtypes[tensor_name] = np.dtype(np_dtype)

    seen: set[str] = set()
    for item in list(graph.input) + list(graph.value_info) + list(graph.output):
        if item.name in seen:
            return None
        seen.add(item.name)
    for node in graph.node:
        for output_name in node.output:
            if output_name and output_name != "output":
                item = tensor_map.get(output_name)
                if item is None or not item.type.HasField("tensor_type"):
                    return None

    with open(trace_path, encoding="utf-8") as trace_file:
        trace_data = json.load(trace_file)
    for event in trace_data:
        if event.get("cat") != "Node" or "args" not in event:
            continue
        if "output_type_shape" not in event["args"]:
            continue
        node_name = event.get("name", "").replace("_kernel_time", "")
        if node_name not in node_outputs:
            continue
        for index, shape_dict in enumerate(event["args"]["output_type_shape"]):
            if index >= len(node_outputs[node_name]):
                continue
            output_name = node_outputs[node_name][index]
            if output_name not in tensor_dtypes:
                continue
            itemsize = np.dtype(tensor_dtypes[output_name]).itemsize
            mem = itemsize * sum(math.prod(dims) for dims in shape_dict.values())
            tensor_memory[output_name] = max(tensor_memory[output_name], mem)
    return tensor_memory


def remove_profile_trace(trace_path: str | None) -> None:
    if trace_path:
        Path(trace_path).unlink(missing_ok=True)


def evaluate_task(
    archive: zipfile.ZipFile,
    entries: dict[str, zipfile.ZipInfo],
    task_id: str,
    work_dir: Path,
    fresh_examples: list[dict] | None = None,
    collect_memory_breakdown: bool = False,
) -> TaskResult:
    result = TaskResult(task_id=task_id, status="ok")
    entry = entries.get(f"{task_id}.onnx")
    if entry is None:
        result.status = "missing"
        result.error = error_json("python_eval_error", f"{task_id}.onnx not found in submission zip")
        return result

    model_path = extract_model(archive, entry, task_id, work_dir)
    return evaluate_model_path(
        model_path,
        task_id,
        work_dir,
        fresh_examples,
        collect_memory_breakdown,
    )


def evaluate_model_path(
    model_path: Path,
    task_id: str,
    work_dir: Path,
    fresh_examples: list[dict] | None = None,
    collect_memory_breakdown: bool = False,
) -> TaskResult:
    result = TaskResult(task_id=task_id, status="ok")
    result.file_size_bytes = model_path.stat().st_size
    session: onnxruntime.InferenceSession | None = None
    trace_path: str | None = None
    profiling_active = False
    try:
        sanitized, session = prepare_session(model_path, task_id, work_dir)
        profiling_active = True
        examples = load_examples(task_id)
        fresh_examples = fresh_examples or []
        result.generated_examples = len(fresh_examples)

        arc_agi = examples["train"] + examples["test"]
        (
            result.arc_agi_pass,
            result.arc_agi_fail,
            skipped_agi,
            arc_agi_error,
            arc_agi_mismatch,
        ) = verify_subset(session, arc_agi, "ARC-AGI")
        (
            result.arc_gen_pass,
            result.arc_gen_fail,
            skipped_gen,
            arc_gen_error,
            arc_gen_mismatch,
        ) = verify_subset(session, examples["arc-gen"], "ARC-GEN")

        trace_path = session.end_profiling()
        profiling_active = False
        if collect_memory_breakdown:
            result.memory_breakdown = calculate_memory_breakdown(sanitized, trace_path)
        memory, params = NG.score_network(sanitized, trace_path)
        remove_profile_trace(trace_path)
        trace_path = None
        result.memory_bytes = memory
        result.params = params
        if memory is None or params is None or memory < 0 or params < 0:
            result.status = "metric_error"
            result.skipped_examples = skipped_agi + skipped_gen
            result.error = arc_agi_error or arc_gen_error or error_json(
                "python_eval_error", "Network performance could not be measured"
            )
            return result

        (
            result.new_samples_pass,
            result.new_samples_fail,
            skipped_new,
            new_samples_error,
            new_samples_mismatch,
        ) = verify_subset(session, fresh_examples, "fresh")
        session = None
        result.skipped_examples = skipped_agi + skipped_gen + skipped_new

        result.points = max(1.0, 25.0 - math.log(max(1.0, memory + params)))
        # <5% static (train+test+arc-gen) miss -> unsafe (Kaggle may still pass), not wrong. Matches
        # the live website. Only a >=5% static miss is a genuine non-solver (`wrong`, points=None).
        static_total = (result.arc_agi_pass + result.arc_agi_fail
                        + result.arc_gen_pass + result.arc_gen_fail)
        static_fail = result.arc_agi_fail + result.arc_gen_fail
        static_fail_rate = (static_fail / static_total) if static_total else 0.0
        if static_fail and static_fail_rate >= STATIC_WRONG_THRESHOLD:
            result.status = "wrong"
            result.mismatch_sample = arc_agi_mismatch or arc_gen_mismatch
            result.error = wrong_result_error(result, result.mismatch_sample)
            result.points = None
        elif static_fail or result.new_samples_fail:
            result.status = "unsafe"
            result.mismatch_sample = arc_agi_mismatch or arc_gen_mismatch or new_samples_mismatch
            result.error = unsafe_result_error(result, result.mismatch_sample)
        else:
            result.error = arc_agi_error or arc_gen_error or new_samples_error
    except (
        ValueError,
        DecodeError,
        ImportError,
        KeyError,
        OSError,
        RuntimeError,
        onnx.checker.ValidationError,
        onnx.shape_inference.InferenceError,
        Exception,
    ) as exc:
        result.status = "error"
        result.error = error_json(
            "python_eval_error",
            str(exc),
            exception_type=type(exc).__name__,
        )
    finally:
        if session is not None and profiling_active:
            remove_profile_trace(session.end_profiling())
        remove_profile_trace(trace_path)
    return result


def evaluate_single_input(
    input_path: Path,
    input_kind: str,
    task_id: str,
    fresh_examples: list[dict] | None = None,
    collect_memory_breakdown: bool = False,
) -> TaskResult:
    try:
        with tempfile.TemporaryDirectory(prefix=f"neurogolf_eval_{task_id}_") as tmp:
            work_dir = Path(tmp)
            if input_kind == "onnx":
                return evaluate_model_path(
                    input_path,
                    task_id,
                    work_dir,
                    fresh_examples,
                    collect_memory_breakdown,
                )

            onnx_path = work_dir / f"{task_id}.onnx"
            try:
                convert = convert_python_candidate(input_path, task_id, work_dir, onnx_path)
            except subprocess.TimeoutExpired as exc:
                return TaskResult(
                    task_id=task_id,
                    status="error",
                    error=error_json(
                        "python_build_error",
                        f"conversion timed out after {CONVERT_TIMEOUT_SECONDS} seconds",
                        stdout=tail_text(exc.stdout),
                        stderr=tail_text(exc.stderr),
                    ),
                )
            if convert.returncode != 0:
                return TaskResult(
                    task_id=task_id,
                    status="error",
                    error=error_json(
                        "python_build_error",
                        "conversion failed",
                        returncode=convert.returncode,
                        stdout=tail_text(convert.stdout),
                        stderr=tail_text(convert.stderr),
                    ),
                )
            return evaluate_model_path(
                onnx_path,
                task_id,
                work_dir,
                fresh_examples,
                collect_memory_breakdown,
            )
    except (OSError, ValueError, ImportError, RuntimeError) as exc:
        return TaskResult(
            task_id=task_id,
            status="error",
            error=error_json("python_eval_error", str(exc), exception_type=type(exc).__name__),
        )


def evaluate_task_from_submission(
    task_id: str,
    submission: str,
    fresh_examples: list[dict],
    collect_memory_breakdown: bool = False,
) -> TaskResult:
    submission_path = Path(submission)
    try:
        entries = zip_entries_by_basename(submission_path)
        with tempfile.TemporaryDirectory(prefix=f"neurogolf_eval_{task_id}_") as tmp:
            with zipfile.ZipFile(submission_path) as archive:
                return evaluate_task(
                    archive,
                    entries,
                    task_id,
                    Path(tmp),
                    fresh_examples,
                    collect_memory_breakdown,
                )
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        return TaskResult(
            task_id=task_id,
            status="error",
            error=error_json("python_eval_error", str(exc), exception_type=type(exc).__name__),
        )


def evaluate_tasks_sequential(
    tasks: list[str],
    submission: Path,
    entries: dict[str, zipfile.ZipInfo],
    progress: ProgressBar,
    fresh_examples_by_task: dict[str, list[dict]],
    collect_memory_breakdown: bool = False,
) -> list[TaskResult]:
    results: list[TaskResult] = []
    with tempfile.TemporaryDirectory(prefix="neurogolf_eval_") as tmp:
        work_dir = Path(tmp)
        with zipfile.ZipFile(submission) as archive:
            for task_id in tasks:
                result = evaluate_task(
                    archive,
                    entries,
                    task_id,
                    work_dir,
                    fresh_examples_by_task.get(task_id, []),
                    collect_memory_breakdown,
                )
                results.append(result)
                progress.update(result)
    return results


def evaluate_tasks_parallel(
    tasks: list[str],
    submission: Path,
    workers: int,
    progress: ProgressBar,
    fresh_examples_by_task: dict[str, list[dict]],
    collect_memory_breakdown: bool = False,
) -> list[TaskResult]:
    results: list[TaskResult | None] = [None] * len(tasks)
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_index = {
            executor.submit(
                evaluate_task_from_submission,
                task_id,
                str(submission),
                fresh_examples_by_task.get(task_id, []),
                collect_memory_breakdown,
            ): index
            for index, task_id in enumerate(tasks)
        }
        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            task_id = tasks[index]
            try:
                result = future.result()
            except Exception as exc:
                result = TaskResult(
                    task_id=task_id,
                    status="error",
                    error=error_json("python_eval_error", repr(exc), exception_type=type(exc).__name__),
                )
            results[index] = result
            progress.update(result)

    return [result for result in results if result is not None]


def format_int(value: int | None) -> str:
    return "" if value is None else str(value)


def format_float(value: float | None) -> str:
    return "" if value is None else f"{value:.6f}"


RESULT_FIELDNAMES = [
    "task_id",
    "status",
    "ready",
    "kaggle_ready",
    "unsafe",
    "arc_agi_pass",
    "arc_agi_fail",
    "arc_gen_pass",
    "arc_gen_fail",
    "generated_examples",
    "new_samples_pass",
    "new_samples_fail",
    "skipped_examples",
    "memory_bytes",
    "params",
    "points",
    "file_size_bytes",
    "mismatch_sample",
    "error",
]


def result_to_row(result: TaskResult) -> dict[str, object]:
    return {
        "task_id": result.task_id,
        "status": result.status,
        "ready": result.ready,
        "kaggle_ready": result.kaggle_ready,
        "unsafe": result.unsafe,
        "arc_agi_pass": result.arc_agi_pass,
        "arc_agi_fail": result.arc_agi_fail,
        "arc_gen_pass": result.arc_gen_pass,
        "arc_gen_fail": result.arc_gen_fail,
        "generated_examples": result.generated_examples,
        "new_samples_pass": result.new_samples_pass,
        "new_samples_fail": result.new_samples_fail,
        "skipped_examples": result.skipped_examples,
        "memory_bytes": format_int(result.memory_bytes),
        "params": format_int(result.params),
        "points": format_float(result.points),
        "file_size_bytes": format_int(result.file_size_bytes),
        "mismatch_sample": result.mismatch_sample,
        "error": result.error.replace("\n", "\\n"),
    }


def write_results_csv(results: list[TaskResult], output) -> None:
    writer = csv.DictWriter(output, fieldnames=RESULT_FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    for result in results:
        writer.writerow(result_to_row(result))


def print_verbose(results: list[TaskResult]) -> None:
    write_results_csv(results, sys.stdout)


def save_results_csv(results: list[TaskResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        write_results_csv(results, f)


MEMORY_BREAKDOWN_FIELDNAMES = ["task_id", "name", "bytes"]


def write_memory_breakdown_csv(results: list[TaskResult], output) -> None:
    writer = csv.DictWriter(output, fieldnames=MEMORY_BREAKDOWN_FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    for result in results:
        if result.memory_breakdown is None:
            continue
        total = sum(result.memory_breakdown.values())
        writer.writerow({"task_id": result.task_id, "name": "_total", "bytes": total})
        for name, bytes_ in sorted(
            result.memory_breakdown.items(),
            key=lambda item: (-item[1], item[0]),
        ):
            writer.writerow({"task_id": result.task_id, "name": name, "bytes": bytes_})


def save_memory_breakdown(results: list[TaskResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        write_memory_breakdown_csv(results, f)


def print_summary(results: list[TaskResult], submission: Path) -> None:
    ready = [result for result in results if result.ready]
    measured = [result for result in results if result.memory_bytes is not None and result.params is not None]
    total_points = sum(result.points or 0.0 for result in ready)
    total_memory = sum(result.memory_bytes or 0 for result in measured)
    total_params = sum(result.params or 0 for result in measured)
    total_arc_agi_pass = sum(result.arc_agi_pass for result in results)
    total_arc_agi_fail = sum(result.arc_agi_fail for result in results)
    total_arc_gen_pass = sum(result.arc_gen_pass for result in results)
    total_arc_gen_fail = sum(result.arc_gen_fail for result in results)
    total_generated = sum(result.generated_examples for result in results)
    total_new_samples_pass = sum(result.new_samples_pass for result in results)
    total_new_samples_fail = sum(result.new_samples_fail for result in results)

    print(f"Submission: {submission}")
    print(f"Tasks evaluated: {len(results)}")
    print(f"Ready tasks: {len(ready)}")
    print(f"Measured tasks: {len(measured)}")
    print(f"ARC-AGI examples: {total_arc_agi_pass} pass, {total_arc_agi_fail} fail")
    print(f"ARC-GEN examples: {total_arc_gen_pass} pass, {total_arc_gen_fail} fail")
    if total_generated:
        print(
            f"Fresh ARC-GEN examples: {total_new_samples_pass} pass, "
            f"{total_new_samples_fail} fail ({total_generated} generated)"
        )
    print(f"Total measured memory bytes: {total_memory}")
    print(f"Total measured params: {total_params}")
    print(f"Total ready-task points: {total_points:.6f}")

    status_counts: dict[str, int] = {}
    for result in results:
        status_counts[result.status] = status_counts.get(result.status, 0) + 1
    print("Statuses: " + ", ".join(f"{status}={count}" for status, count in sorted(status_counts.items())))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--submission",
        "--input",
        dest="submission",
        type=Path,
        default=DEFAULT_SUBMISSION,
        help=f"Path to a submission zip, one taskXXX.onnx file, or one Python builder file. Defaults to {DEFAULT_SUBMISSION}.",
    )
    parser.add_argument(
        "--tasks",
        nargs="*",
        help="Optional task ids to evaluate, e.g. --tasks task001 2 003 10-15 or --tasks task001,task002.",
    )
    parser.add_argument(
        "--task-file",
        type=Path,
        help="Optional text file containing task ids separated by whitespace, commas, or newlines.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-task metrics as CSV before the summary.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help="Write per-task metrics to this CSV file.",
    )
    parser.add_argument(
        "--memory-breakdown",
        type=Path,
        help="Write evaluator-derived per-task tensor memory distribution as CSV to this file. Columns: task_id, name, bytes. Each task starts with a name='_total' row, then per-tensor rows sorted by descending bytes.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Number of parallel task workers. Use 1 for sequential evaluation. Defaults to {DEFAULT_WORKERS}.",
    )
    parser.add_argument(
        "--new-samples",
        type=int,
        default=0,
        help="Generate and evaluate this many fresh ARC-GEN examples per task. Defaults to 0.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Optional seed for reproducible --new-samples generation.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the stderr progress bar.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    submission = args.submission.resolve()
    if not submission.is_file():
        print(f"Error: input file does not exist: {submission}", file=sys.stderr)
        return 2

    try:
        input_kind = classify_submission_input(submission)
        tasks = resolve_tasks_for_input(args.tasks, args.task_file, submission, input_kind)
        entries = zip_entries_by_basename(submission) if input_kind == "zip" else {}
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    if args.workers < 1:
        print("Error: --workers must be at least 1", file=sys.stderr)
        return 2
    if args.new_samples < 0:
        print("Error: --new-samples must be at least 0", file=sys.stderr)
        return 2

    try:
        sample_progress = ProgressBar(
            len(tasks) * args.new_samples,
            enabled=not args.no_progress and args.new_samples > 0,
            label="Generating",
        )
        fresh_examples_by_task = generate_new_examples_for_tasks(
            tasks,
            args.new_samples,
            sample_progress,
            seed=args.seed,
        )
    except (OSError, ImportError, KeyError, RuntimeError, ValueError) as exc:
        print(f"Error: unable to generate fresh ARC-GEN examples: {exc}", file=sys.stderr)
        return 2

    workers = min(args.workers, len(tasks)) if tasks else 1
    progress = ProgressBar(len(tasks), enabled=not args.no_progress, label="Evaluating")
    try:
        if input_kind != "zip":
            result = evaluate_single_input(
                submission,
                input_kind,
                tasks[0],
                fresh_examples_by_task.get(tasks[0], []),
                args.memory_breakdown is not None,
            )
            results = [result]
            progress.update(result)
        elif workers == 1:
            results = evaluate_tasks_sequential(
                tasks,
                submission,
                entries,
                progress,
                fresh_examples_by_task,
                args.memory_breakdown is not None,
            )
        else:
            results = evaluate_tasks_parallel(
                tasks,
                submission,
                workers,
                progress,
                fresh_examples_by_task,
                args.memory_breakdown is not None,
            )
    finally:
        progress.finish()

    if args.verbose:
        print_verbose(results)
    if args.csv:
        try:
            save_results_csv(results, args.csv)
        except OSError as exc:
            print(f"Error: unable to write CSV {args.csv}: {exc}", file=sys.stderr)
            return 2
        print(f"Wrote per-task metrics CSV: {args.csv}")
    if args.memory_breakdown:
        try:
            save_memory_breakdown(results, args.memory_breakdown)
        except OSError as exc:
            print(f"Error: unable to write memory breakdown {args.memory_breakdown}: {exc}", file=sys.stderr)
            return 2
        print(f"Wrote memory breakdown CSV: {args.memory_breakdown}")
    print_summary(results, submission)
    return 1 if any(result.status in {"missing", "error", "metric_error"} for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
