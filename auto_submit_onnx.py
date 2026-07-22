#!/usr/bin/env python3
"""Submit harvested NeuroGolf ONNX to the LIVE HF Space, classified by folder.

Two modes:

  * FOLDER-CLASSIFY (default): walk ROOT/{safe,unsafe}, submit each
    onnx under a per-folder author, and act on the Space's verdict:
      - status == "ok"         -> passes all; leave in safe/.
      - status == "unsafe"     -> passed static, failed fresh ARC-GEN; leave in
                                  unsafe/. If the file was in safe/, MOVE it to
                                  unsafe/ and log the label/verdict mismatch.
      - status == "eval_error" -> model failed to evaluate (load / shape-infer /
                                  checker error / Space exception);
                                  MOVE the file to invalid/.
      - submit TIMEOUT (default >300s, configurable) -> status "timeout";
                                  MOVE the file to timeout/ (kept distinct from
                                  invalid/ — a hang is not an evaluator rejection).
    A populated `error` field alone does NOT mean invalid — fresh-fails set it too;
    we key off `status`.

  * WATCH (--watch-dir DIR): continuously poll a directory and submit new *.onnx,
    up to --concurrency in flight (the v3–v7 sweeps run in this mode).

Filenames are renamed taskNNN_… during harvest, so task inference is a regex;
the Claude LLM call is only a fallback (and is lazy — no ANTHROPIC_API_KEY needed
unless a filename has no taskNNN prefix).

Usage:
  export HF_TOKEN=...
  python auto_submit_onnx.py                 # classify safe/ + unsafe/, submit, --once implied
  python auto_submit_onnx.py --dry-run       # show what would be submitted, no API calls
  python auto_submit_onnx.py --root claude-for-chrome-runtime-v1 --submit-timeout 600 \
    --safe-submitted-by claude-for-chrome-runtime-v1 \
    --unsafe-submitted-by claude-for-chrome-runtime-v1-fresh-unsafe
  python auto_submit_onnx.py --watch-dir ~/Downloads   # continuous watch-dir poller
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from hashlib import sha256
from pathlib import Path

from dotenv import load_dotenv
from gradio_client import Client, handle_file

load_dotenv()

DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_SPACE = "golfingteam10000pts/neurogolf-leaderboard-v2"
# Eval runs on Modal (up to 60 min). With concurrent submits a long wait no longer stalls the queue,
# so wait out the full Modal budget rather than false-timing-out a still-running eval into timeout/.
DEFAULT_SUBMIT_TIMEOUT = 3700.0  # seconds around client.predict (just over Modal's 3600s)
DEFAULT_CONCURRENCY = 32  # simultaneous /submit_api calls per poller

# Default artifact root (sibling of this script).
DEFAULT_ARTIFACT_ROOT = Path(__file__).resolve().parent / "playweight-onnx-v3"

# Per-folder submission author (HF username auto-prepended).
DEFAULT_FOLDER_AUTHOR = {
    "safe": "playwright-download-ui-v3",
    "unsafe": "playwright-download-ui-v3-fresh-unsafe",
}

_TASK_RE = re.compile(r"task0*(\d+)", re.IGNORECASE)


def infer_task_regex(filename: str) -> int | None:
    m = _TASK_RE.search(filename)
    return int(m.group(1)) if m else None


def infer_task_llm(model: str, filename: str) -> int | None:
    """Fallback only — lazily imports anthropic so no API key is needed normally."""
    import anthropic
    from pydantic import BaseModel, Field

    class TaskInference(BaseModel):
        is_task_onnx: bool = Field(description="True if filename encodes a task number.")
        task_number: int | None = Field(default=None, description="The integer task number.")

    client = anthropic.Anthropic()
    resp = client.messages.parse(
        model=model,
        max_tokens=256,
        system=(
            "You extract the NeuroGolf task number from an ONNX filename. Task "
            "numbers look like 'task091', 'task 91', 'task091_v6'. Return the integer."
        ),
        messages=[{"role": "user", "content": f"Filename: {filename}"}],
        output_format=TaskInference,
    )
    out = resp.parsed_output
    return out.task_number if out.is_task_onnx else None


def infer_task(model: str, filename: str) -> int | None:
    t = infer_task_regex(filename)
    if t is not None:
        return t
    try:
        return infer_task_llm(model, filename)
    except Exception as exc:
        print(f"  LLM task-inference failed: {type(exc).__name__}: {exc}", flush=True)
        return None


def hf_username(token: str | None) -> str:
    from huggingface_hub import HfApi

    name = HfApi(token=token).whoami().get("name")
    if not name:
        raise RuntimeError("Cannot resolve HF username from token.")
    return str(name)


def submitted_by_name(suffix: str, username: str) -> str:
    import posixpath

    suffix = (suffix or "").strip().strip("/")
    if suffix.split("/", 1)[0] == username:
        raise ValueError("--submitted-by must be a suffix only; do not include the HF username.")
    return posixpath.join(username, suffix)


def _predict(space: Client, task_id: str, submitted_by: str, source: str) -> dict:
    result = space.predict(
        task_id,
        submitted_by,
        None,                    # python_file: unused
        handle_file(source),     # onnx_file: local path
        api_name="/submit_api",
    )
    payload = result[0] if isinstance(result, (list, tuple)) else result
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            payload = {"raw": payload}
    return payload if isinstance(payload, dict) else {"raw": payload}


def submit_with_timeout(
    space: Client,
    task_id: str,
    submitted_by: str,
    source: str,
    submit_timeout: float,
) -> dict:
    """Run /submit_api with a hard timeout. On timeout -> synthesize eval_error.

    NOTE: we do NOT use `with ThreadPoolExecutor(...)` — its __exit__ calls
    shutdown(wait=True), which would re-block on a genuinely hung predict thread
    and defeat the timeout. We create the executor manually and shut it down with
    wait=False so a hang is abandoned (the daemon-ish worker is left to die with
    the process) and the sweep advances to the next file.
    """
    print(f"  -> /submit_api task={task_id} as {submitted_by}", flush=True)
    ex = ThreadPoolExecutor(max_workers=1)
    fut = ex.submit(_predict, space, task_id, submitted_by, source)
    try:
        return fut.result(timeout=submit_timeout)
    except FutureTimeout:
        print(f"  TIMEOUT after {submit_timeout}s -> timeout/", flush=True)
        return {"status": "timeout", "error": f"submit timeout >{submit_timeout}s"}
    finally:
        ex.shutdown(wait=False)


def load_state(path: Path) -> dict:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    # legacy format: {"done": [hash, ...]} -> {hash: {"status": "legacy"}}
    if isinstance(raw, dict) and "done" in raw and isinstance(raw["done"], list):
        return {h: {"status": "legacy"} for h in raw["done"]}
    return raw if isinstance(raw, dict) else {}


def save_state(path: Path, state: dict) -> None:
    path.write_text(json.dumps(state, indent=2, sort_keys=True))


def append_result(results_csv: Path, row: dict) -> None:
    header = "task,folder,filename,status,points,action,note\n"
    if not results_csv.exists():
        results_csv.write_text(header)
    line = ",".join(str(row.get(k, "")).replace(",", ";") for k in
                     ("task", "folder", "filename", "status", "points", "action", "note"))
    with results_csv.open("a") as fh:
        fh.write(line + "\n")


def classify_folders(args, username: str, token: str) -> int:
    root = args.root.expanduser().resolve()
    state_path = args.state_file
    state = load_state(state_path)
    results_csv = Path(__file__).resolve().parent / "zzz_sessions" / "onnx-submit-results.csv"
    invalid_dir = root / "invalid"
    unsafe_dir = root / "unsafe"
    timeout_dir = root / "timeout"
    for folder in ("safe", "unsafe", "invalid", "timeout"):
        (root / folder).mkdir(parents=True, exist_ok=True)

    counts = {"ok": 0, "unsafe": 0, "eval_error": 0, "timeout": 0, "moved_to_invalid": 0,
              "moved_to_unsafe": 0, "moved_to_timeout": 0, "skipped": 0, "submitted": 0}
    lock = threading.Lock()

    # Build the work-list single-threaded (skip already-done + un-inferrable files).
    worklist: list[tuple[str, Path, str, str, str]] = []  # (folder, onnx, digest, task_id, submitted_by)
    for folder in ("safe", "unsafe"):
        fdir = root / folder
        author_suffix = args.safe_submitted_by if folder == "safe" else args.unsafe_submitted_by
        submitted_by = submitted_by_name(author_suffix, username)
        files = sorted(fdir.glob("*.onnx"))
        print(f"\n=== folder {folder}/  ({len(files)} onnx)  author={submitted_by} ===", flush=True)
        for onnx in files:
            if not onnx.exists():
                continue
            digest = sha256(onnx.read_bytes()).hexdigest()
            prev = state.get(digest)
            if prev and prev.get("status") not in (None, "legacy"):
                counts["skipped"] += 1
                continue
            task_num = infer_task(args.model, onnx.name)
            if task_num is None:
                print(f"  skip {onnx.name} — no task number", flush=True)
                counts["skipped"] += 1
                continue
            task_id = f"task{task_num:03d}"
            if args.dry_run:
                print(f"  DRY-RUN would submit {folder}/{onnx.name} ({task_id}) as {submitted_by}", flush=True)
                continue
            worklist.append((folder, onnx, digest, task_id, submitted_by))

    if not args.dry_run and worklist:
        space = Client(args.space, token=token)

        def _process(item: tuple[str, Path, str, str, str]) -> None:
            folder, onnx, digest, task_id, submitted_by = item
            print(f"\n[file] {folder}/{onnx.name}  ({task_id})", flush=True)
            try:
                result = submit_with_timeout(space, task_id, submitted_by, str(onnx), args.submit_timeout)
            except Exception as exc:
                print(f"  submit EXCEPTION -> eval_error: {type(exc).__name__}: {exc}", flush=True)
                result = {"status": "eval_error", "error": f"{type(exc).__name__}: {exc}"}
            status = result.get("status")
            points = result.get("points") or result.get("score") or ""
            action = "kept"
            note = (result.get("error") or "")[:160]
            print(f"  status={status} points={points} {json.dumps(result)[:300]}", flush=True)
            with lock:
                counts["submitted"] += 1
                if status == "timeout":
                    shutil.move(str(onnx), str(timeout_dir / onnx.name))
                    action = "moved_to_timeout"
                    counts["timeout"] += 1
                    counts["moved_to_timeout"] += 1
                elif status in ("eval_error", "error"):
                    # Space could not evaluate the model (load / shape-infer / checker /
                    # QLinearConv mismatch / generic exception) -> not submit-valid.
                    shutil.move(str(onnx), str(invalid_dir / onnx.name))
                    action = "moved_to_invalid"
                    counts["eval_error"] += 1
                    counts["moved_to_invalid"] += 1
                elif status in ("unsafe", "wrong"):
                    # "unsafe" = passed static, failed fresh ARC-GEN.
                    # "wrong"  = wrong results (fails even static ARC-AGI/GEN) -> a
                    #            non-solver; not safe. Both belong in unsafe/.
                    counts["unsafe"] += 1
                    if folder == "safe":
                        shutil.move(str(onnx), str(unsafe_dir / onnx.name))
                        action = f"moved_to_unsafe (label mismatch: safe->{status})"
                        counts["moved_to_unsafe"] += 1
                        print(f"  !! MISMATCH: safe-labeled but Space says {status}; moved to unsafe/", flush=True)
                elif status == "ok":
                    counts["ok"] += 1
                else:
                    note = f"unexpected status={status}; " + note
                state[digest] = {"task": task_id, "folder": folder, "status": status,
                                 "points": points, "action": action}
                save_state(state_path, state)
                append_result(results_csv, {"task": task_id, "folder": folder,
                                            "filename": onnx.name, "status": status,
                                            "points": points, "action": action, "note": note})

        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            list(pool.map(_process, worklist))

    print(f"\n=== DONE === {json.dumps(counts)}", flush=True)
    print(f"results: {results_csv}", flush=True)
    return 0


# ---------------------------------------------------------------------------
# watch-dir poller: continuously submit new *.onnx, up to --concurrency in flight
# ---------------------------------------------------------------------------
PARTIAL_SUFFIXES = (".crdownload", ".part", ".download", ".tmp")


def watch_loop(args, username: str, token: str) -> int:
    state = load_state(args.state_file)
    submitted_by = submitted_by_name(args.submitted_by, username)
    space = Client(args.space, token=token)
    print(f"Watching {args.watch_dir} as {submitted_by} (concurrency={args.concurrency})", flush=True)

    lock = threading.Lock()
    inflight: set[str] = set()
    pool = ThreadPoolExecutor(max_workers=args.concurrency)

    def _process(digest: str, onnx: Path, task_id: str) -> None:
        try:
            result = submit_with_timeout(space, task_id, submitted_by, str(onnx), args.submit_timeout)
        except Exception as exc:
            # transport-level failure -> don't record, retry on a later pass
            print(f"  submit ERROR (retry): {type(exc).__name__}: {exc}", flush=True)
            with lock:
                inflight.discard(digest)
            return
        print(f"  {json.dumps(result)[:400]}", flush=True)
        with lock:
            state[digest] = {"task": task_id, "status": result.get("status")}
            save_state(args.state_file, state)
            inflight.discard(digest)

    def once() -> None:
        now = time.time()
        for onnx in sorted(Path(args.watch_dir).glob("*.onnx")):
            if onnx.name.endswith(PARTIAL_SUFFIXES):
                continue
            try:
                mtime = onnx.stat().st_mtime
            except FileNotFoundError:
                continue
            if args.max_age_seconds and (now - mtime) > args.max_age_seconds:
                continue
            if (now - mtime) < args.settle_seconds:
                continue
            digest = sha256(onnx.read_bytes()).hexdigest()
            with lock:
                if digest in state or digest in inflight:
                    continue
                inflight.add(digest)
            task_num = infer_task(args.model, onnx.name)
            if task_num is None:
                with lock:
                    state[digest] = {"status": "no-task"}
                    save_state(args.state_file, state)
                    inflight.discard(digest)
                continue
            task_id = f"task{task_num:03d}"
            print(f"\n[file] {onnx.name} -> {task_id}", flush=True)
            pool.submit(_process, digest, onnx, task_id)

    try:
        once()
        if args.once:
            return 0
        while True:
            time.sleep(args.interval)
            once()
    finally:
        # drains in-flight submits (notably on --once before returning)
        pool.shutdown(wait=True)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--watch-dir", type=Path, default=None,
                   help="Legacy poller over a directory instead of folder-classify mode.")
    p.add_argument("--submitted-by", default="playwright-download-ui-v3")
    p.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ARTIFACT_ROOT,
        help="Artifact root containing safe/, unsafe/, invalid/, and timeout/.",
    )
    p.add_argument("--safe-submitted-by", default=DEFAULT_FOLDER_AUTHOR["safe"])
    p.add_argument("--unsafe-submitted-by", default=DEFAULT_FOLDER_AUTHOR["unsafe"])
    p.add_argument(
        "--submit-timeout",
        type=float,
        default=DEFAULT_SUBMIT_TIMEOUT,
        help=f"Seconds to wait for one /submit_api call before moving to timeout/ (default {DEFAULT_SUBMIT_TIMEOUT}).",
    )
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--space", default=DEFAULT_SPACE)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--state-file", type=Path,
                   default=Path(__file__).resolve().parent / "auto_submit_onnx_state.json")
    p.add_argument("--once", action="store_true")
    p.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help=f"Simultaneous /submit_api calls (default {DEFAULT_CONCURRENCY}); eval runs on Modal so submits no longer serialize.",
    )
    p.add_argument("--interval", type=float, default=10.0)
    p.add_argument("--max-age-seconds", type=float, default=3600.0)
    p.add_argument("--settle-seconds", type=float, default=5.0)
    args = p.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print("HF_TOKEN not set in environment.", file=sys.stderr)
        return 1
    username = hf_username(token) if not args.dry_run else "DRYRUN"

    if args.watch_dir is not None:
        return watch_loop(args, username, token)
    return classify_folders(args, username, token)


if __name__ == "__main__":
    raise SystemExit(main())
