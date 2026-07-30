#!/usr/bin/env python3
"""Auto-post downloaded NeuroGolf markdown write-ups to the HF discussions, by task.

Companion to auto_submit_onnx.py. For every *.md / *.markdown in a watch directory,
ask Claude (Sonnet 4.6, structured output) to decide — given the file and the current
open discussion threads — whether to COMMENT on an existing task thread (and which one)
or PUBLISH a new discussion, plus a title + excerpt for the new-thread case. Then post
directly via huggingface_hub.HfApi (get_repo_discussions / comment_discussion /
create_discussion) — no skill files needed, works even when the Space is down.

Usage:
  export ANTHROPIC_API_KEY=...   # the Sonnet decision call
  export HF_TOKEN=...            # HfApi auth
  python auto_post_md.py --once
  python auto_post_md.py         # poll forever
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from hashlib import sha256
from pathlib import Path
from typing import Literal

import anthropic
from dotenv import load_dotenv
from huggingface_hub import HfApi
from pydantic import BaseModel, Field

# Load HF_TOKEN / ANTHROPIC_API_KEY from a .env in the current directory.
load_dotenv()

DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_REPO = "golfingteam10000pts/neurogolf-leaderboard-v2"
DEFAULT_REPO_TYPE = "space"

# Discussion field limits (same as the neurogolf-discussions-* skill).
MIN_TITLE, MAX_TITLE = 12, 160
MIN_EXCERPT, MAX_EXCERPT = 40, 500
# Cap markdown bytes sent to the model (the body is posted in full regardless).
MAX_MD_CHARS = 24000

PARTIAL_SUFFIXES = (".crdownload", ".part", ".download", ".tmp")
MD_GLOBS = ("*.md", "*.markdown")


class DiscussionDecision(BaseModel):
    """Sonnet's decision for one markdown file."""

    task_number: int | None = Field(
        default=None,
        description="Inferred NeuroGolf task number (e.g. 91). Null if the file is not "
        "a task write-up.",
    )
    action: Literal["comment", "publish"] = Field(
        description="'comment' to add to an existing thread, 'publish' to open a new one."
    )
    discussion_id: int | None = Field(
        default=None,
        description="Required when action=='comment': the id of an existing thread from "
        "the provided list. Must be one of those ids.",
    )
    title: str | None = Field(
        default=None,
        description=f"Required when action=='publish': {MIN_TITLE}-{MAX_TITLE} chars, "
        "should start with 'taskNNN'.",
    )
    excerpt: str | None = Field(
        default=None,
        description=f"Required when action=='publish': a {MIN_EXCERPT}-{MAX_EXCERPT} char "
        "one-sentence summary of the write-up.",
    )
    reason: str = Field(description="One short sentence explaining the decision.")


def decide(client: anthropic.Anthropic, model: str, filename: str, body: str,
           threads: list[dict]) -> DiscussionDecision:
    """One structured Sonnet call: comment-vs-publish + thread id + title/excerpt."""
    thread_lines = "\n".join(f"  id={t['id']}  title={t['title']!r}" for t in threads) or "  (none)"
    system = (
        "You route a NeuroGolf task write-up to the project's Hugging Face discussions. "
        "Task numbers look like task091 / task 91 / task091_v6. Discussion threads are "
        "named per task, e.g. 'task091 ...'. Decide:\n"
        "- If an existing thread clearly belongs to this file's task, action='comment' "
        "and set discussion_id to that thread's id (it MUST be one of the listed ids).\n"
        "- Otherwise action='publish' and write a title "
        f"({MIN_TITLE}-{MAX_TITLE} chars, starting with the taskNNN id) and an excerpt "
        f"({MIN_EXCERPT}-{MAX_EXCERPT} chars, one sentence).\n"
        "- If the file is not a task write-up, set task_number=null (still pick an action "
        "but it will be skipped).\n"
        "Prefer commenting on an existing task thread over opening duplicates."
    )
    user = (
        f"Filename: {filename}\n\n"
        f"Open discussion threads:\n{thread_lines}\n\n"
        f"--- markdown content (may be truncated) ---\n{body[:MAX_MD_CHARS]}"
    )
    resp = client.messages.parse(
        model=model,
        max_tokens=1024,
        system=system,
        messages=[{"role": "user", "content": user}],
        output_format=DiscussionDecision,
    )
    return resp.parsed_output


def fetch_threads(api: HfApi, repo: str, repo_type: str) -> list[dict]:
    """List open discussions as [{id, title}] — the exact HfApi call the skill uses."""
    return [
        {"id": d.num, "title": d.title}
        for d in api.get_repo_discussions(
            repo_id=repo,
            repo_type=repo_type,
            discussion_type="discussion",
            discussion_status="open",
        )
    ]


def comment_thread(api: HfApi, repo: str, repo_type: str, discussion_id: int, body: str) -> dict:
    """Comment on an existing thread with raw markdown (skill's comment_discussion)."""
    api.comment_discussion(
        repo_id=repo, repo_type=repo_type, discussion_num=discussion_id, comment=body
    )
    return {"id": discussion_id, "commented": True}


def publish_thread(api: HfApi, repo: str, repo_type: str, title: str, excerpt: str, body: str) -> dict:
    """Open a new discussion (skill's create_discussion). Excerpt-prefixed body."""
    description = f"{excerpt}\n\n{body.strip()}"
    d = api.create_discussion(
        repo_id=repo, repo_type=repo_type, title=title, description=description
    )
    return {"id": d.num, "url": d.url, "title": title}


def clamp(text: str, lo: int, hi: int) -> str | None:
    text = " ".join((text or "").split())
    if len(text) > hi:
        text = text[:hi].rstrip()
    return text if len(text) >= lo else None


def load_state(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return set(json.loads(path.read_text()).get("done", []))


def save_state(path: Path, done: set[str]) -> None:
    path.write_text(json.dumps({"done": sorted(done)}, indent=2))


class ThreadCache:
    """Caches the discussions list for --list-ttl seconds across a burst of files."""

    def __init__(self, api: HfApi, repo: str, repo_type: str, ttl: float):
        self.api, self.repo, self.repo_type, self.ttl = api, repo, repo_type, ttl
        self._threads: list[dict] = []
        self._fetched_at = 0.0

    def get(self, now: float) -> list[dict]:
        if not self._threads or (now - self._fetched_at) > self.ttl:
            self._threads = fetch_threads(self.api, self.repo, self.repo_type)
            self._fetched_at = now
        return self._threads


def handle_file(md: Path, args, llm: anthropic.Anthropic, api: HfApi, cache: ThreadCache, now: float) -> None:
    """Decide + post one markdown. Raises on transient failure (caller retries)."""
    threads = cache.get(now)
    body = md.read_text(encoding="utf-8")
    dec = decide(llm, args.model, md.name, body, threads)
    print(f"  decision: {dec.action} task={dec.task_number} id={dec.discussion_id} — {dec.reason}", flush=True)

    if dec.task_number is None:
        print("  skip — not a task write-up", flush=True)
        return

    valid_ids = {t["id"] for t in threads}
    if dec.action == "comment" and dec.discussion_id in valid_ids:
        payload = comment_thread(api, args.repo, args.repo_type, dec.discussion_id, body)
    else:
        if dec.action == "comment":
            print(f"  [warn] discussion_id {dec.discussion_id} not in thread list — publishing instead", flush=True)
        title = clamp(dec.title or f"task{dec.task_number:03d} write-up", MIN_TITLE, MAX_TITLE)
        excerpt = clamp(dec.excerpt or "", MIN_EXCERPT, MAX_EXCERPT)
        if not title or not excerpt:
            print(f"  skip — title/excerpt out of bounds (title={dec.title!r} excerpt={dec.excerpt!r})", flush=True)
            return
        payload = publish_thread(api, args.repo, args.repo_type, title, excerpt, body)

    print(f"  post OK: {json.dumps(payload)[:500]}", flush=True)


def process_once(args, llm: anthropic.Anthropic, api: HfApi, cache: ThreadCache, done: set[str]) -> None:
    now = time.time()
    # Sort by mtime (oldest first), NOT filename: files are downloaded top-to-bottom of the
    # session and the mover preserves each download's mtime, so mtime-ascending == conversation
    # turn order. This makes the HF comments land in turn order (earlier turns posted first),
    # and across sessions later sessions have later mtimes so they still post after earlier ones.
    def _mtime(p: Path) -> float:
        try:
            return p.stat().st_mtime
        except FileNotFoundError:
            return 0.0
    files = sorted({p for g in MD_GLOBS for p in Path(args.watch_dir).glob(g)}, key=_mtime)
    for md in files:
        if md.name.endswith(PARTIAL_SUFFIXES):
            continue
        try:
            mtime = md.stat().st_mtime
        except FileNotFoundError:
            continue
        if args.max_age_seconds and (now - mtime) > args.max_age_seconds:
            continue
        if (now - mtime) < args.settle_seconds:
            print(f"[wait] {md.name} still settling", flush=True)
            continue
        digest = sha256(md.read_bytes()).hexdigest()
        if digest in done:
            continue
        print(f"\n[file] {md.name}", flush=True)
        try:
            handle_file(md, args, llm, api, cache, now)
        except Exception as exc:  # could not reach the service — leave un-recorded, retry next poll
            is_429 = "429" in str(exc) or "Too Many Requests" in str(exc)
            # HF caps discussion comments at 60/hour; on 429 wait out the window before retrying.
            backoff = args.rate_limit_backoff if is_429 else args.error_backoff
            print(f"  post ERROR (will retry after {backoff:.0f}s): {type(exc).__name__}: {exc}", flush=True)
            time.sleep(backoff)
            break  # stop this poll cycle so we don't hammer the API; resume next poll from here
        # Record after any completed attempt (posted, rejected, or skipped) — each file once.
        done.add(digest)
        save_state(args.state_file, done)
        time.sleep(args.post_delay)  # throttle: stay under the HF discussion-comment rate limit


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--watch-dir", type=Path, required=True,
                   help="markdown folder to drain (e.g. claude-for-chrome-compress-einsum-v1-markdown)")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--repo", default=DEFAULT_REPO)
    p.add_argument("--repo-type", default=DEFAULT_REPO_TYPE)
    p.add_argument(
        "--state-file",
        type=Path,
        default=Path(__file__).resolve().parent / "auto_post_md_state.json",
    )
    p.add_argument("--once", action="store_true", help="One pass then exit (default: poll)")
    p.add_argument("--interval", type=float, default=10.0, help="Poll seconds when not --once")
    p.add_argument("--max-age-seconds", type=float, default=0.0,
                   help="Only consider files downloaded within this window (0 = no limit)")
    p.add_argument("--settle-seconds", type=float, default=5.0,
                   help="Skip files modified within this many seconds (still downloading)")
    p.add_argument("--list-ttl", type=float, default=120.0,
                   help="Seconds to cache the discussions list across files")
    p.add_argument("--post-delay", type=float, default=62.0,
                   help="Seconds after each successful post. HF caps comments at 60/hour, so >=61s.")
    p.add_argument("--error-backoff", type=float, default=20.0,
                   help="Seconds to back off on a non-rate-limit post error")
    p.add_argument("--rate-limit-backoff", type=float, default=3660.0,
                   help="Seconds to wait after an HTTP 429 (HF comment cap is 60/hour)")
    args = p.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print("HF_TOKEN not set in environment.", file=sys.stderr)
        return 1
    llm = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
    api = HfApi(token=token)
    cache = ThreadCache(api, args.repo, args.repo_type, args.list_ttl)
    done = load_state(args.state_file)
    print(f"Watching {args.watch_dir} for markdown ({len(done)} already posted)", flush=True)

    if args.once:
        process_once(args, llm, api, cache, done)
        return 0
    while True:
        process_once(args, llm, api, cache, done)
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
