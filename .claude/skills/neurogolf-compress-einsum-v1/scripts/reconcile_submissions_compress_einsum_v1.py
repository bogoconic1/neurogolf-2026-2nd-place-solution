#!/usr/bin/env python3
"""Reconcile NeuroGolf compress-einsum-v1 ONNX submissions by HF Space verdict.

Usage:
  python reconcile_submissions_compress_einsum_v1.py --dry-run
  python reconcile_submissions_compress_einsum_v1.py --apply
"""
import csv
import glob
import hashlib
import json
import os
import re
import shutil
import sys
from collections import Counter

# repo root = four levels above this script (.../<root>/.claude/skills/<skill>/scripts/)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
BASE = f"{ROOT}/claude-for-chrome-compress-einsum-v1"
STATE = f"{ROOT}/auto_submit_onnx_state_claude_for_chrome_compress_einsum_v1.json"
MAN = f"{ROOT}/zzz_sessions/onnx-harvest-manifest-compress-einsum-v1.csv"
LOGS = [
    f"{ROOT}/zzz_sessions/compress-einsum-v1-submit-{k}.log" for k in ("safe", "unsafe")
]

ROUTE = {
    "ok": None,
    "unsafe": "unsafe",
    "wrong": "unsafe",
    "error": "invalid",
    "eval_error": "invalid",
    "timeout": "timeout",
}
API_STATUS = {"error": "eval_error"}


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def parse_points():
    pts = {}
    for log in LOGS:
        if not os.path.exists(log):
            continue
        cur = None
        for ln in open(log):
            if ln.startswith("[file]"):
                cur = ln.split()[1]
            s = ln.strip()
            if s.startswith("{") and cur:
                m = re.search(r'"points":\s*([0-9.]+)', s)
                if m:
                    pts[cur] = m.group(1)
                cur = None
    return pts


def main():
    apply = "--apply" in sys.argv
    if not apply and "--dry-run" not in sys.argv:
        print(__doc__)
        sys.exit(2)

    if not os.path.exists(STATE):
        print(f"(no state file at {STATE} yet — nothing submitted. Clean no-op.)")
        return

    state = json.load(open(STATE))
    pts = parse_points()
    disk = {sha(f): f for f in glob.glob(f"{BASE}/*/*.onnx")}
    print(
        "state entries:",
        len(state),
        " status tally:",
        dict(Counter(e.get("status") for e in state.values())),
    )

    moves = []
    flags = []
    fn_status = {}
    for h, e in state.items():
        st = e.get("status")
        path = disk.get(h)
        if not path:
            continue
        fn = os.path.basename(path)
        fn_status[fn] = st
        cur_folder = os.path.basename(os.path.dirname(path))
        dest = ROUTE.get(st)
        if dest and dest != cur_folder:
            moves.append((path, dest, st))
            if cur_folder == "safe" and dest == "unsafe":
                flags.append(
                    (
                        fn,
                        "submitted under SAFE author but verdict=%s; "
                        "dedup may skip re-post under unsafe author" % st,
                    )
                )

    print("\n=== planned moves ===")
    for path, dest, st in moves:
        print(
            f"  {os.path.basename(path)}: {os.path.basename(os.path.dirname(path))}/ "
            f"-> {dest}/  (status={st})"
        )
    if not moves:
        print("  (none — every file already in the folder its verdict implies)")
    if flags:
        print("\n=== FLAGS ===")
        for fn, why in flags:
            print(f"  {fn}: {why}")

    if not apply:
        print("\n(dry-run — nothing moved, manifest untouched. Re-run with --apply.)")
        return

    moved = {}
    for path, dest, st in moves:
        fn = os.path.basename(path)
        dst = f"{BASE}/{dest}/{fn}"
        shutil.move(path, dst)
        moved[fn] = dest
        print(f"moved {fn} -> {dest}/")

    if os.path.exists(MAN):
        rows = list(csv.reader(open(MAN)))
        out = [rows[0]] if rows else []
        for r in rows[1:]:
            if not r or not r[0].strip():
                out.append(r)
                continue
            while len(r) < 8:
                r.append("")
            fn = r[2]
            st = fn_status.get(fn)
            if st is None:
                out.append(r)
                continue
            if fn in moved:
                r[3] = f"{moved[fn]}/{fn}"
                r[4] = "invalid" if moved[fn] == "invalid" else moved[fn]
            r[5] = API_STATUS.get(st, st)
            if fn in pts and r[5] == "ok":
                r[6] = pts[fn]
            out.append(r)
        csv.writer(open(MAN, "w")).writerows(out)
        print("\nmanifest rewritten.")

    for d in ("safe", "unsafe", "invalid", "timeout"):
        print(f"  {d}: {len(glob.glob(f'{BASE}/{d}/*.onnx'))}")


if __name__ == "__main__":
    main()
