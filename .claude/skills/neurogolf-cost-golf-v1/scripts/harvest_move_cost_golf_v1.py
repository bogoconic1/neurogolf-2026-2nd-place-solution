#!/usr/bin/env python3
"""Move downloaded ONNX into claude-for-chrome-cost-golf-v1/{label}/ and log.

Use snapshot-diff so shared ~/Downloads files are not touched.

Usage:
  python harvest_move_cost_golf_v1.py snap
  python harvest_move_cost_golf_v1.py move <task_num> <session_id> <label1> [label2 ...]

Labels: safe | unsafe | skip
"""
import glob
import os
import shutil
import sys

SNAP = "/tmp/harvest_cost_golf_v1_snap.txt"
DL = os.path.expanduser("~/Downloads")
# repo root = four levels above this script (.../<root>/.claude/skills/<skill>/scripts/)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
BASE = f"{ROOT}/claude-for-chrome-cost-golf-v1"
MAN = f"{ROOT}/zzz_sessions/onnx-harvest-manifest-cost-golf-v1.csv"


def current_onnx():
    return set(glob.glob(DL + "/*.onnx"))


def cmd_snap():
    files = current_onnx()
    with open(SNAP, "w") as f:
        f.write("\n".join(sorted(files)))
    print(f"snapshot: {len(files)} onnx already in Downloads (recorded)")


def cmd_move(task, sid, labels):
    # accept either "214" or "task214"
    if task.lower().startswith("task"):
        task = task[4:]
    task = task.zfill(3)
    before = set()
    if os.path.exists(SNAP):
        before = set(l for l in open(SNAP).read().splitlines() if l.strip())

    new = [f for f in current_onnx() if f not in before]
    tok = f"task{task}"
    matching = [f for f in new if tok in os.path.basename(f)]
    new = matching if matching else new
    new = sorted(new, key=os.path.getmtime)

    if len(new) != len(labels):
        print(
            f"!! MISMATCH task{task}: {len(new)} newly-appeared onnx "
            f"but {len(labels)} labels given"
        )
        print("   new files:", [os.path.basename(f) for f in new])
        sys.exit(1)

    rows = []
    for f, lbl in zip(new, labels):
        if lbl == "skip":
            continue
        if lbl not in {"safe", "unsafe"}:
            print(f"invalid label {lbl!r}; expected safe, unsafe, or skip")
            sys.exit(2)
        bn = os.path.basename(f)
        nn = bn if bn.startswith(f"task{task}") else f"task{task}_{bn}"
        dest = f"{BASE}/{lbl}/{nn}"
        if os.path.exists(dest):
            stem, ext = os.path.splitext(nn)
            i = 1
            while os.path.exists(f"{BASE}/{lbl}/{stem}_{i}{ext}"):
                i += 1
            nn = f"{stem}_{i}{ext}"
            dest = f"{BASE}/{lbl}/{nn}"
        shutil.move(f, dest)
        rows.append(f"{task},{sid},{bn},{lbl}/{nn},{lbl},,,")
        print(f"  {lbl}: {bn} -> {lbl}/{nn}")

    if rows:
        os.makedirs(os.path.dirname(MAN), exist_ok=True)
        with open(MAN, "a") as m:
            m.write("\n".join(rows) + "\n")

    print(
        f"task{task}: moved {len(rows)} file(s). "
        f"[safe={len(glob.glob(BASE + '/safe/*.onnx'))} "
        f"unsafe={len(glob.glob(BASE + '/unsafe/*.onnx'))} "
        f"invalid={len(glob.glob(BASE + '/invalid/*.onnx'))} "
        f"timeout={len(glob.glob(BASE + '/timeout/*.onnx'))}]"
    )


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "snap":
        cmd_snap()
    elif len(sys.argv) >= 5 and sys.argv[1] == "move":
        cmd_move(sys.argv[2], sys.argv[3], sys.argv[4:])
    else:
        print(__doc__)
        sys.exit(2)
