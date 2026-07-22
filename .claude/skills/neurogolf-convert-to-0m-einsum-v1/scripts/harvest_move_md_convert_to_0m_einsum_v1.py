#!/usr/bin/env python3
"""Move downloaded markdown write-ups into claude-for-chrome-convert-to-0m-einsum-v1-markdown/.

Use snapshot-diff so shared ~/Downloads files are not touched.

Usage:
  python harvest_move_md_convert_to_0m_einsum_v1.py snap
  python harvest_move_md_convert_to_0m_einsum_v1.py move <task_num> <session_id>
"""
import glob
import os
import shutil
import sys

SNAP = "/tmp/harvest_md_convert_to_0m_einsum_v1_snap.txt"
DL = os.path.expanduser("~/Downloads")
BASE = os.path.expanduser("~/Documents/NeuroGolf-2026/claude-for-chrome-convert-to-0m-einsum-v1-markdown")
MAN = os.path.expanduser(
    "~/Documents/NeuroGolf-2026/zzz_sessions/markdown-harvest-manifest-convert-to-0m-einsum-v1.csv"
)


def current_md():
    return set(glob.glob(DL + "/*.md")) | set(glob.glob(DL + "/*.markdown"))


def cmd_snap():
    files = current_md()
    with open(SNAP, "w") as f:
        f.write("\n".join(sorted(files)))
    print(f"snapshot: {len(files)} md already in Downloads (recorded)")


def cmd_move(task, sid):
    # accept either "214" or "task214"
    if task.lower().startswith("task"):
        task = task[4:]
    task = task.zfill(3)
    before = set()
    if os.path.exists(SNAP):
        before = set(l for l in open(SNAP).read().splitlines() if l.strip())
    new = [f for f in current_md() if f not in before]
    tok = f"task{task}"
    new = [f for f in new if tok in os.path.basename(f)]
    new = sorted(new, key=os.path.getmtime)

    if not new:
        appeared = len([f for f in current_md() if f not in before])
        print(
            f"task{task}: 0 md matching '{tok}' ({appeared} new md appeared but none "
            "carried the token). Check names / rename manually before moving."
        )
        return

    rows = []
    os.makedirs(BASE, exist_ok=True)
    for f in new:
        bn = os.path.basename(f)
        nn = bn if bn.startswith(f"task{task}") else f"task{task}_{bn}"
        dest = f"{BASE}/{nn}"
        if os.path.exists(dest):
            stem, ext = os.path.splitext(nn)
            i = 1
            while os.path.exists(f"{BASE}/{stem}_{i}{ext}"):
                i += 1
            nn = f"{stem}_{i}{ext}"
            dest = f"{BASE}/{nn}"
        shutil.move(f, dest)
        rows.append(f"{task},{sid},{bn},{nn},")
        print(f"  moved: {bn} -> {nn}")

    os.makedirs(os.path.dirname(MAN), exist_ok=True)
    with open(MAN, "a") as m:
        m.write("\n".join(rows) + "\n")

    total = len(glob.glob(BASE + "/*.md")) + len(glob.glob(BASE + "/*.markdown"))
    print(f"task{task}: moved {len(rows)} md. [folder total={total}]")


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "snap":
        cmd_snap()
    elif len(sys.argv) >= 4 and sys.argv[1] == "move":
        cmd_move(sys.argv[2], sys.argv[3])
    else:
        print(__doc__)
        sys.exit(2)
