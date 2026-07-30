#!/usr/bin/env python
"""Install the NeuroGolf ONNX stack from the bundled wheels — NO NETWORK REQUIRED.

The sandbox cannot reach PyPI, so `pip install onnx onnxruntime` fails. Every wheel needed
(including transitive deps) is shipped next to this script; this installs from that folder only.

  python skills-offline/wheels/install_offline.py

Run it only when the stack is MISSING — it self-checks first and exits immediately (no pip call) if
onnx/onnxruntime/numpy already import, so it is safe to invoke defensively. Pass --force to install
regardless. Wheels are cp313 / manylinux_2_28 / x86_64 (Python 3.13, Linux 64-bit); the pinned set
(onnx 1.21.0, onnxruntime 1.24.4, numpy 2.4.4, protobuf 6.31.1, onnx-tool 1.0.1) is the combination
the scorer runs, so do not "upgrade" past it.
"""
import subprocess
import sys
from pathlib import Path

WHEELS = Path(__file__).resolve().parent

# Pinned to the scorer's combination. numpy is a floor, not a pin, so an already-present
# NumPy 2.x is kept rather than churned; a 1.x sandbox gets upgraded from the bundled wheel.
REQUIREMENTS = [
    "onnx==1.21.0",
    "onnxruntime==1.24.4",
    "onnx-tool==1.0.1",
    "protobuf==6.31.1",
    "numpy>=2.0",
]

CHECKS = [("onnx", "onnx"), ("onnxruntime", "onnxruntime"), ("numpy", "numpy"),
          ("google.protobuf", "protobuf"), ("onnx_tool", "onnx-tool")]

VERIFY = ("import onnx, onnxruntime, numpy, google.protobuf, onnx_tool; "
          "print(f'OK — onnx {onnx.__version__}, onnxruntime {onnxruntime.__version__}, "
          "numpy {numpy.__version__}')")


def missing() -> list[str]:
    out = []
    for module, dist in CHECKS:
        try:
            __import__(module)
        except Exception:  # noqa: BLE001 - any import failure means "not usable"
            out.append(dist)
    return out


def main() -> int:
    force = "--force" in sys.argv
    absent = missing()
    if not absent and not force:
        import numpy, onnx, onnxruntime  # noqa: E401 - version banner only
        print(f"already installed — onnx {onnx.__version__}, onnxruntime {onnxruntime.__version__}, "
              f"numpy {numpy.__version__}; nothing to do")
        return 0
    if absent:
        print(f"missing: {', '.join(absent)} — installing from the bundled wheels", flush=True)

    if not WHEELS.is_dir() or not list(WHEELS.glob("*.whl")):
        print(f"FATAL: no wheels found in {WHEELS} — re-extract skills-offline.zip", file=sys.stderr)
        return 2

    cmd = [sys.executable, "-m", "pip", "install", "--no-index",
           "--find-links", str(WHEELS), *REQUIREMENTS]
    print("$ " + " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stdout.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        print("\nFATAL: offline install failed. Do NOT fall back to a networked `pip install` — "
              "it cannot reach PyPI. Report the error above instead of working around it.", file=sys.stderr)
        return proc.returncode
    print(proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else "pip: ok")

    # Verify in a FRESH interpreter — this process started before the install, so its import
    # machinery may not see the packages pip just wrote.
    verify = subprocess.run([sys.executable, "-c", VERIFY], capture_output=True, text=True)
    if verify.returncode != 0:
        sys.stderr.write(verify.stderr)
        print("\nFATAL: pip succeeded but the stack is not importable (see above).", file=sys.stderr)
        return 1
    print(verify.stdout.strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
