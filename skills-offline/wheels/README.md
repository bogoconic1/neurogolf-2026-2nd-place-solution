# Offline wheelhouse — install the ONNX stack without PyPI

The sandbox has **no network access to PyPI**, so `pip install onnx onnxruntime` fails. Every wheel
needed to run `skills-offline/evaluate/scripts/evaluate.py` and the `skills-offline/solutions_py/*.py`
builders — including transitive dependencies — is shipped here.

Install only when the stack is actually missing — check first:

```bash
python -c "import onnx, onnxruntime, numpy" 2>/dev/null || python skills-offline/wheels/install_offline.py
```

`install_offline.py` also self-checks, so calling it directly is safe: if everything already imports it
prints `already installed — …` and exits without touching pip. Pass `--force` to reinstall anyway. On a
real install it prints `OK — onnx 1.21.0, onnxruntime 1.24.4, numpy 2.4.4`.

Equivalent raw command, if you prefer pip directly:

```bash
python -m pip install --no-index --find-links skills-offline/wheels \
  onnx==1.21.0 onnxruntime==1.24.4 onnx-tool==1.0.1 protobuf==6.31.1 "numpy>=2.0"
```

`--no-index` is required: it stops pip from trying (and failing) to reach PyPI. If the install errors,
report the error — a networked `pip install` is not a fallback, it cannot work.

## What is here

Wheels are **cp313 / manylinux_2_28 / x86_64** — Python 3.13 on 64-bit Linux, matching the sandbox.

| Package | Version | Why |
| --- | --- | --- |
| onnx | 1.21.0 | build/inspect graphs |
| onnxruntime | 1.24.4 | run graphs; the scorer's runtime |
| numpy | 2.4.4 | everything |
| protobuf | 6.31.1 | onnx serialization — the version proven against onnx 1.21.0, **not** the latest |
| onnx-tool | 1.0.1 | `neurogolf_utils` memory profiling |
| ml_dtypes, typing_extensions, sympy, mpmath, flatbuffers, packaging, tabulate | — | transitive deps |

This pinned set is the combination the scorer runs. Do not upgrade past it, and do not add wheels for
other Python versions — the sandbox is 3.13.

## Refreshing

From the repo root, on a machine with network:

```bash
python -m pip download --only-binary=:all: \
  --python-version 3.13 --implementation cp --abi cp313 --abi abi3 --abi none \
  --platform manylinux_2_28_x86_64 --platform manylinux2014_x86_64 --platform any \
  -d skills-offline/wheels \
  onnx==1.21.0 onnxruntime==1.24.4 numpy==2.4.4 onnx-tool==1.0.1 protobuf==6.31.1 \
  ml_dtypes==0.5.3 typing_extensions==4.15.0 sympy==1.14.0 flatbuffers==25.9.23 packaging==25.0
```

Verify the wheelhouse is self-contained before shipping it — this must resolve with no network:

```bash
python -m pip download --no-index --find-links skills-offline/wheels --only-binary=:all: \
  --python-version 3.13 --implementation cp --abi cp313 --abi abi3 --abi none \
  --platform manylinux_2_28_x86_64 --platform manylinux2014_x86_64 --platform any \
  -d /tmp/closure-check onnx onnxruntime onnx-tool
```

The wheels ride inside `skills-offline.zip` automatically (the daily bundle archives the whole
`skills-offline/` tree), which adds ~61 MB to the bundle.
