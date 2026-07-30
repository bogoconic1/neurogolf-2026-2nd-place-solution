from __future__ import annotations

import argparse
import json
import keyword
import math
import pprint
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, numpy_helper


DTYPE_BY_TENSOR_TYPE = {
    TensorProto.FLOAT: "np.float32",
    TensorProto.UINT8: "np.uint8",
    TensorProto.INT8: "np.int8",
    TensorProto.UINT16: "np.uint16",
    TensorProto.INT16: "np.int16",
    TensorProto.INT32: "np.int32",
    TensorProto.INT64: "np.int64",
    TensorProto.BOOL: "np.bool_",
    TensorProto.FLOAT16: "np.float16",
    TensorProto.DOUBLE: "np.float64",
    TensorProto.UINT32: "np.uint32",
    TensorProto.UINT64: "np.uint64",
    TensorProto.STRING: "np.object_",
}


TEMPLATE = '''from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = {task_id!r}
TASK_NUM = {task_num}
KAGGLE = {kaggle}
MEMORY_BYTES = {memory_bytes}
PARAMS = {params}
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = {ir_version}
PRODUCER_NAME = {producer_name!r}
PRODUCER_VERSION = {producer_version!r}
DOMAIN = {domain!r}
MODEL_VERSION = {model_version}
GRAPH_NAME = {graph_name!r}
OPSETS = {opsets}


{initializer_constants}


NP_DTYPE = {{
    TensorProto.FLOAT: np.float32,
    TensorProto.UINT8: np.uint8,
    TensorProto.INT8: np.int8,
    TensorProto.UINT16: np.uint16,
    TensorProto.INT16: np.int16,
    TensorProto.INT32: np.int32,
    TensorProto.INT64: np.int64,
    TensorProto.BOOL: np.bool_,
    TensorProto.FLOAT16: np.float16,
    TensorProto.DOUBLE: np.float64,
    TensorProto.UINT32: np.uint32,
    TensorProto.UINT64: np.uint64,
}}


def _num(value):
    if value == "inf":
        return float("inf")
    if value == "-inf":
        return float("-inf")
    if value == "nan":
        return float("nan")
    return value


def _tensor(name: str, elem_type: int, shape: tuple[int, ...], values: list) -> onnx.TensorProto:
    if elem_type == TensorProto.STRING:
        vals = [v.encode("utf-8") if isinstance(v, str) else v for v in values]
        return helper.make_tensor(name=name, data_type=TensorProto.STRING, dims=list(shape), vals=vals)
    flat = [_num(value) for value in values]
    array = np.asarray(flat, dtype=NP_DTYPE[elem_type]).reshape(shape)
    return numpy_helper.from_array(array, name=name)


def _vi(name: str, elem_type: int, shape: list[int | str | None]) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, elem_type, shape)


class _EmptyAttr:
    # Sentinel for an EMPTY list-valued attribute (INTS/FLOATS/STRINGS, e.g. a scalar
    # RandomUniform's shape=[]). helper.make_node() cannot infer the attribute type from a
    # bare [], so _node() re-adds these with an explicit attr_type after building the node.
    __slots__ = ("kind",)

    def __init__(self, kind: str) -> None:
        self.kind = kind


def _node(op_type, inputs, outputs, name="", domain="", **attrs):
    empties = [(k, v.kind) for k, v in attrs.items() if isinstance(v, _EmptyAttr)]
    for k, _ in empties:
        attrs.pop(k)
    node = helper.make_node(op_type, inputs, outputs, name=name, domain=domain, **attrs)
    for k, kind in empties:
        node.attribute.append(helper.make_attribute(k, [], attr_type=getattr(AttributeProto, kind)))
    return node


def make_onnx() -> onnx.ModelProto:
{node_statements}

    graph = helper.make_graph(
        nodes,
        GRAPH_NAME,
        [
{input_lines}
        ],
        [
{output_lines}
        ],
        initializer=[
{initializer_lines}
        ],
        value_info=[
{value_info_lines}
        ],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid(domain, version) for domain, version in OPSETS],
        producer_name=PRODUCER_NAME,
        producer_version=PRODUCER_VERSION,
        domain=DOMAIN,
        model_version=MODEL_VERSION,
        doc_string="",
    )
    model.ir_version = IR_VERSION
    onnx.checker.check_model(model)
    return model


def save_model(path: str | Path = OUT) -> None:
    onnx.save_model(make_onnx(), path)


if __name__ == "__main__":
    save_model(sys.argv[1] if len(sys.argv) > 1 else OUT)
'''


def literal(value, *, width: int = 100) -> str:
    return pprint.pformat(value, width=width, compact=True, sort_dicts=False)


def clean_number(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return value
    if isinstance(value, (bool, int, str)):
        return value
    raise TypeError(f"Unsupported scalar {type(value)!r}: {value!r}")


def tensor_spec(tensor):
    dtype = int(tensor.data_type)
    if dtype not in DTYPE_BY_TENSOR_TYPE:
        raise ValueError(f"Unsupported tensor dtype {dtype} for {tensor.name!r}")
    array = numpy_helper.to_array(tensor)
    if dtype == TensorProto.STRING:
        values = [
            value.decode("utf-8") if isinstance(value, (bytes, bytearray)) else str(value)
            for value in array.reshape(-1).tolist()
        ]
    else:
        values = [clean_number(value) for value in array.reshape(-1).tolist()]
    return {
        "name": tensor.name,
        "dtype": dtype,
        "shape": tuple(int(dim) for dim in array.shape),
        "values": values,
    }


def dim_spec(dim):
    if dim.HasField("dim_value"):
        return int(dim.dim_value)
    if dim.HasField("dim_param"):
        return str(dim.dim_param)
    return None


def value_info_spec(value_info):
    tensor_type = value_info.type.tensor_type
    if not tensor_type.HasField("elem_type"):
        raise ValueError(f"Missing elem_type for {value_info.name!r}")
    shape = tensor_type.shape
    return {
        "name": value_info.name,
        "dtype": int(tensor_type.elem_type),
        "shape": [dim_spec(dim) for dim in shape.dim],
    }


def attr_spec(attr):
    kind = AttributeProto.AttributeType.Name(attr.type)
    if attr.type == AttributeProto.FLOAT:
        value = clean_number(float(attr.f))
    elif attr.type == AttributeProto.INT:
        value = int(attr.i)
    elif attr.type == AttributeProto.STRING:
        value = attr.s.decode("utf-8")
    elif attr.type == AttributeProto.FLOATS:
        value = [clean_number(float(item)) for item in attr.floats]
    elif attr.type == AttributeProto.INTS:
        value = [int(item) for item in attr.ints]
    elif attr.type == AttributeProto.STRINGS:
        value = [item.decode("utf-8") for item in attr.strings]
    elif attr.type == AttributeProto.TENSOR:
        value = tensor_spec(attr.t)
    else:
        raise ValueError(f"Unsupported attribute type {kind} for {attr.name!r}")
    return {"kind": kind, "name": attr.name, "value": value}


def model_spec(task_id: str, model, leaderboard_entry: dict | None = None):
    return {
        "task_id": task_id,
        "task_num": int(task_id.removeprefix("task")),
        "ir_version": int(model.ir_version),
        "producer_name": model.producer_name,
        "producer_version": model.producer_version,
        "domain": model.domain,
        "model_version": int(model.model_version),
        "graph_name": model.graph.name,
        "opsets": [(op.domain, int(op.version)) for op in model.opset_import],
        "inputs": [value_info_spec(item) for item in model.graph.input],
        "outputs": [value_info_spec(item) for item in model.graph.output],
        "value_info": [value_info_spec(item) for item in model.graph.value_info],
        "leaderboard": leaderboard_entry or {},
        "initializers": [tensor_spec(item) for item in model.graph.initializer],
        "nodes": [
            {
                "op_type": node.op_type,
                "inputs": list(node.input),
                "outputs": list(node.output),
                "name": node.name,
                "domain": node.domain,
                "attrs": [attr_spec(attr) for attr in node.attribute],
            }
            for node in model.graph.node
        ],
    }


def indent(text: str, spaces: int) -> str:
    prefix = " " * spaces
    return "\n".join(prefix + line if line else line for line in text.splitlines())


def safe_name(name: str, fallback: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z_]+", "_", name).strip("_").lower()
    if not cleaned:
        cleaned = fallback
    if cleaned[0].isdigit():
        cleaned = f"_{cleaned}"
    return cleaned


def dtype_name(dtype: int) -> str:
    return f"TensorProto.{TensorProto.DataType.Name(dtype)}"


def shape_expr(shape) -> str:
    return repr(tuple(shape))


def tensor_expr(spec, *, name_override: str | None = None) -> str:
    name = spec["name"] if name_override is None else name_override
    return (
        f"_tensor({name!r}, {dtype_name(spec['dtype'])}, {shape_expr(spec['shape'])}, "
        f"{literal(spec['values'])})"
    )


def value_info_expr(spec) -> str:
    return f"_vi({spec['name']!r}, {dtype_name(spec['dtype'])}, {literal(spec['shape'])})"


def initializer_summary(specs) -> str:
    if not specs:
        return "none"
    pieces = []
    for item in specs[:8]:
        pieces.append(f"{item['name']}:{TensorProto.DataType.Name(item['dtype'])}{list(item['shape'])}")
    if len(specs) > 8:
        pieces.append(f"... +{len(specs) - 8} more")
    return ", ".join(pieces)


def value_summary(specs) -> str:
    if not specs:
        return "none"
    return ", ".join(f"{item['name']}:{TensorProto.DataType.Name(item['dtype'])}{item['shape']}" for item in specs)


def node_summary(specs) -> str:
    if not specs:
        return "none"
    counts = {}
    for node in specs:
        counts[node["op_type"]] = counts.get(node["op_type"], 0) + 1
    return ", ".join(f"{op}x{count}" for op, count in sorted(counts.items()))


def render_value_infos(specs, spaces: int = 8) -> str:
    if not specs:
        return ""
    prefix = " " * spaces
    return "\n".join(f"{prefix}{value_info_expr(spec)}," for spec in specs)


def initializer_stem(spec, index: int) -> str:
    return safe_name(spec["name"], f"tensor_{index}")


def initializer_function_name(spec, index: int) -> str:
    # Append the (unique) index so distinct initializers never collide — e.g. tensors named 'p' and 'P'
    # both stem to "p"/"P" and, before this, both produced the SAME builder constant (INIT_P), so the
    # second silently clobbered the first (task008: FLOAT[1] bias 'p' overwritten by FLOAT[3,3] 'P').
    return f"_init_{initializer_stem(spec, index)}_{index}"


def initializer_constant_name(spec, index: int) -> str:
    return f"INIT_{initializer_stem(spec, index).upper()}_{index}"


def render_initializer_constant(spec, index: int) -> str:
    const_name = initializer_constant_name(spec, index)
    comment = (
        f"# {spec['name']}: {TensorProto.DataType.Name(spec['dtype'])}{list(spec['shape'])}, "
        f"{len(spec['values'])} value(s)"
    )
    return f"{comment}\n{const_name} = {literal(spec['values'], width=120)}"


def render_initializer_constants(specs) -> str:
    if not specs:
        return "# No initializer constants."
    return "\n\n".join(render_initializer_constant(spec, index) for index, spec in enumerate(specs))


def render_initializer_function(spec, index: int) -> str:
    fn_name = initializer_function_name(spec, index)
    const_name = initializer_constant_name(spec, index)
    header = (
        f"def {fn_name}() -> onnx.TensorProto:\n"
        f"    # {spec['name']}: {TensorProto.DataType.Name(spec['dtype'])}{list(spec['shape'])}\n"
    )
    body = (
        f"return _tensor({spec['name']!r}, {dtype_name(spec['dtype'])}, "
        f"{shape_expr(spec['shape'])}, {const_name})"
    )
    return header + indent(body, 4)


def render_initializer_functions(specs) -> str:
    if not specs:
        return "# No initializers."
    return "\n\n".join(render_initializer_function(spec, index) for index, spec in enumerate(specs))


def render_initializer_inline_lines(specs, spaces: int = 12) -> str:
    if not specs:
        return ""
    prefix = " " * spaces
    lines = []
    for index, spec in enumerate(specs):
        lines.append(
            f"{prefix}_tensor({spec['name']!r}, {dtype_name(spec['dtype'])}, "
            f"{shape_expr(spec['shape'])}, {initializer_constant_name(spec, index)}),"
        )
    return "\n".join(lines)


def render_initializer_lines(specs) -> str:
    if not specs:
        return ""
    return "\n".join(f"        {initializer_function_name(spec, index)}()," for index, spec in enumerate(specs))


_NONFINITE_EXPR = {"nan": "float('nan')", "inf": "float('inf')", "-inf": "float('-inf')"}


def _float_token(x) -> str:
    # clean_number() encodes non-finite floats as the sentinel strings "nan"/"inf"/"-inf"
    # (so the spec stays JSON-safe). For FLOAT/FLOATS attributes there is no _num() wrapper
    # at build time, so render them back to real float(...) literals — otherwise a NaN weight
    # renders as the string 'nan' in a float list and onnx.helper.make_node cannot infer the
    # attribute type (e.g. task146's TfIdfVectorizer weights).
    if isinstance(x, str) and x in _NONFINITE_EXPR:
        return _NONFINITE_EXPR[x]
    return repr(float(x))


def attr_value_expr(attr, node_index: int) -> str:
    value = attr["value"]
    kind = attr["kind"]
    if kind == "TENSOR":
        return tensor_expr(value, name_override=value["name"])
    # Empty list-valued attributes (INTS/FLOATS/STRINGS) lose their type when passed as a bare
    # [] to helper.make_node; emit a typed sentinel that _node() re-adds with an explicit type.
    if kind in ("INTS", "FLOATS", "STRINGS") and len(value) == 0:
        return f"_EmptyAttr({kind!r})"
    if kind == "FLOAT":
        return _float_token(value)
    if kind == "FLOATS":
        return "[" + ", ".join(_float_token(x) for x in value) + "]"
    return literal(value)


def render_attrs(attrs, node_index: int) -> str:
    if not attrs:
        return "    attrs = {}\n"
    lines = ["    attrs = {"]
    for attr in attrs:
        expr = attr_value_expr(attr, node_index)
        lines.append(f"        {attr['name']!r}: {expr},")
    lines.append("    }\n")
    return "\n".join(lines)


def render_attr_kwarg(attr, node_index: int) -> str:
    name = attr["name"]
    expr = attr_value_expr(attr, node_index)
    if name.isidentifier() and not keyword.iskeyword(name):
        return f"            {name}={expr},"
    return f"            **{{{name!r}: {expr}}},"


def render_inline_node_call(node, index: int) -> str:
    lines = [
        "        _node(",
        f"            {node['op_type']!r},",
        indent(literal(node["inputs"]), 12) + ",",
        indent(literal(node["outputs"]), 12) + ",",
        f"            name={node['name']!r},",
        f"            domain={node['domain']!r},",
    ]
    for attr in node["attrs"]:
        lines.append(render_attr_kwarg(attr, index))
    lines.append("        ),")
    return "\n".join(lines)


def render_node_statements(specs) -> str:
    lines = ["    nodes = ["]
    for index, node in enumerate(specs):
        lines.append(f"        # {index}: {node['op_type']} inputs={len(node['inputs'])} outputs={len(node['outputs'])}")
        lines.append(render_inline_node_call(node, index))
    lines.append("    ]")
    return "\n".join(lines)


def node_function_name(node, index: int) -> str:
    stem = node["name"] or node["op_type"]
    return f"_node_{index}_{safe_name(stem, node['op_type'].lower())}"


def render_node_function(node, index: int) -> str:
    fn_name = node_function_name(node, index)
    lines = [
        f"def {fn_name}() -> onnx.NodeProto:",
        f"    # {node['op_type']} inputs={len(node['inputs'])} outputs={len(node['outputs'])}",
    ]
    lines.append(render_attrs(node["attrs"], index).rstrip())
    lines.extend(
        [
            "    return _node(",
            f"        {node['op_type']!r},",
            indent(literal(node["inputs"]), 8) + ",",
            indent(literal(node["outputs"]), 8) + ",",
            f"        name={node['name']!r},",
            f"        domain={node['domain']!r},",
            "        **attrs,",
            "    )",
        ]
    )
    return "\n".join(lines)


def render_node_functions(specs) -> str:
    if not specs:
        return "# No nodes."
    return "\n\n".join(render_node_function(node, index) for index, node in enumerate(specs))


def render_node_lines(specs) -> str:
    if not specs:
        return ""
    return "\n".join(f"        {node_function_name(node, index)}()," for index, node in enumerate(specs))


def kaggle_metadata(entry: dict) -> dict:
    if not entry:
        raise ValueError("missing safe champion leaderboard metadata")
    updated_at = entry.get("updated_at") or ""
    meta = {
        "score": entry.get("score"),
        "date": updated_at[:10] or None,
    }
    return meta


def _is_safe_champion_row(item: dict) -> bool:
    # Unsafe sidecars are exposed separately by the API. For this export, only
    # rows marked champion are safe-source metadata.
    return item.get("status") == "champion" and item.get("task_id", "").startswith("task")


def load_leaderboard_entries(path: Path | None) -> dict[str, dict]:
    if path is None:
        return {}
    data = json.loads(path.read_text())
    entries: dict[str, dict] = {}
    for item in data.get("tasks", []):
        if not _is_safe_champion_row(item):
            continue
        task_id = item["task_id"]
        previous = entries.get(task_id)
        if previous is None or (item.get("updated_at") or "") >= (previous.get("updated_at") or ""):
            entries[task_id] = item
    return entries


def render_builder(spec) -> str:
    return TEMPLATE.format(
        task_id=spec["task_id"],
        task_num=spec["task_num"],
        graph_name=spec["graph_name"],
        producer_name=spec["producer_name"],
        producer_version=spec["producer_version"],
        domain=spec["domain"],
        model_version=spec["model_version"],
        ir_version=spec["ir_version"],
        opsets=literal(spec["opsets"]),
        kaggle=literal(kaggle_metadata(spec["leaderboard"]), width=120),
        memory_bytes=spec["leaderboard"].get("memory_bytes"),
        params=spec["leaderboard"].get("params"),
        candidate_id=spec["leaderboard"].get("candidate_id"),
        submitted_by=spec["leaderboard"].get("submitted_by"),
        updated_at=spec["leaderboard"].get("updated_at"),
        initializer_constants=render_initializer_constants(spec["initializers"]),
        input_summary=value_summary(spec["inputs"]),
        output_summary=value_summary(spec["outputs"]),
        initializer_summary=initializer_summary(spec["initializers"]),
        node_summary=node_summary(spec["nodes"]),
        input_lines=render_value_infos(spec["inputs"], spaces=12),
        output_lines=render_value_infos(spec["outputs"], spaces=12),
        value_info_lines=render_value_infos(spec["value_info"], spaces=12),
        initializer_lines=render_initializer_inline_lines(spec["initializers"], spaces=12),
        node_statements=render_node_statements(spec["nodes"]),
    )


def write_builder(
    task_id: str,
    source_onnx: Path,
    out_dir: Path,
    flat_py_dir: Path | None = None,
    leaderboard_entries: dict[str, dict] | None = None,
) -> tuple[Path, Path, Path | None]:
    model = onnx.load(source_onnx)
    leaderboard_entry = (leaderboard_entries or {}).get(task_id)
    spec = model_spec(task_id, model, leaderboard_entry)
    task_dir = out_dir / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    solution_py = task_dir / "solution.py"
    built_onnx = task_dir / f"{task_id}.onnx"
    source = render_builder(spec)
    solution_py.write_text(source)

    flat_py = None
    if flat_py_dir is not None:
        flat_py_dir.mkdir(parents=True, exist_ok=True)
        flat_py = flat_py_dir / f"{task_id}.py"
        flat_py.write_text(source)

    subprocess.run([sys.executable, str(solution_py), str(built_onnx)], check=True)
    onnx.checker.check_model(onnx.load(built_onnx))
    return solution_py, built_onnx, flat_py


def extract_zip(zip_path: Path, extract_dir: Path) -> list[Path]:
    extract_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    with zipfile.ZipFile(zip_path) as archive:
        entries = sorted(name for name in archive.namelist() if name.endswith(".onnx"))
        for name in entries:
            task_id = Path(name).stem
            task_dir = extract_dir / task_id
            task_dir.mkdir(parents=True, exist_ok=True)
            destination = task_dir / f"{task_id}.onnx"
            with archive.open(name) as src, destination.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            paths.append(destination)
    return paths


def write_flat_zip(flat_py_dir: Path, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for py_path in sorted(flat_py_dir.glob("task*.py")):
            archive.write(py_path, f"solutions_py/{py_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--extract-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--flat-py-dir", type=Path)
    parser.add_argument("--flat-zip", type=Path)
    parser.add_argument("--leaderboard-json", type=Path)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    flat_py_dir = args.flat_py_dir
    if args.flat_zip is not None and flat_py_dir is None:
        flat_py_dir = args.flat_zip.parent / "solutions_py"

    leaderboard_entries = load_leaderboard_entries(args.leaderboard_json)

    onnx_paths = extract_zip(args.zip, args.extract_dir)
    if len(onnx_paths) != 400:
        raise SystemExit(f"expected 400 ONNX entries, found {len(onnx_paths)}")
    if args.leaderboard_json is not None:
        missing = sorted(path.stem for path in onnx_paths if path.stem not in leaderboard_entries)
        if missing:
            preview = ", ".join(missing[:10])
            raise SystemExit(
                f"leaderboard JSON is missing safe champion rows for {len(missing)} task(s): {preview}"
            )

    converted = []
    for onnx_path in onnx_paths[: args.limit or None]:
        task_id = onnx_path.stem
        solution_py, built_onnx, flat_py = write_builder(
            task_id, onnx_path, args.out_dir, flat_py_dir, leaderboard_entries
        )
        converted.append(
            {
                "task_id": task_id,
                "solution_py": str(solution_py),
                "onnx": str(built_onnx),
                "flat_py": str(flat_py) if flat_py is not None else None,
            }
        )
        print(f"{task_id}: wrote {solution_py} and {built_onnx}", flush=True)

    if args.flat_zip is not None:
        if flat_py_dir is None:
            raise SystemExit("--flat-zip requires --flat-py-dir or an inferred flat directory")
        write_flat_zip(flat_py_dir, args.flat_zip)
        print(f"flat_zip={args.flat_zip}")

    print(f"converted={len(converted)}")


if __name__ == "__main__":
    main()
