#!/usr/bin/env python3
"""Trace ONNX Einsum operand-order runtime proxies.

This is a lightweight diagnostic for high-arity NeuroGolf Einsum graphs. It
does not profile ONNX Runtime. Instead, it follows the explicit operand order in
the equation and estimates how large the live label set is after each pairwise
step. The result is useful for spotting reorderings that keep large axes alive
too long.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import onnx
from onnx import numpy_helper


ELLIPSIS_PREFIX = "@"


@dataclass(frozen=True)
class OperandTrace:
    index: int
    input_name: str
    subscript: str
    live: int
    reduce: str
    alive: str


@dataclass(frozen=True)
class EinsumTrace:
    label: str
    path: str
    node_index: int
    einsum_index: int
    equation: str
    operands: int
    dims: dict[str, int]
    peak_live: int
    sum_live: int
    threshold_counts: dict[str, int]
    rows: list[OperandTrace]


def _shape_from_value_info(value_info: onnx.ValueInfoProto) -> tuple[int, ...]:
    dims: list[int] = []
    for dim in value_info.type.tensor_type.shape.dim:
        if dim.dim_value:
            dims.append(int(dim.dim_value))
        else:
            raise ValueError(f"unresolved dimension in {value_info.name}")
    return tuple(dims)


def _shape_map(model: onnx.ModelProto) -> dict[str, tuple[int, ...]]:
    shapes: dict[str, tuple[int, ...]] = {}

    for initializer in model.graph.initializer:
        shapes[initializer.name] = tuple(numpy_helper.to_array(initializer).shape)

    for value_info in (
        list(model.graph.input)
        + list(model.graph.value_info)
        + list(model.graph.output)
    ):
        if value_info.type.HasField("tensor_type"):
            try:
                shapes.setdefault(value_info.name, _shape_from_value_info(value_info))
            except ValueError:
                pass

    return shapes


def _expand_ellipsis(subscript: str, rank: int) -> list[str]:
    if "..." not in subscript:
        return list(subscript)

    before, after = subscript.split("...", 1)
    ellipsis_count = rank - len(before) - len(after)
    if ellipsis_count < 0:
        raise ValueError(
            f"subscript {subscript!r} has rank larger than tensor rank {rank}"
        )

    return (
        list(before)
        + [f"{ELLIPSIS_PREFIX}{i}" for i in range(ellipsis_count)]
        + list(after)
    )


def _labels_to_text(labels: Iterable[str]) -> str:
    return "".join(sorted(labels))


def _size(labels: Iterable[str], dims: dict[str, int]) -> int:
    out = 1
    for label in labels:
        out *= dims[label]
    return out


def _threshold_label(threshold: int) -> str:
    if threshold % 1_000_000 == 0:
        return f"steps>={threshold // 1_000_000}m"
    if threshold % 1_000 == 0:
        return f"steps>={threshold // 1_000}k"
    return f"steps>={threshold}"


def _get_equation(node: onnx.NodeProto) -> str:
    for attr in node.attribute:
        if attr.name == "equation":
            return attr.s.decode()
    raise ValueError("Einsum node is missing equation attribute")


def trace_einsum(
    path: Path,
    *,
    label: str,
    target_einsum: int,
    thresholds: tuple[int, ...],
) -> EinsumTrace:
    trace_label = label
    model = onnx.shape_inference.infer_shapes(onnx.load(path))
    shapes = _shape_map(model)
    einsums = [
        (node_index, node)
        for node_index, node in enumerate(model.graph.node)
        if node.op_type == "Einsum"
    ]

    if not einsums:
        raise ValueError(f"no Einsum nodes found in {path}")
    if target_einsum < 0 or target_einsum >= len(einsums):
        raise ValueError(
            f"{path} has {len(einsums)} Einsum nodes; requested {target_einsum}"
        )

    node_index, node = einsums[target_einsum]
    equation = _get_equation(node)
    lhs, rhs = equation.split("->", 1)
    raw_subscripts = lhs.split(",")
    if len(raw_subscripts) != len(node.input):
        raise ValueError(
            f"Einsum has {len(raw_subscripts)} subscripts but "
            f"{len(node.input)} inputs"
        )

    missing = [
        name for name in list(node.input) + list(node.output) if name not in shapes
    ]
    if missing:
        raise ValueError(
            "shape inference did not resolve: " + ", ".join(sorted(missing))
        )

    subscripts = [
        _expand_ellipsis(raw_subscript, len(shapes[input_name]))
        for raw_subscript, input_name in zip(raw_subscripts, node.input)
    ]
    rhs_labels = set(_expand_ellipsis(rhs, len(shapes[node.output[0]])))

    dims: dict[str, int] = {}
    for input_name, subscript in zip(node.input, subscripts):
        for axis_label, dim in zip(subscript, shapes[input_name]):
            dims[axis_label] = max(dims.get(axis_label, 1), int(dim))

    last_use = {axis_label: -1 for axis_label in dims if axis_label not in rhs_labels}
    for index, subscript in enumerate(subscripts):
        for axis_label in set(subscript):
            if axis_label in last_use:
                last_use[axis_label] = index

    rows: list[OperandTrace] = []
    live_sizes: list[int] = []
    alive: set[str] = set()
    for index, (input_name, subscript) in enumerate(zip(node.input, subscripts)):
        before_reduce = alive | set(subscript)
        reduced = {label for label in before_reduce if last_use.get(label) == index}
        alive = before_reduce - reduced
        live = _size(alive, dims)
        live_sizes.append(live)
        rows.append(
            OperandTrace(
                index=index,
                input_name=input_name,
                subscript="".join(subscript),
                live=live,
                reduce=_labels_to_text(reduced),
                alive=_labels_to_text(alive),
            )
        )

    return EinsumTrace(
        label=trace_label,
        path=str(path),
        node_index=node_index,
        einsum_index=target_einsum,
        equation=equation,
        operands=len(subscripts),
        dims=dict(sorted(dims.items())),
        peak_live=max(live_sizes),
        sum_live=sum(live_sizes),
        threshold_counts={
            _threshold_label(threshold): sum(1 for live in live_sizes if live >= threshold)
            for threshold in thresholds
        },
        rows=rows,
    )


def _print_text(trace: EinsumTrace, *, summary_only: bool, limit: int | None) -> None:
    print(f"=== {trace.label} {trace.path}")
    print(f"node_index {trace.node_index} operands {trace.operands} eq")
    print(trace.equation)
    print("dims", trace.dims)
    threshold_text = " ".join(
        f"{name} {count}" for name, count in trace.threshold_counts.items()
    )
    print(
        "summary",
        "peak",
        trace.peak_live,
        "sum_live",
        trace.sum_live,
        threshold_text,
    )

    if summary_only:
        return

    print("operands:")
    rows = trace.rows if limit is None else trace.rows[:limit]
    for row in rows:
        print(
            f"{row.index:02d} "
            f"{row.input_name[:6]:<6} "
            f"{row.subscript:<6} "
            f"live={row.live:9d} "
            f"reduce={row.reduce:<8} "
            f"alive={row.alive}"
        )
    if limit is not None and len(trace.rows) > limit:
        print(f"... {len(trace.rows) - limit} operands omitted")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trace live-size and sum_live for each Einsum in an ONNX file."
    )
    parser.add_argument("onnx", type=Path, help="ONNX file to trace.")
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Print only equation, dims, and summary metrics.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Print at most N operand rows per traced Einsum.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)

    if args.limit is not None and args.limit < 0:
        print("ERROR: --limit must be non-negative", file=sys.stderr)
        return 2

    thresholds = (1_000_000, 5_000_000)
    traces: list[EinsumTrace] = []

    try:
        model = onnx.shape_inference.infer_shapes(onnx.load(args.onnx))
        einsum_count = sum(1 for node in model.graph.node if node.op_type == "Einsum")
        for target_einsum in range(einsum_count):
            traces.append(
                trace_einsum(
                    args.onnx,
                    label=args.onnx.stem,
                    target_einsum=target_einsum,
                    thresholds=thresholds,
                )
            )
    except Exception as exc:
        print(f"ERROR: {args.onnx}: {exc}", file=sys.stderr)
        return 1

    for index, trace in enumerate(traces):
        if index:
            print()
        _print_text(trace, summary_only=args.summary_only, limit=args.limit)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
