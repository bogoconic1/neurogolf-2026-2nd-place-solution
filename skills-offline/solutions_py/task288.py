from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task288'
TASK_NUM = 288
KAGGLE = {'score': 20.255068, 'date': '2026-07-09'}
MEMORY_BYTES = 40
PARAMS = 75
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 8
PRODUCER_NAME = 'task288_updated_v4'
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task288_updated_v4_m40_p75'
OPSETS = [('', 12)]


# F: FLOAT[1, 1, 5, 11], 55 value(s)
INIT_F_0 = [142.0, 0.0, 46.0, 34.0, 0.0, 44.0, -50.0, -28.0, 46.0, 40.0, -4.0, 18.0, -2.0, 38.0, 48.0, -24.0, -8.0, -22.0, 48.0,
 38.0, -2.0, 18.0, -90.0, 68.0, -36.0, 40.0, 82.0, -112.0, 82.0, 40.0, -36.0, 34.0, -82.0, 6.0, -8.0, 14.0, -4.0, 44.0,
 -154.0, 44.0, 30.0, 6.0, -8.0, 38.0, 84.0, -48.0, 44.0, 14.0, -52.0, 364.0, -54.0, -18.0, 34.0, 4.0, 42.0]

# P: FLOAT[1, 1, 5, 2], 10 value(s)
INIT_P_1 = [-140.0, 174.0, 36.0, -38.0, -174.0, -48.0, -38.0, 34.0, -48.0, -20.0]

# B: FLOAT[10], 10 value(s)
INIT_B_2 = [-169.0, -81.0, -81.0, -81.0, -81.0, -81.0, -81.0, -81.0, -81.0, -81.0]


NP_DTYPE = {
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
}


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
    nodes = [
        # 0: ConvTranspose inputs=3 outputs=1
        _node(
            'ConvTranspose',
            ['P', 'input', 'B'],
            ['bias'],
            name='',
            domain='',
            kernel_shape=[30, 30],
            pads=[7, 3, 26, 27],
        ),
        # 1: ConvTranspose inputs=3 outputs=1
        _node(
            'ConvTranspose',
            ['F', 'input', 'bias'],
            ['output'],
            name='',
            domain='',
            kernel_shape=[30, 30],
            pads=[4, 5, 0, 5],
        ),
    ]

    graph = helper.make_graph(
        nodes,
        GRAPH_NAME,
        [
            _vi('input', TensorProto.FLOAT, [1, 10, 30, 30]),
        ],
        [
            _vi('output', TensorProto.FLOAT, [1, 10, 30, 30]),
        ],
        initializer=[
            _tensor('F', TensorProto.FLOAT, (1, 1, 5, 11), INIT_F_0),
            _tensor('P', TensorProto.FLOAT, (1, 1, 5, 2), INIT_P_1),
            _tensor('B', TensorProto.FLOAT, (10,), INIT_B_2),
        ],
        value_info=[

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
