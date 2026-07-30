from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task214'
TASK_NUM = 214
KAGGLE = {'score': 19.662462, 'date': '2026-07-15'}
MEMORY_BYTES = 171
PARAMS = 37
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = ''
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task214_shared_zero_point'
OPSETS = [('', 18)]


# W: FLOAT[1, 10, 1, 1], 10 value(s)
INIT_W_0 = [0.0, -128.0, -127.0, -64.0, 64.0, 108.0, 126.0, 127.0, 85.0, -43.0]

# sep: INT8[1, 1, 3, 1], 3 value(s)
INIT_SEP_1 = [108, 108, 108]

# mults: INT8[1, 2, 1, 1], 2 value(s)
INIT_MULTS_2 = [1, -122]

# scale: FLOAT[], 1 value(s)
INIT_SCALE_3 = [5.368551137507893e-05]

# zp: INT8[], 1 value(s)
INIT_ZP_4 = [0]

# qW: INT8[10, 2, 1, 1], 20 value(s)
INIT_QW_5 = [0, 0, -73, 48, -72, -36, -2, -73, 2, -73, 16, -64, 73, -11, 12, 65, 1, 74, -2, 74]


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
        # 0: Conv inputs=2 outputs=1
        _node(
            'Conv',
            ['input', 'W'],
            ['code_f'],
            name='',
            domain='',
            kernel_shape=[1, 1],
            pads=[0, 0, -27, -27],
        ),
        # 1: Cast inputs=1 outputs=1
        _node(
            'Cast',
            ['code_f'],
            ['code'],
            name='',
            domain='',
            to=3,
        ),
        # 2: Transpose inputs=1 outputs=1
        _node(
            'Transpose',
            ['code'],
            ['ct'],
            name='',
            domain='',
            perm=[0, 1, 3, 2],
        ),
        # 3: MaxPool inputs=1 outputs=1
        _node(
            'MaxPool',
            ['ct'],
            ['r90'],
            name='',
            domain='',
            kernel_shape=[1, 1],
            pads=[0, -2, 0, -2],
            strides=[1, -1],
        ),
        # 4: MaxPool inputs=1 outputs=1
        _node(
            'MaxPool',
            ['code'],
            ['r180'],
            name='',
            domain='',
            kernel_shape=[1, 1],
            pads=[-2, -2, -2, -2],
            strides=[-1, -1],
        ),
        # 5: Concat inputs=5 outputs=1
        _node(
            'Concat',
            ['code', 'sep', 'r90', 'sep', 'r180'],
            ['codeall'],
            name='',
            domain='',
            axis=3,
        ),
        # 6: Mul inputs=2 outputs=1
        _node(
            'Mul',
            ['codeall', 'mults'],
            ['feat'],
            name='',
            domain='',
        ),
        # 7: QLinearConv inputs=8 outputs=1
        _node(
            'QLinearConv',
            ['feat', 'scale', 'zp', 'qW', 'scale', 'zp', 'scale', 'zp'],
            ['output'],
            name='',
            domain='',
            kernel_shape=[1, 1],
            pads=[0, 0, 27, 19],
        ),
    ]

    graph = helper.make_graph(
        nodes,
        GRAPH_NAME,
        [
            _vi('input', TensorProto.FLOAT, [1, 10, 30, 30]),
        ],
        [
            _vi('output', TensorProto.INT8, [1, 10, 30, 30]),
        ],
        initializer=[
            _tensor('W', TensorProto.FLOAT, (1, 10, 1, 1), INIT_W_0),
            _tensor('sep', TensorProto.INT8, (1, 1, 3, 1), INIT_SEP_1),
            _tensor('mults', TensorProto.INT8, (1, 2, 1, 1), INIT_MULTS_2),
            _tensor('scale', TensorProto.FLOAT, (), INIT_SCALE_3),
            _tensor('zp', TensorProto.INT8, (), INIT_ZP_4),
            _tensor('qW', TensorProto.INT8, (10, 2, 1, 1), INIT_QW_5),
        ],
        value_info=[
            _vi('code_f', TensorProto.FLOAT, [1, 1, 3, 3]),
            _vi('code', TensorProto.INT8, [1, 1, 3, 3]),
            _vi('ct', TensorProto.INT8, [1, 1, 3, 3]),
            _vi('r90', TensorProto.INT8, [1, 1, 3, 3]),
            _vi('r180', TensorProto.INT8, [1, 1, 3, 3]),
            _vi('codeall', TensorProto.INT8, [1, 1, 3, 11]),
            _vi('feat', TensorProto.INT8, [1, 2, 3, 11]),
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
