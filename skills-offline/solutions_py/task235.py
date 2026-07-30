from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task235'
TASK_NUM = 235
KAGGLE = {'score': 20.905655, 'date': '2026-07-12'}
MEMORY_BYTES = 39
PARAMS = 21
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = ''
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task235_cost60'
OPSETS = [('', 10)]


# conv_bias: FLOAT[1], 1 value(s)
INIT_CONV_BIAS_0 = [126.0]

# mapper_w: INT8[10, 1, 1, 2], 20 value(s)
INIT_MAPPER_W_1 = [0, 0, 0, 0, -1, -126, 0, 1, 1, 126, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0]


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
        # 0: Conv inputs=3 outputs=1
        _node(
            'Conv',
            ['input', 'input', 'conv_bias'],
            ['code0_f'],
            name='',
            domain='',
            dilations=[3, 6],
            kernel_shape=[30, 30],
            pads=[3, 9, 55, 136],
            strides=[1, 1],
        ),
        # 1: Cast inputs=1 outputs=1
        _node(
            'Cast',
            ['code0_f'],
            ['q0'],
            name='',
            domain='',
            to=3,
        ),
        # 2: Conv inputs=3 outputs=1
        _node(
            'Conv',
            ['input', 'input', 'conv_bias'],
            ['code1_f'],
            name='',
            domain='',
            dilations=[3, 6],
            kernel_shape=[30, 30],
            pads=[3, 26, 55, 119],
            strides=[1, 1],
        ),
        # 3: Cast inputs=1 outputs=1
        _node(
            'Cast',
            ['code1_f'],
            ['q1'],
            name='',
            domain='',
            to=3,
        ),
        # 4: Conv inputs=3 outputs=1
        _node(
            'Conv',
            ['input', 'input', 'conv_bias'],
            ['code2_f'],
            name='',
            domain='',
            dilations=[3, 4],
            kernel_shape=[30, 30],
            pads=[3, 48, 55, 39],
            strides=[1, 1],
        ),
        # 5: Cast inputs=1 outputs=1
        _node(
            'Cast',
            ['code2_f'],
            ['q2'],
            name='',
            domain='',
            to=3,
        ),
        # 6: Concat inputs=3 outputs=1
        _node(
            'Concat',
            ['q0', 'q1', 'q2'],
            ['q_col'],
            name='',
            domain='',
            axis=2,
        ),
        # 7: Shrink inputs=1 outputs=1
        _node(
            'Shrink',
            ['q_col'],
            ['shrink_col'],
            name='',
            domain='',
            bias=127.5,
            lambd=0.0,
        ),
        # 8: Concat inputs=6 outputs=1
        _node(
            'Concat',
            ['q_col', 'shrink_col', 'q_col', 'shrink_col', 'q_col', 'shrink_col'],
            ['feat_i8'],
            name='',
            domain='',
            axis=3,
        ),
        # 9: ConvInteger inputs=2 outputs=1
        _node(
            'ConvInteger',
            ['feat_i8', 'mapper_w'],
            ['output'],
            name='',
            domain='',
            kernel_shape=[1, 2],
            pads=[0, 0, 27, 54],
            strides=[1, 2],
        ),
    ]

    graph = helper.make_graph(
        nodes,
        GRAPH_NAME,
        [
            _vi('input', TensorProto.FLOAT, [1, 10, 30, 30]),
        ],
        [
            _vi('output', TensorProto.INT32, [1, 10, 30, 30]),
        ],
        initializer=[
            _tensor('conv_bias', TensorProto.FLOAT, (1,), INIT_CONV_BIAS_0),
            _tensor('mapper_w', TensorProto.INT8, (10, 1, 1, 2), INIT_MAPPER_W_1),
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
