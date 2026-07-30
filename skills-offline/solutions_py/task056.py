from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task056'
TASK_NUM = 56
KAGGLE = {'score': 22.166787, 'date': '2026-07-10'}
MEMORY_BYTES = 16
PARAMS = 1
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 8
PRODUCER_NAME = 'task056_seeded_rng'
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task056_seeded_rng_m16_p1'
OPSETS = [('', 15)]


# p: FLOAT[1, 1, 1, 1], 1 value(s)
INIT_P_0 = [0.7559999823570251]


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
        # 0: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p'],
            ['b1'],
            name='',
            domain='',
            dtype=3,
            seed=-2081042048.0,
        ),
        # 1: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p'],
            ['b2'],
            name='',
            domain='',
            dtype=3,
            seed=-1546299648.0,
        ),
        # 2: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p'],
            ['b6'],
            name='',
            domain='',
            dtype=3,
            seed=-47435104.0,
        ),
        # 3: Shrink inputs=1 outputs=1
        _node(
            'Shrink',
            ['b6'],
            ['x'],
            name='',
            domain='',
            bias=110.0,
            lambd=-1.0,
        ),
        # 4: Shrink inputs=1 outputs=1
        _node(
            'Shrink',
            ['b2'],
            ['z'],
            name='',
            domain='',
            bias=-127.0,
            lambd=-1.0,
        ),
        # 5: Sub inputs=2 outputs=1
        _node(
            'Sub',
            ['b1', 'z'],
            ['t'],
            name='',
            domain='',
        ),
        # 6: Concat inputs=10 outputs=1
        _node(
            'Concat',
            ['b6', 'b1', 'b2', 't', 'b6', 'b6', 'z', 'b6', 'b6', 'b6'],
            ['w'],
            name='',
            domain='',
            axis=0,
        ),
        # 7: ConvInteger inputs=2 outputs=1
        _node(
            'ConvInteger',
            ['x', 'w'],
            ['output'],
            name='',
            domain='',
            kernel_shape=[1, 1],
            pads=[0, 0, 29, 29],
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
            _tensor('p', TensorProto.FLOAT, (1, 1, 1, 1), INIT_P_0),
        ],
        value_info=[
            _vi('b1', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('b2', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('b6', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('x', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('z', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('t', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('w', TensorProto.INT8, [10, 1, 1, 1]),
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
