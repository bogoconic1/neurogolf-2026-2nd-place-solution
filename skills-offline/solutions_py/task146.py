from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task146'
TASK_NUM = 146
KAGGLE = {'score': 21.704163, 'date': '2026-07-13'}
MEMORY_BYTES = 23
PARAMS = 4
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = ''
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task146_shared_p01'
OPSETS = [('', 21)]


# p01: FLOAT[1, 1, 1, 1], 1 value(s)
INIT_P01_0 = [0.6269247531890869]

# p2: FLOAT[1, 1, 1, 1], 1 value(s)
INIT_P2_1 = [0.4540787935256958]

# scale: FLOAT[1, 1, 2, 1], 2 value(s)
INIT_SCALE_2 = [1.0, '-inf']


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
            ['p01'],
            ['s0'],
            name='',
            domain='',
            dtype=2,
            seed=89489168.0,
        ),
        # 1: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p01'],
            ['s1'],
            name='',
            domain='',
            dtype=2,
            seed=6424746.0,
        ),
        # 2: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p2'],
            ['s2'],
            name='',
            domain='',
            dtype=2,
            seed=21657620.0,
        ),
        # 3: Concat inputs=4 outputs=1
        _node(
            'Concat',
            ['s2', 's1', 's0', 's0'],
            ['bits'],
            name='',
            domain='',
            axis=2,
        ),
        # 4: DequantizeLinear inputs=2 outputs=1
        _node(
            'DequantizeLinear',
            ['bits', 'scale'],
            ['seed'],
            name='',
            domain='',
            axis=2,
            block_size=3,
        ),
        # 5: ConvTranspose inputs=2 outputs=1
        _node(
            'ConvTranspose',
            ['seed', 'input'],
            ['output'],
            name='',
            domain='',
            group=1,
            kernel_shape=[30, 30],
            pads=[6, 0, 3, 0],
            strides=[3, 1],
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
            _tensor('p01', TensorProto.FLOAT, (1, 1, 1, 1), INIT_P01_0),
            _tensor('p2', TensorProto.FLOAT, (1, 1, 1, 1), INIT_P2_1),
            _tensor('scale', TensorProto.FLOAT, (1, 1, 2, 1), INIT_SCALE_2),
        ],
        value_info=[
            _vi('s0', TensorProto.UINT8, [1, 1, 1, 1]),
            _vi('s1', TensorProto.UINT8, [1, 1, 1, 1]),
            _vi('s2', TensorProto.UINT8, [1, 1, 1, 1]),
            _vi('bits', TensorProto.UINT8, [1, 1, 4, 1]),
            _vi('seed', TensorProto.FLOAT, [1, 1, 4, 1]),
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
