from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task391'
TASK_NUM = 391
KAGGLE = {'score': 20.889126, 'date': '2026-07-13'}
MEMORY_BYTES = 30
PARAMS = 31
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 8
PRODUCER_NAME = ''
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task391_rng_q2_shared0'
OPSETS = [('', 18)]


# p: FLOAT[1, 1, 1, 1], 1 value(s)
INIT_P_0 = [0.390625]

# s1: FLOAT[], 1 value(s)
INIT_S1_1 = [0.07500000298023224]

# z: INT8[], 1 value(s)
INIT_Z_2 = [-1]

# W1: INT8[2, 1, 4, 1], 8 value(s)
INIT_W1_3 = [113, -39, -69, -20, 82, 8, -113, 26]

# W2: INT8[10, 2, 1, 1], 20 value(s)
INIT_W2_4 = [-1, -1, -1, -4, 19, -20, 17, -11, -3, -1, 14, -21, 1, 0, -4, 2, -3, 2, 7, -14]


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
            ['b0'],
            name='',
            domain='',
            dtype=3,
            seed=-479615072.0,
        ),
        # 1: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p'],
            ['b1'],
            name='',
            domain='',
            dtype=3,
            seed=-78270816.0,
        ),
        # 2: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p'],
            ['b2'],
            name='',
            domain='',
            dtype=3,
            seed=-4278521344.0,
        ),
        # 3: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p'],
            ['b3'],
            name='',
            domain='',
            dtype=3,
            seed=5061082.0,
        ),
        # 4: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p'],
            ['b4'],
            name='',
            domain='',
            dtype=3,
            seed=1796339840.0,
        ),
        # 5: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p'],
            ['b5'],
            name='',
            domain='',
            dtype=3,
            seed=-25017042.0,
        ),
        # 6: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p'],
            ['b6'],
            name='',
            domain='',
            dtype=3,
            seed=-17150888.0,
        ),
        # 7: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p'],
            ['b7'],
            name='',
            domain='',
            dtype=3,
            seed=-40192648.0,
        ),
        # 8: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p'],
            ['b8'],
            name='',
            domain='',
            dtype=3,
            seed=-48057432.0,
        ),
        # 9: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p'],
            ['b9'],
            name='',
            domain='',
            dtype=3,
            seed=-2716102400.0,
        ),
        # 10: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p'],
            ['b10'],
            name='',
            domain='',
            dtype=3,
            seed=-153326720.0,
        ),
        # 11: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p'],
            ['b11'],
            name='',
            domain='',
            dtype=3,
            seed=12433420.0,
        ),
        # 12: Concat inputs=12 outputs=1
        _node(
            'Concat',
            ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 'b10', 'b11'],
            ['f'],
            name='',
            domain='',
            axis=2,
        ),
        # 13: QLinearConv inputs=8 outputs=1
        _node(
            'QLinearConv',
            ['f', 's1', 'z', 'W1', 's1', 'z', 's1', 'z'],
            ['h'],
            name='',
            domain='',
            kernel_shape=[4, 1],
            strides=[4, 1],
        ),
        # 14: QLinearConv inputs=8 outputs=1
        _node(
            'QLinearConv',
            ['h', 's1', 'z', 'W2', 's1', 'z', 's1', 'z'],
            ['output'],
            name='',
            domain='',
            kernel_shape=[1, 1],
            pads=[0, 0, 27, 29],
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
            _tensor('p', TensorProto.FLOAT, (1, 1, 1, 1), INIT_P_0),
            _tensor('s1', TensorProto.FLOAT, (), INIT_S1_1),
            _tensor('z', TensorProto.INT8, (), INIT_Z_2),
            _tensor('W1', TensorProto.INT8, (2, 1, 4, 1), INIT_W1_3),
            _tensor('W2', TensorProto.INT8, (10, 2, 1, 1), INIT_W2_4),
        ],
        value_info=[
            _vi('b0', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('b1', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('b2', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('b3', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('b4', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('b5', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('b6', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('b7', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('b8', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('b9', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('b10', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('b11', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('f', TensorProto.INT8, [1, 1, 12, 1]),
            _vi('h', TensorProto.INT8, [1, 2, 3, 1]),
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
