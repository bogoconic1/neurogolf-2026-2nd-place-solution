from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task174'
TASK_NUM = 174
KAGGLE = {'score': 20.810345, 'date': '2026-07-14'}
MEMORY_BYTES = 64
PARAMS = 2
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = ''
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task174_compact_stateful_bernoulli'
OPSETS = [('', 21)]


# negone: INT8[1, 1, 1, 1], 1 value(s)
INIT_NEGONE_0 = [-1]

# p: FLOAT[1, 1, 1, 1], 1 value(s)
INIT_P_1 = [0.6420000195503235]


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
            ['A4'],
            name='',
            domain='',
            dtype=3,
            seed=4622661.0,
        ),
        # 1: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p'],
            ['W5'],
            name='',
            domain='',
            dtype=3,
            seed=1471458.0,
        ),
        # 2: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p'],
            ['H'],
            name='',
            domain='',
            dtype=3,
            seed=254917.0,
        ),
        # 3: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p'],
            ['R0'],
            name='',
            domain='',
            dtype=3,
            seed=1486494.0,
        ),
        # 4: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p'],
            ['A3'],
            name='',
            domain='',
            dtype=3,
            seed=3741283.0,
        ),
        # 5: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p'],
            ['W4'],
            name='',
            domain='',
            dtype=3,
            seed=16714919.0,
        ),
        # 6: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p'],
            ['ROW1P'],
            name='',
            domain='',
            dtype=3,
            seed=8890.0,
        ),
        # 7: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p'],
            ['COL1P'],
            name='',
            domain='',
            dtype=3,
            seed=-2012611.0,
        ),
        # 8: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p'],
            ['COL2P'],
            name='',
            domain='',
            dtype=3,
            seed=1069367.0,
        ),
        # 9: Shrink inputs=1 outputs=1
        _node(
            'Shrink',
            ['R0'],
            ['row0'],
            name='',
            domain='',
            bias=3.0,
            lambd=-1.0,
        ),
        # 10: Shrink inputs=1 outputs=1
        _node(
            'Shrink',
            ['H'],
            ['col0'],
            name='',
            domain='',
            bias=3.0,
            lambd=-1.0,
        ),
        # 11: Add inputs=2 outputs=1
        _node(
            'Add',
            ['col0', 'ROW1P'],
            ['row1'],
            name='',
            domain='',
        ),
        # 12: Add inputs=2 outputs=1
        _node(
            'Add',
            ['A3', 'H'],
            ['row2'],
            name='',
            domain='',
        ),
        # 13: Mul inputs=2 outputs=1
        _node(
            'Mul',
            ['W4', 'row1'],
            ['col3'],
            name='',
            domain='',
        ),
        # 14: Sub inputs=2 outputs=1
        _node(
            'Sub',
            ['COL1P', 'col3'],
            ['col1'],
            name='',
            domain='',
        ),
        # 15: Sub inputs=2 outputs=1
        _node(
            'Sub',
            ['COL2P', 'col3'],
            ['col2'],
            name='',
            domain='',
        ),
        # 16: Concat inputs=4 outputs=1
        _node(
            'Concat',
            ['row0', 'row1', 'row2', 'A4'],
            ['rows'],
            name='',
            domain='',
            axis=2,
        ),
        # 17: Concat inputs=5 outputs=1
        _node(
            'Concat',
            ['col0', 'col1', 'col2', 'col3', 'W5'],
            ['cols'],
            name='',
            domain='',
            axis=3,
        ),
        # 18: BitwiseAnd inputs=2 outputs=1
        _node(
            'BitwiseAnd',
            ['rows', 'cols'],
            ['signed_mask'],
            name='',
            domain='',
        ),
        # 19: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p'],
            ['z1'],
            name='',
            domain='',
            dtype=3,
            seed=545513.0,
        ),
        # 20: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p'],
            ['z2'],
            name='',
            domain='',
            dtype=3,
            seed=6819792.0,
        ),
        # 21: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p'],
            ['z3'],
            name='',
            domain='',
            dtype=3,
            seed=298001.0,
        ),
        # 22: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p'],
            ['z4'],
            name='',
            domain='',
            dtype=3,
            seed=9894571.0,
        ),
        # 23: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p'],
            ['z5'],
            name='',
            domain='',
            dtype=3,
            seed=66874428.0,
        ),
        # 24: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p'],
            ['z6'],
            name='',
            domain='',
            dtype=3,
            seed=4171935.0,
        ),
        # 25: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p'],
            ['z7'],
            name='',
            domain='',
            dtype=3,
            seed=1994264.0,
        ),
        # 26: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p'],
            ['z8'],
            name='',
            domain='',
            dtype=3,
            seed=16533189.0,
        ),
        # 27: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p'],
            ['z9'],
            name='',
            domain='',
            dtype=3,
            seed=684894.0,
        ),
        # 28: Concat inputs=10 outputs=1
        _node(
            'Concat',
            ['negone', 'z1', 'z2', 'z3', 'z4', 'z5', 'z6', 'z7', 'z8', 'z9'],
            ['dynW'],
            name='',
            domain='',
            axis=0,
        ),
        # 29: ConvInteger inputs=2 outputs=1
        _node(
            'ConvInteger',
            ['signed_mask', 'dynW'],
            ['output'],
            name='',
            domain='',
            pads=[0, 0, 26, 25],
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
            _tensor('negone', TensorProto.INT8, (1, 1, 1, 1), INIT_NEGONE_0),
            _tensor('p', TensorProto.FLOAT, (1, 1, 1, 1), INIT_P_1),
        ],
        value_info=[
            _vi('A4', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('W5', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('H', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('R0', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('A3', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('W4', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('ROW1P', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('COL1P', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('COL2P', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('row0', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('col0', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('row1', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('row2', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('col3', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('col1', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('col2', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('rows', TensorProto.INT8, [1, 1, 4, 1]),
            _vi('cols', TensorProto.INT8, [1, 1, 1, 5]),
            _vi('signed_mask', TensorProto.INT8, [1, 1, 4, 5]),
            _vi('z1', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('z2', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('z3', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('z4', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('z5', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('z6', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('z7', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('z8', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('z9', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('dynW', TensorProto.INT8, [10, 1, 1, 1]),
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
