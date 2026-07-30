from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task290'
TASK_NUM = 290
KAGGLE = {'score': 20.522663, 'date': '2026-07-14'}
MEMORY_BYTES = 84
PARAMS = 4
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 8
PRODUCER_NAME = ''
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task290_cost88'
OPSETS = [('', 21)]


# p_main: FLOAT[1, 1, 1, 1], 1 value(s)
INIT_P_MAIN_0 = [0.5889999866485596]

# qzero: INT8[1, 1, 1, 1], 1 value(s)
INIT_QZERO_1 = [0]

# one: INT8[1, 1, 1, 1], 1 value(s)
INIT_ONE_2 = [1]

# p_color: FLOAT[1, 1, 1, 1], 1 value(s)
INIT_P_COLOR_3 = [0.26303333044052124]


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
            ['p_main'],
            ['cv0'],
            name='color_source_0',
            domain='',
            dtype=3,
            seed=1091012.0,
        ),
        # 1: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p_main'],
            ['cv1'],
            name='color_source_1',
            domain='',
            dtype=3,
            seed=16232675.0,
        ),
        # 2: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p_main'],
            ['cv2'],
            name='color_source_2',
            domain='',
            dtype=3,
            seed=4924318.0,
        ),
        # 3: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p_color'],
            ['cv3'],
            name='color_source_3',
            domain='',
            dtype=3,
            seed=83581800.0,
        ),
        # 4: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p_color'],
            ['cv4'],
            name='color_source_4',
            domain='',
            dtype=3,
            seed=16127146.0,
        ),
        # 5: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p_color'],
            ['cv5'],
            name='color_source_5',
            domain='',
            dtype=3,
            seed=29923736.0,
        ),
        # 6: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p_color'],
            ['cv6'],
            name='color_source_6',
            domain='',
            dtype=3,
            seed=2050924.0,
        ),
        # 7: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p_main'],
            ['cv7'],
            name='color_source_7',
            domain='',
            dtype=3,
            seed=10345211.0,
        ),
        # 8: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p_main'],
            ['cv8'],
            name='color_source_8',
            domain='',
            dtype=3,
            seed=7272248.0,
        ),
        # 9: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p_color'],
            ['cv9'],
            name='color_source_9',
            domain='',
            dtype=3,
            seed=75724120.0,
        ),
        # 10: Sub inputs=2 outputs=1
        _node(
            'Sub',
            ['cv1', 'cv2'],
            ['q1'],
            name='color_q1',
            domain='',
        ),
        # 11: Sub inputs=2 outputs=1
        _node(
            'Sub',
            ['cv6', 'cv9'],
            ['q2'],
            name='color_q2',
            domain='',
        ),
        # 12: Sub inputs=2 outputs=1
        _node(
            'Sub',
            ['cv2', 'cv3'],
            ['q3'],
            name='color_q3',
            domain='',
        ),
        # 13: Sub inputs=2 outputs=1
        _node(
            'Sub',
            ['cv8', 'cv1'],
            ['q4'],
            name='color_q4',
            domain='',
        ),
        # 14: Sub inputs=2 outputs=1
        _node(
            'Sub',
            ['cv3', 'cv4'],
            ['q5'],
            name='color_q5',
            domain='',
        ),
        # 15: Sub inputs=2 outputs=1
        _node(
            'Sub',
            ['cv4', 'cv5'],
            ['q6'],
            name='color_q6',
            domain='',
        ),
        # 16: Sub inputs=2 outputs=1
        _node(
            'Sub',
            ['cv7', 'cv8'],
            ['q7'],
            name='color_q7',
            domain='',
        ),
        # 17: Sub inputs=2 outputs=1
        _node(
            'Sub',
            ['cv5', 'cv6'],
            ['q9'],
            name='color_q9',
            domain='',
        ),
        # 18: Concat inputs=10 outputs=1
        _node(
            'Concat',
            ['qzero', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'cv0', 'q9'],
            ['q'],
            name='color_q',
            domain='',
            axis=0,
        ),
        # 19: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p_main'],
            ['x'],
            name='',
            domain='',
            dtype=3,
            seed=384460320.0,
        ),
        # 20: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p_main'],
            ['y'],
            name='',
            domain='',
            dtype=3,
            seed=1518123904.0,
        ),
        # 21: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p_main'],
            ['z'],
            name='',
            domain='',
            dtype=3,
            seed=862754560.0,
        ),
        # 22: Sub inputs=2 outputs=1
        _node(
            'Sub',
            ['y', 'one'],
            ['g0'],
            name='',
            domain='',
        ),
        # 23: Add inputs=2 outputs=1
        _node(
            'Add',
            ['y', 'g0'],
            ['g1'],
            name='',
            domain='',
        ),
        # 24: Sub inputs=2 outputs=1
        _node(
            'Sub',
            ['x', 'z'],
            ['g2'],
            name='',
            domain='',
        ),
        # 25: BitwiseXor inputs=2 outputs=1
        _node(
            'BitwiseXor',
            ['one', 'g2'],
            ['g3'],
            name='',
            domain='',
        ),
        # 26: Sub inputs=2 outputs=1
        _node(
            'Sub',
            ['g3', 'g2'],
            ['g4'],
            name='',
            domain='',
        ),
        # 27: Concat inputs=6 outputs=1
        _node(
            'Concat',
            ['one', 'g1', 'g4', 'g2', 'y', 'z'],
            ['row'],
            name='',
            domain='',
            axis=2,
        ),
        # 28: Transpose inputs=1 outputs=1
        _node(
            'Transpose',
            ['row'],
            ['col'],
            name='col',
            domain='',
            perm=[0, 1, 3, 2],
        ),
        # 29: BitwiseAnd inputs=2 outputs=1
        _node(
            'BitwiseAnd',
            ['row', 'col'],
            ['spatial'],
            name='spatial',
            domain='',
        ),
        # 30: ConvInteger inputs=2 outputs=1
        _node(
            'ConvInteger',
            ['spatial', 'q'],
            ['output'],
            name='output',
            domain='',
            pads=[0, 0, 24, 24],
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
            _tensor('p_main', TensorProto.FLOAT, (1, 1, 1, 1), INIT_P_MAIN_0),
            _tensor('qzero', TensorProto.INT8, (1, 1, 1, 1), INIT_QZERO_1),
            _tensor('one', TensorProto.INT8, (1, 1, 1, 1), INIT_ONE_2),
            _tensor('p_color', TensorProto.FLOAT, (1, 1, 1, 1), INIT_P_COLOR_3),
        ],
        value_info=[
            _vi('cv0', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('cv1', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('cv2', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('cv3', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('cv4', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('cv5', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('cv6', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('cv7', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('cv8', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('cv9', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('q1', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('q2', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('q3', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('q4', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('q5', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('q6', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('q7', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('q9', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('q', TensorProto.INT8, [10, 1, 1, 1]),
            _vi('x', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('y', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('z', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('g0', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('g1', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('g2', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('g3', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('g4', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('row', TensorProto.INT8, [1, 1, 6, 1]),
            _vi('col', TensorProto.INT8, [1, 1, 1, 6]),
            _vi('spatial', TensorProto.INT8, [1, 1, 6, 6]),
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
