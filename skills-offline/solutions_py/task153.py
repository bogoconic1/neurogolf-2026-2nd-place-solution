from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task153'
TASK_NUM = 153
KAGGLE = {'score': 20.939557, 'date': '2026-07-14'}
MEMORY_BYTES = 51
PARAMS = 7
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 8
PRODUCER_NAME = ''
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task153_seeded_rng_row_lookup'
OPSETS = [('', 21)]


# q: FLOAT[1, 1, 1, 1], 1 value(s)
INIT_Q_0 = [0.6499999761581421]

# src: INT8[1, 1, 1, 3], 3 value(s)
INIT_SRC_1 = [-61, -115, -39]

# scale1: FLOAT[], 1 value(s)
INIT_SCALE1_2 = [0.2750003933906555]

# scale2: FLOAT[], 1 value(s)
INIT_SCALE2_3 = [0.20317277312278748]

# zp10: INT8[], 1 value(s)
INIT_ZP10_4 = [10]


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
            ['q'],
            ['r0a'],
            name='rng_r0a',
            domain='',
            dtype=3,
            seed=15785524.0,
        ),
        # 1: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['q'],
            ['r0b'],
            name='rng_r0b',
            domain='',
            dtype=3,
            seed=-44357116.0,
        ),
        # 2: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['scale2'],
            ['r0c'],
            name='rng_r0c',
            domain='',
            dtype=3,
            seed=-1903991.0,
        ),
        # 3: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['q'],
            ['r1a'],
            name='rng_r1a',
            domain='',
            dtype=3,
            seed=-56020100.0,
        ),
        # 4: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['q'],
            ['r1b'],
            name='rng_r1b',
            domain='',
            dtype=3,
            seed=-8805757.0,
        ),
        # 5: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['scale2'],
            ['r1c'],
            name='rng_r1c',
            domain='',
            dtype=3,
            seed=-2118237440.0,
        ),
        # 6: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['q'],
            ['r2a'],
            name='rng_r2a',
            domain='',
            dtype=3,
            seed=-9752636.0,
        ),
        # 7: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['q'],
            ['r2b'],
            name='rng_r2b',
            domain='',
            dtype=3,
            seed=315596384.0,
        ),
        # 8: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['scale2'],
            ['r2c'],
            name='rng_r2c',
            domain='',
            dtype=3,
            seed=324807872.0,
        ),
        # 9: QLinearConv inputs=8 outputs=1
        _node(
            'QLinearConv',
            ['r0a', 'scale1', 'zp10', 'r0b', 'scale1', 'zp10', 'scale2', 'r0c'],
            ['code0'],
            name='ql_code0',
            domain='',
            kernel_shape=[1, 1],
        ),
        # 10: QLinearConv inputs=8 outputs=1
        _node(
            'QLinearConv',
            ['r1a', 'scale1', 'zp10', 'r1b', 'scale1', 'zp10', 'scale1', 'r1c'],
            ['code1'],
            name='ql_code1',
            domain='',
            kernel_shape=[1, 1],
        ),
        # 11: QLinearConv inputs=8 outputs=1
        _node(
            'QLinearConv',
            ['r2a', 'scale2', 'zp10', 'r2b', 'scale2', 'zp10', 'scale2', 'r2c'],
            ['code2'],
            name='ql_code2',
            domain='',
            kernel_shape=[1, 1],
        ),
        # 12: Concat inputs=3 outputs=1
        _node(
            'Concat',
            ['code0', 'code1', 'code2'],
            ['codes'],
            name='row_codes',
            domain='',
            axis=2,
        ),
        # 13: Mul inputs=2 outputs=1
        _node(
            'Mul',
            ['src', 'codes'],
            ['mask'],
            name='mask_lookup',
            domain='',
        ),
        # 14: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['q'],
            ['p2'],
            name='rng_p2',
            domain='',
            dtype=3,
            seed=-230069200.0,
        ),
        # 15: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['q'],
            ['p3'],
            name='rng_p3',
            domain='',
            dtype=3,
            seed=-48326312.0,
        ),
        # 16: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['q'],
            ['p4'],
            name='rng_p4',
            domain='',
            dtype=3,
            seed=-1013208320.0,
        ),
        # 17: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['q'],
            ['p5'],
            name='rng_p5',
            domain='',
            dtype=3,
            seed=-1807973376.0,
        ),
        # 18: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['q'],
            ['p6'],
            name='rng_p6',
            domain='',
            dtype=3,
            seed=-53209872.0,
        ),
        # 19: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['q'],
            ['p9'],
            name='rng_p9',
            domain='',
            dtype=3,
            seed=23324250.0,
        ),
        # 20: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['q'],
            ['n1'],
            name='rng_n1',
            domain='',
            dtype=3,
            seed=122091968.0,
        ),
        # 21: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['q'],
            ['n3'],
            name='rng_n3',
            domain='',
            dtype=3,
            seed=3119236.0,
        ),
        # 22: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['q'],
            ['n6'],
            name='rng_n6',
            domain='',
            dtype=3,
            seed=52559944.0,
        ),
        # 23: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['q'],
            ['n7'],
            name='rng_n7',
            domain='',
            dtype=3,
            seed=-129961368.0,
        ),
        # 24: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['q'],
            ['n8'],
            name='rng_n8',
            domain='',
            dtype=3,
            seed=-1517517056.0,
        ),
        # 25: Sub inputs=2 outputs=1
        _node(
            'Sub',
            ['p2', 'p2'],
            ['w0'],
            name='route_zero',
            domain='',
        ),
        # 26: Shrink inputs=1 outputs=1
        _node(
            'Shrink',
            ['n1'],
            ['w1'],
            name='route_neg1',
            domain='',
            bias=2.0,
            lambd=0.5,
        ),
        # 27: Sub inputs=2 outputs=1
        _node(
            'Sub',
            ['p3', 'n3'],
            ['w3'],
            name='route_3',
            domain='',
        ),
        # 28: Sub inputs=2 outputs=1
        _node(
            'Sub',
            ['p6', 'n6'],
            ['w6'],
            name='route_6',
            domain='',
        ),
        # 29: Shrink inputs=1 outputs=1
        _node(
            'Shrink',
            ['n7'],
            ['w7'],
            name='route_neg7',
            domain='',
            bias=2.0,
            lambd=0.5,
        ),
        # 30: Shrink inputs=1 outputs=1
        _node(
            'Shrink',
            ['n8'],
            ['w8'],
            name='route_neg8',
            domain='',
            bias=2.0,
            lambd=0.5,
        ),
        # 31: Concat inputs=10 outputs=1
        _node(
            'Concat',
            ['w0', 'w1', 'p2', 'w3', 'p4', 'p5', 'w6', 'w7', 'w8', 'p9'],
            ['route'],
            name='route_concat',
            domain='',
            axis=0,
        ),
        # 32: ConvInteger inputs=2 outputs=1
        _node(
            'ConvInteger',
            ['mask', 'route'],
            ['output'],
            name='emit',
            domain='',
            kernel_shape=[1, 1],
            pads=[0, 0, 27, 27],
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
            _tensor('q', TensorProto.FLOAT, (1, 1, 1, 1), INIT_Q_0),
            _tensor('src', TensorProto.INT8, (1, 1, 1, 3), INIT_SRC_1),
            _tensor('scale1', TensorProto.FLOAT, (), INIT_SCALE1_2),
            _tensor('scale2', TensorProto.FLOAT, (), INIT_SCALE2_3),
            _tensor('zp10', TensorProto.INT8, (), INIT_ZP10_4),
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
