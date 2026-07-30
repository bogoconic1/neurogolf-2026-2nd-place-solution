from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task339'
TASK_NUM = 339
KAGGLE = {'score': 21.416481, 'date': '2026-07-12'}
MEMORY_BYTES = 24
PARAMS = 12
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = ''
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task339_seeded_rng_m24_p12'
OPSETS = [('', 15)]


# pL: FLOAT[1], 1 value(s)
INIT_PL_0 = [0.4690000116825104]

# pC: FLOAT[1, 1, 1, 1], 1 value(s)
INIT_PC_1 = [0.800000011920929]

# zero: UINT8[1, 1, 1, 1], 1 value(s)
INIT_ZERO_2 = [0]

# X: UINT8[1, 1, 1, 9], 9 value(s)
INIT_X_3 = [14, 13, 11, 10, 9, 8, 7, 6, 5]


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
            ['pL'],
            ['b0'],
            name='',
            domain='',
            dtype=2,
            seed=19742562.0,
        ),
        # 1: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['pL'],
            ['b1'],
            name='',
            domain='',
            dtype=2,
            seed=1699835.0,
        ),
        # 2: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['pL'],
            ['b2'],
            name='',
            domain='',
            dtype=2,
            seed=778169024.0,
        ),
        # 3: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['pL'],
            ['b3'],
            name='',
            domain='',
            dtype=2,
            seed=-12009449.0,
        ),
        # 4: Shrink inputs=1 outputs=1
        _node(
            'Shrink',
            ['b0'],
            ['t'],
            name='',
            domain='',
            bias=4.0,
            lambd=-2.0,
        ),
        # 5: QLinearMatMul inputs=8 outputs=1
        _node(
            'QLinearMatMul',
            ['b1', 'pL', 't', 'b2', 'pL', 't', 'pL', 'b3'],
            ['q'],
            name='',
            domain='',
        ),
        # 6: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['pC'],
            ['c1'],
            name='',
            domain='',
            dtype=2,
            seed=14171056.0,
        ),
        # 7: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['pC'],
            ['c2'],
            name='',
            domain='',
            dtype=2,
            seed=3161654.0,
        ),
        # 8: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['pC'],
            ['c3'],
            name='',
            domain='',
            dtype=2,
            seed=13633916.0,
        ),
        # 9: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['pC'],
            ['c4'],
            name='',
            domain='',
            dtype=2,
            seed=15811738.0,
        ),
        # 10: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['pC'],
            ['c5'],
            name='',
            domain='',
            dtype=2,
            seed=11536202.0,
        ),
        # 11: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['pC'],
            ['c7'],
            name='',
            domain='',
            dtype=2,
            seed=13643477.0,
        ),
        # 12: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['pC'],
            ['c8'],
            name='',
            domain='',
            dtype=2,
            seed=15729074.0,
        ),
        # 13: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['pC'],
            ['c9'],
            name='',
            domain='',
            dtype=2,
            seed=13635087.0,
        ),
        # 14: Concat inputs=10 outputs=1
        _node(
            'Concat',
            ['zero', 'c1', 'c2', 'c3', 'c4', 'c5', 'zero', 'c7', 'c8', 'c9'],
            ['W'],
            name='',
            domain='',
            axis=0,
        ),
        # 15: ConvInteger inputs=3 outputs=1
        _node(
            'ConvInteger',
            ['X', 'W', 'q'],
            ['output'],
            name='',
            domain='',
            kernel_shape=[1, 1],
            pads=[0, 0, 29, 21],
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
            _tensor('pL', TensorProto.FLOAT, (1,), INIT_PL_0),
            _tensor('pC', TensorProto.FLOAT, (1, 1, 1, 1), INIT_PC_1),
            _tensor('zero', TensorProto.UINT8, (1, 1, 1, 1), INIT_ZERO_2),
            _tensor('X', TensorProto.UINT8, (1, 1, 1, 9), INIT_X_3),
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
