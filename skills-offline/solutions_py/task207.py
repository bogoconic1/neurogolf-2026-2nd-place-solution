from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task207'
TASK_NUM = 207
KAGGLE = {'score': 21.362414, 'date': '2026-07-12'}
MEMORY_BYTES = 35
PARAMS = 3
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 8
PRODUCER_NAME = ''
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task207_signed_selected_color'
OPSETS = [('', 15)]


# pA: FLOAT[1, 1, 1, 1], 1 value(s)
INIT_PA_0 = [0.6759999990463257]

# pB: FLOAT[1, 1, 1, 1], 1 value(s)
INIT_PB_1 = [0.5299999713897705]

# bg: INT8[1, 1, 1, 1], 1 value(s)
INIT_BG_2 = [-1]


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
            ['pA'],
            ['q0'],
            name='',
            domain='',
            dtype=3,
            seed=439098880.0,
        ),
        # 1: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['pB'],
            ['q1'],
            name='',
            domain='',
            dtype=3,
            seed=639781632.0,
        ),
        # 2: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['pB'],
            ['q2'],
            name='',
            domain='',
            dtype=3,
            seed=4083807232.0,
        ),
        # 3: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['pA'],
            ['q3'],
            name='',
            domain='',
            dtype=3,
            seed=109102512.0,
        ),
        # 4: Concat inputs=2 outputs=1
        _node(
            'Concat',
            ['q0', 'q1'],
            ['row0'],
            name='',
            domain='',
            axis=3,
        ),
        # 5: Concat inputs=2 outputs=1
        _node(
            'Concat',
            ['q2', 'q3'],
            ['row1'],
            name='',
            domain='',
            axis=3,
        ),
        # 6: Concat inputs=2 outputs=1
        _node(
            'Concat',
            ['row0', 'row1'],
            ['mask01'],
            name='',
            domain='',
            axis=2,
        ),
        # 7: Shrink inputs=1 outputs=1
        _node(
            'Shrink',
            ['mask01'],
            ['x'],
            name='',
            domain='',
            bias=2.0,
            lambd=-0.5,
        ),
        # 8: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['pA'],
            ['c1'],
            name='',
            domain='',
            dtype=3,
            seed=10254543.0,
        ),
        # 9: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['pA'],
            ['c2'],
            name='',
            domain='',
            dtype=3,
            seed=107023.0,
        ),
        # 10: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['pA'],
            ['c3'],
            name='',
            domain='',
            dtype=3,
            seed=970113.0,
        ),
        # 11: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['pA'],
            ['c4'],
            name='',
            domain='',
            dtype=3,
            seed=731.0,
        ),
        # 12: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['pA'],
            ['c5'],
            name='',
            domain='',
            dtype=3,
            seed=1315061.0,
        ),
        # 13: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['pA'],
            ['c6'],
            name='',
            domain='',
            dtype=3,
            seed=461211.0,
        ),
        # 14: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['pA'],
            ['c7'],
            name='',
            domain='',
            dtype=3,
            seed=320866.0,
        ),
        # 15: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['pA'],
            ['c8'],
            name='',
            domain='',
            dtype=3,
            seed=833869.0,
        ),
        # 16: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['pA'],
            ['c9'],
            name='',
            domain='',
            dtype=3,
            seed=201088.0,
        ),
        # 17: Concat inputs=10 outputs=1
        _node(
            'Concat',
            ['bg', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9'],
            ['W'],
            name='',
            domain='',
            axis=0,
        ),
        # 18: ConvInteger inputs=2 outputs=1
        _node(
            'ConvInteger',
            ['x', 'W'],
            ['output'],
            name='',
            domain='',
            pads=[0, 0, 28, 28],
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
            _tensor('pA', TensorProto.FLOAT, (1, 1, 1, 1), INIT_PA_0),
            _tensor('pB', TensorProto.FLOAT, (1, 1, 1, 1), INIT_PB_1),
            _tensor('bg', TensorProto.INT8, (1, 1, 1, 1), INIT_BG_2),
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
