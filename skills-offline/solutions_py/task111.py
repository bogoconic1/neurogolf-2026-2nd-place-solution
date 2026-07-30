from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task111'
TASK_NUM = 111
KAGGLE = {'score': 20.956949, 'date': '2026-07-12'}
MEMORY_BYTES = 52
PARAMS = 5
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 8
PRODUCER_NAME = ''
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task111_shrink'
OPSETS = [('', 18)]


# two_i8: INT8[1, 1, 1, 1], 1 value(s)
INIT_TWO_I8_0 = [2]

# neg1: INT8[1, 1, 1, 1], 1 value(s)
INIT_NEG1_1 = [-1]

# zero: INT8[1, 1, 1, 1], 1 value(s)
INIT_ZERO_2 = [0]

# p3: FLOAT[1, 1, 1, 1], 1 value(s)
INIT_P3_3 = [0.5566664338111877]

# p5: FLOAT[1, 1, 1, 1], 1 value(s)
INIT_P5_4 = [0.47347819805145264]


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
            ['p5'],
            ['b00'],
            name='',
            domain='',
            dtype=3,
            seed=-641034944.0,
        ),
        # 1: Shrink inputs=1 outputs=1
        _node(
            'Shrink',
            ['b00'],
            ['b00s'],
            name='',
            domain='',
            bias=-1.0,
            lambd=-1.0,
        ),
        # 2: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p3'],
            ['b02'],
            name='',
            domain='',
            dtype=3,
            seed=953438016.0,
        ),
        # 3: Shrink inputs=1 outputs=1
        _node(
            'Shrink',
            ['b02'],
            ['b02s'],
            name='',
            domain='',
            bias=2.0,
            lambd=-1.0,
        ),
        # 4: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p3'],
            ['b10'],
            name='',
            domain='',
            dtype=3,
            seed=-121524616.0,
        ),
        # 5: Shrink inputs=1 outputs=1
        _node(
            'Shrink',
            ['b10'],
            ['b10s'],
            name='',
            domain='',
            bias=-1.0,
            lambd=-1.0,
        ),
        # 6: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p3'],
            ['b11'],
            name='',
            domain='',
            dtype=3,
            seed=6557712.0,
        ),
        # 7: Shrink inputs=1 outputs=1
        _node(
            'Shrink',
            ['b11'],
            ['b11s'],
            name='',
            domain='',
            bias=2.0,
            lambd=-1.0,
        ),
        # 8: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p3'],
            ['b12'],
            name='',
            domain='',
            dtype=3,
            seed=-605828352.0,
        ),
        # 9: Shrink inputs=1 outputs=1
        _node(
            'Shrink',
            ['b12'],
            ['b12s'],
            name='',
            domain='',
            bias=-1.0,
            lambd=-1.0,
        ),
        # 10: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p5'],
            ['b20'],
            name='',
            domain='',
            dtype=3,
            seed=-36791320.0,
        ),
        # 11: Shrink inputs=1 outputs=1
        _node(
            'Shrink',
            ['b20'],
            ['b20s'],
            name='',
            domain='',
            bias=-1.0,
            lambd=-1.0,
        ),
        # 12: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p3'],
            ['b21'],
            name='',
            domain='',
            dtype=3,
            seed=77334800.0,
        ),
        # 13: Shrink inputs=1 outputs=1
        _node(
            'Shrink',
            ['b21'],
            ['b21s'],
            name='',
            domain='',
            bias=2.0,
            lambd=-1.0,
        ),
        # 14: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p3'],
            ['b22'],
            name='',
            domain='',
            dtype=3,
            seed=2008296064.0,
        ),
        # 15: Shrink inputs=1 outputs=1
        _node(
            'Shrink',
            ['b22'],
            ['b22s'],
            name='',
            domain='',
            bias=-1.0,
            lambd=-1.0,
        ),
        # 16: Concat inputs=3 outputs=1
        _node(
            'Concat',
            ['b00s', 'two_i8', 'b02s'],
            ['r0'],
            name='',
            domain='',
            axis=3,
        ),
        # 17: Concat inputs=3 outputs=1
        _node(
            'Concat',
            ['b10s', 'b11s', 'b12s'],
            ['r1'],
            name='',
            domain='',
            axis=3,
        ),
        # 18: Concat inputs=3 outputs=1
        _node(
            'Concat',
            ['b20s', 'b21s', 'b22s'],
            ['r2'],
            name='',
            domain='',
            axis=3,
        ),
        # 19: Concat inputs=3 outputs=1
        _node(
            'Concat',
            ['r0', 'r1', 'r2'],
            ['tile'],
            name='',
            domain='',
            axis=2,
        ),
        # 20: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p3'],
            ['c1'],
            name='',
            domain='',
            dtype=3,
            seed=-432806752.0,
        ),
        # 21: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p3'],
            ['c2'],
            name='',
            domain='',
            dtype=3,
            seed=424107072.0,
        ),
        # 22: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p3'],
            ['c3'],
            name='',
            domain='',
            dtype=3,
            seed=162912.0,
        ),
        # 23: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p3'],
            ['c4'],
            name='',
            domain='',
            dtype=3,
            seed=-674870656.0,
        ),
        # 24: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p3'],
            ['c6'],
            name='',
            domain='',
            dtype=3,
            seed=182340160.0,
        ),
        # 25: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p3'],
            ['c7'],
            name='',
            domain='',
            dtype=3,
            seed=-25272326.0,
        ),
        # 26: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p3'],
            ['c8'],
            name='',
            domain='',
            dtype=3,
            seed=-487002752.0,
        ),
        # 27: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p3'],
            ['c9'],
            name='',
            domain='',
            dtype=3,
            seed=2793048064.0,
        ),
        # 28: Concat inputs=10 outputs=1
        _node(
            'Concat',
            ['neg1', 'c1', 'c2', 'c3', 'c4', 'zero', 'c6', 'c7', 'c8', 'c9'],
            ['w'],
            name='',
            domain='',
            axis=0,
        ),
        # 29: ConvInteger inputs=2 outputs=1
        _node(
            'ConvInteger',
            ['tile', 'w'],
            ['output'],
            name='',
            domain='',
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
            _tensor('two_i8', TensorProto.INT8, (1, 1, 1, 1), INIT_TWO_I8_0),
            _tensor('neg1', TensorProto.INT8, (1, 1, 1, 1), INIT_NEG1_1),
            _tensor('zero', TensorProto.INT8, (1, 1, 1, 1), INIT_ZERO_2),
            _tensor('p3', TensorProto.FLOAT, (1, 1, 1, 1), INIT_P3_3),
            _tensor('p5', TensorProto.FLOAT, (1, 1, 1, 1), INIT_P5_4),
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
