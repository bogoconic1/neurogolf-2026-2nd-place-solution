from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task263'
TASK_NUM = 263
KAGGLE = {'score': 20.889126, 'date': '2026-07-11'}
MEMORY_BYTES = 56
PARAMS = 5
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = ''
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task263_seeded_rng'
OPSETS = [('', 18)]


# p057: FLOAT16[1, 1, 1, 1], 1 value(s)
INIT_P057_0 = [0.396484375]

# p2: DOUBLE[1, 1, 1, 1], 1 value(s)
INIT_P2_1 = [0.48]

# p6: FLOAT[1, 1, 1, 1], 1 value(s)
INIT_P6_2 = [0.642872154712677]

# neg: INT8[1, 1, 1, 1], 1 value(s)
INIT_NEG_3 = [-1]

# zero: INT8[1, 1, 1, 1], 1 value(s)
INIT_ZERO_4 = [0]


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
            ['p057'],
            ['o0'],
            name='',
            domain='',
            dtype=3,
            seed=329855392.0,
        ),
        # 1: RandomUniformLike inputs=1 outputs=1
        _node(
            'RandomUniformLike',
            ['p057'],
            ['p4'],
            name='',
            domain='',
            dtype=10,
            high=1.0,
            low=0.0,
            seed=721949.0,
        ),
        # 2: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p4'],
            ['o1'],
            name='',
            domain='',
            dtype=3,
            seed=2933807.0,
        ),
        # 3: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p2'],
            ['o2'],
            name='',
            domain='',
            dtype=3,
            seed=11782870.0,
        ),
        # 4: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p2'],
            ['o3'],
            name='',
            domain='',
            dtype=3,
            seed=192322064.0,
        ),
        # 5: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p4'],
            ['o4'],
            name='',
            domain='',
            dtype=3,
            seed=-608291264.0,
        ),
        # 6: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p057'],
            ['o5'],
            name='',
            domain='',
            dtype=3,
            seed=3645204.0,
        ),
        # 7: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p6'],
            ['o6'],
            name='',
            domain='',
            dtype=3,
            seed=-1243661184.0,
        ),
        # 8: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p057'],
            ['o7'],
            name='',
            domain='',
            dtype=3,
            seed=257066784.0,
        ),
        # 9: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p057'],
            ['o8'],
            name='',
            domain='',
            dtype=3,
            seed=1946696448.0,
        ),
        # 10: Concat inputs=3 outputs=1
        _node(
            'Concat',
            ['o0', 'o1', 'o2'],
            ['r0'],
            name='',
            domain='',
            axis=3,
        ),
        # 11: Concat inputs=3 outputs=1
        _node(
            'Concat',
            ['o3', 'o4', 'o5'],
            ['r1'],
            name='',
            domain='',
            axis=3,
        ),
        # 12: Concat inputs=3 outputs=1
        _node(
            'Concat',
            ['o6', 'o7', 'o8'],
            ['r2'],
            name='',
            domain='',
            axis=3,
        ),
        # 13: Concat inputs=3 outputs=1
        _node(
            'Concat',
            ['r0', 'r1', 'r2'],
            ['grid'],
            name='',
            domain='',
            axis=2,
        ),
        # 14: Shrink inputs=1 outputs=1
        _node(
            'Shrink',
            ['grid'],
            ['grid2'],
            name='',
            domain='',
            bias=-1.0,
            lambd=-1.0,
        ),
        # 15: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p6'],
            ['c2'],
            name='',
            domain='',
            dtype=3,
            seed=7927123.0,
        ),
        # 16: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p6'],
            ['c3'],
            name='',
            domain='',
            dtype=3,
            seed=28321.0,
        ),
        # 17: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p6'],
            ['c4'],
            name='',
            domain='',
            dtype=3,
            seed=13823583.0,
        ),
        # 18: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p6'],
            ['c5'],
            name='',
            domain='',
            dtype=3,
            seed=5376589.0,
        ),
        # 19: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p6'],
            ['c6'],
            name='',
            domain='',
            dtype=3,
            seed=3649841.0,
        ),
        # 20: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p6'],
            ['c7'],
            name='',
            domain='',
            dtype=3,
            seed=2362293.0,
        ),
        # 21: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p6'],
            ['c8'],
            name='',
            domain='',
            dtype=3,
            seed=5455878.0,
        ),
        # 22: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p6'],
            ['c9'],
            name='',
            domain='',
            dtype=3,
            seed=6545595.0,
        ),
        # 23: Concat inputs=10 outputs=1
        _node(
            'Concat',
            ['neg', 'zero', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9'],
            ['W'],
            name='',
            domain='',
            axis=0,
        ),
        # 24: ConvInteger inputs=2 outputs=1
        _node(
            'ConvInteger',
            ['grid2', 'W'],
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
            _tensor('p057', TensorProto.FLOAT16, (1, 1, 1, 1), INIT_P057_0),
            _tensor('p2', TensorProto.DOUBLE, (1, 1, 1, 1), INIT_P2_1),
            _tensor('p6', TensorProto.FLOAT, (1, 1, 1, 1), INIT_P6_2),
            _tensor('neg', TensorProto.INT8, (1, 1, 1, 1), INIT_NEG_3),
            _tensor('zero', TensorProto.INT8, (1, 1, 1, 1), INIT_ZERO_4),
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
