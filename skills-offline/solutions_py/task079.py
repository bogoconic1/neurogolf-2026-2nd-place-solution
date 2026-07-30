from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task079'
TASK_NUM = 79
KAGGLE = {'score': 20.922463, 'date': '2026-07-13'}
MEMORY_BYTES = 55
PARAMS = 4
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 8
PRODUCER_NAME = ''
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task079_rng_shrink59'
OPSETS = [('', 15)]


# neg1: INT8[1, 1, 1, 1], 1 value(s)
INIT_NEG1_0 = [-1]

# p0: FLOAT[1, 1, 1, 1], 1 value(s)
INIT_P0_1 = [0.4634449779987335]

# p1: FLOAT[1, 1, 1, 1], 1 value(s)
INIT_P1_2 = [0.4948031008243561]

# p2: FLOAT[1, 1, 1, 1], 1 value(s)
INIT_P2_3 = [0.6139536499977112]


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
            ['p1'],
            ['m00'],
            name='',
            domain='',
            dtype=3,
            seed=130661552.0,
        ),
        # 1: Shrink inputs=1 outputs=1
        _node(
            'Shrink',
            ['m00'],
            ['s0'],
            name='',
            domain='',
            bias=-1.0,
            lambd=-0.5,
        ),
        # 2: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p1'],
            ['m01'],
            name='',
            domain='',
            dtype=3,
            seed=-14810422.0,
        ),
        # 3: Shrink inputs=1 outputs=1
        _node(
            'Shrink',
            ['m01'],
            ['s1'],
            name='',
            domain='',
            bias=-1.0,
            lambd=-0.5,
        ),
        # 4: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p0'],
            ['n02'],
            name='',
            domain='',
            dtype=3,
            seed=2659262208.0,
        ),
        # 5: Shrink inputs=1 outputs=1
        _node(
            'Shrink',
            ['n02'],
            ['s2'],
            name='',
            domain='',
            bias=2.0,
            lambd=-0.5,
        ),
        # 6: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p2'],
            ['m10'],
            name='',
            domain='',
            dtype=3,
            seed=-901414336.0,
        ),
        # 7: Shrink inputs=1 outputs=1
        _node(
            'Shrink',
            ['m10'],
            ['s3'],
            name='',
            domain='',
            bias=-1.0,
            lambd=-0.5,
        ),
        # 8: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p0'],
            ['m11'],
            name='',
            domain='',
            dtype=3,
            seed=132693800.0,
        ),
        # 9: Shrink inputs=1 outputs=1
        _node(
            'Shrink',
            ['m11'],
            ['s4'],
            name='',
            domain='',
            bias=-1.0,
            lambd=-0.5,
        ),
        # 10: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p0'],
            ['m12'],
            name='',
            domain='',
            dtype=3,
            seed=1818615168.0,
        ),
        # 11: Shrink inputs=1 outputs=1
        _node(
            'Shrink',
            ['m12'],
            ['s5'],
            name='',
            domain='',
            bias=-1.0,
            lambd=-0.5,
        ),
        # 12: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p2'],
            ['m20'],
            name='',
            domain='',
            dtype=3,
            seed=2270369792.0,
        ),
        # 13: Shrink inputs=1 outputs=1
        _node(
            'Shrink',
            ['m20'],
            ['s6'],
            name='',
            domain='',
            bias=-1.0,
            lambd=-0.5,
        ),
        # 14: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p1'],
            ['m21'],
            name='',
            domain='',
            dtype=3,
            seed=-123147744.0,
        ),
        # 15: Shrink inputs=1 outputs=1
        _node(
            'Shrink',
            ['m21'],
            ['s7'],
            name='',
            domain='',
            bias=-1.0,
            lambd=-0.5,
        ),
        # 16: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p0'],
            ['m22'],
            name='',
            domain='',
            dtype=3,
            seed=8652254.0,
        ),
        # 17: Shrink inputs=1 outputs=1
        _node(
            'Shrink',
            ['m22'],
            ['s8'],
            name='',
            domain='',
            bias=-1.0,
            lambd=-0.5,
        ),
        # 18: Concat inputs=3 outputs=1
        _node(
            'Concat',
            ['s0', 's1', 's2'],
            ['r0'],
            name='',
            domain='',
            axis=3,
        ),
        # 19: Concat inputs=3 outputs=1
        _node(
            'Concat',
            ['s3', 's4', 's5'],
            ['r1'],
            name='',
            domain='',
            axis=3,
        ),
        # 20: Concat inputs=3 outputs=1
        _node(
            'Concat',
            ['s6', 's7', 's8'],
            ['r2'],
            name='',
            domain='',
            axis=3,
        ),
        # 21: Concat inputs=3 outputs=1
        _node(
            'Concat',
            ['r0', 'r1', 'r2'],
            ['tile'],
            name='',
            domain='',
            axis=2,
        ),
        # 22: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p2'],
            ['c1'],
            name='',
            domain='',
            dtype=3,
            seed=191694.0,
        ),
        # 23: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p2'],
            ['c2'],
            name='',
            domain='',
            dtype=3,
            seed=11008369.0,
        ),
        # 24: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p2'],
            ['c3'],
            name='',
            domain='',
            dtype=3,
            seed=258816032.0,
        ),
        # 25: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p2'],
            ['c4'],
            name='',
            domain='',
            dtype=3,
            seed=4674503.0,
        ),
        # 26: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p2'],
            ['c5'],
            name='',
            domain='',
            dtype=3,
            seed=388804.0,
        ),
        # 27: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p2'],
            ['c6'],
            name='',
            domain='',
            dtype=3,
            seed=4508234.0,
        ),
        # 28: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p2'],
            ['c7'],
            name='',
            domain='',
            dtype=3,
            seed=4631402.0,
        ),
        # 29: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p2'],
            ['c8'],
            name='',
            domain='',
            dtype=3,
            seed=4555407.0,
        ),
        # 30: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p2'],
            ['c9'],
            name='',
            domain='',
            dtype=3,
            seed=5539916.0,
        ),
        # 31: Concat inputs=10 outputs=1
        _node(
            'Concat',
            ['neg1', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9'],
            ['w'],
            name='',
            domain='',
            axis=0,
        ),
        # 32: ConvInteger inputs=2 outputs=1
        _node(
            'ConvInteger',
            ['tile', 'w'],
            ['output'],
            name='',
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
            _tensor('neg1', TensorProto.INT8, (1, 1, 1, 1), INIT_NEG1_0),
            _tensor('p0', TensorProto.FLOAT, (1, 1, 1, 1), INIT_P0_1),
            _tensor('p1', TensorProto.FLOAT, (1, 1, 1, 1), INIT_P1_2),
            _tensor('p2', TensorProto.FLOAT, (1, 1, 1, 1), INIT_P2_3),
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
