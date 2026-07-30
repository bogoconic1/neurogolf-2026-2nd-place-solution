from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task394'
TASK_NUM = 394
KAGGLE = {'score': 20.992667, 'date': '2026-07-14'}
MEMORY_BYTES = 53
PARAMS = 2
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = ''
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task394_v16'
OPSETS = [('', 18)]


# p: FLOAT[1, 1, 1, 1], 1 value(s)
INIT_P_0 = [0.6499999761581421]

# zero: INT8[1, 1, 1, 1], 1 value(s)
INIT_ZERO_1 = [0]


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
            ['ba0'],
            name='ba0',
            domain='',
            dtype=3,
            seed=2231098880.0,
        ),
        # 1: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p'],
            ['ba1'],
            name='ba1',
            domain='',
            dtype=3,
            seed=2577687808.0,
        ),
        # 2: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p'],
            ['ba3'],
            name='ba3',
            domain='',
            dtype=3,
            seed=37760668.0,
        ),
        # 3: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p'],
            ['ba4'],
            name='ba4',
            domain='',
            dtype=3,
            seed=32417346.0,
        ),
        # 4: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p'],
            ['bsize'],
            name='bsize',
            domain='',
            dtype=3,
            seed=437422.0,
        ),
        # 5: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p'],
            ['bw1'],
            name='bw1',
            domain='',
            dtype=3,
            seed=254917.0,
        ),
        # 6: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p'],
            ['bw2p'],
            name='bw2p',
            domain='',
            dtype=3,
            seed=17778300.0,
        ),
        # 7: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p'],
            ['bw24'],
            name='bw24',
            domain='',
            dtype=3,
            seed=9414109.0,
        ),
        # 8: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p'],
            ['bw3'],
            name='bw3',
            domain='',
            dtype=3,
            seed=32211756.0,
        ),
        # 9: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p'],
            ['bw4n'],
            name='bw4n',
            domain='',
            dtype=3,
            seed=24126524.0,
        ),
        # 10: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p'],
            ['bw5'],
            name='bw5',
            domain='',
            dtype=3,
            seed=86649584.0,
        ),
        # 11: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p'],
            ['bw6'],
            name='bw6',
            domain='',
            dtype=3,
            seed=3895923.0,
        ),
        # 12: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p'],
            ['bw7'],
            name='bw7',
            domain='',
            dtype=3,
            seed=392254.0,
        ),
        # 13: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p'],
            ['bw8'],
            name='bw8',
            domain='',
            dtype=3,
            seed=6528198.0,
        ),
        # 14: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p'],
            ['bw9'],
            name='bw9',
            domain='',
            dtype=3,
            seed=7062705.0,
        ),
        # 15: Shrink inputs=1 outputs=1
        _node(
            'Shrink',
            ['ba1'],
            ['s1'],
            name='',
            domain='',
            bias=-1.0,
            lambd=-1.0,
        ),
        # 16: Shrink inputs=1 outputs=1
        _node(
            'Shrink',
            ['ba3'],
            ['s3'],
            name='',
            domain='',
            bias=-1.0,
            lambd=-1.0,
        ),
        # 17: Shrink inputs=1 outputs=1
        _node(
            'Shrink',
            ['ba4'],
            ['s4'],
            name='',
            domain='',
            bias=-1.0,
            lambd=-1.0,
        ),
        # 18: Shrink inputs=1 outputs=1
        _node(
            'Shrink',
            ['ba0'],
            ['s0'],
            name='',
            domain='',
            bias=2.0,
            lambd=-1.0,
        ),
        # 19: Shrink inputs=1 outputs=1
        _node(
            'Shrink',
            ['bsize'],
            ['nsize'],
            name='',
            domain='',
            bias=2.0,
            lambd=0.0,
        ),
        # 20: Concat inputs=3 outputs=1
        _node(
            'Concat',
            ['s0', 's1', 'nsize'],
            ['row0'],
            name='',
            domain='',
            axis=3,
        ),
        # 21: Concat inputs=3 outputs=1
        _node(
            'Concat',
            ['s3', 's4', 'bsize'],
            ['row1'],
            name='',
            domain='',
            axis=3,
        ),
        # 22: Concat inputs=3 outputs=1
        _node(
            'Concat',
            ['nsize', 'bsize', 'nsize'],
            ['row2'],
            name='',
            domain='',
            axis=3,
        ),
        # 23: Concat inputs=3 outputs=1
        _node(
            'Concat',
            ['row0', 'row1', 'row2'],
            ['spatial3'],
            name='',
            domain='',
            axis=2,
        ),
        # 24: Sub inputs=2 outputs=1
        _node(
            'Sub',
            ['bw2p', 'bw24'],
            ['w2'],
            name='',
            domain='',
        ),
        # 25: Sub inputs=2 outputs=1
        _node(
            'Sub',
            ['bw24', 'bw4n'],
            ['w4'],
            name='',
            domain='',
        ),
        # 26: Shrink inputs=1 outputs=1
        _node(
            'Shrink',
            ['bw7'],
            ['w7'],
            name='',
            domain='',
            bias=2.0,
            lambd=0.0,
        ),
        # 27: Shrink inputs=1 outputs=1
        _node(
            'Shrink',
            ['bw8'],
            ['w8'],
            name='',
            domain='',
            bias=2.0,
            lambd=0.0,
        ),
        # 28: Shrink inputs=1 outputs=1
        _node(
            'Shrink',
            ['bw9'],
            ['w9'],
            name='',
            domain='',
            bias=2.0,
            lambd=0.0,
        ),
        # 29: Concat inputs=10 outputs=1
        _node(
            'Concat',
            ['zero', 'bw1', 'w2', 'bw3', 'w4', 'bw5', 'bw6', 'w7', 'w8', 'w9'],
            ['weights10'],
            name='',
            domain='',
            axis=0,
        ),
        # 30: ConvInteger inputs=2 outputs=1
        _node(
            'ConvInteger',
            ['spatial3', 'weights10'],
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
            _tensor('p', TensorProto.FLOAT, (1, 1, 1, 1), INIT_P_0),
            _tensor('zero', TensorProto.INT8, (1, 1, 1, 1), INIT_ZERO_1),
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
