from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task247'
TASK_NUM = 247
KAGGLE = {'score': 20.581159, 'date': '2026-07-14'}
MEMORY_BYTES = 70
PARAMS = 13
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = ''
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task247_v19_noconst_expr_correct'
OPSETS = [('', 18)]


# colors: UINT8[10, 1, 1, 1], 10 value(s)
INIT_COLORS_0 = [6, 255, 1, 251, 253, 5, 254, 0, 2, 3]

# one: UINT8[], 1 value(s)
INIT_ONE_1 = [1]

# p: FLOAT[1, 1, 1, 1], 1 value(s)
INIT_P_2 = [0.36500000953674316]

# p065: FLOAT[1, 1, 1, 1], 1 value(s)
INIT_P065_3 = [0.6499999761581421]


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
            ['p065'],
            ['s0a'],
            name='',
            domain='',
            dtype=2,
            seed=2125634.0,
        ),
        # 1: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p065'],
            ['s2a'],
            name='',
            domain='',
            dtype=2,
            seed=229013536.0,
        ),
        # 2: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p065'],
            ['s0b'],
            name='',
            domain='',
            dtype=2,
            seed=-35056708.0,
        ),
        # 3: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p065'],
            ['s1b'],
            name='',
            domain='',
            dtype=2,
            seed=328606368.0,
        ),
        # 4: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p065'],
            ['s2b'],
            name='',
            domain='',
            dtype=2,
            seed=-1428772992.0,
        ),
        # 5: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p'],
            ['s0c'],
            name='',
            domain='',
            dtype=2,
            seed=670207232.0,
        ),
        # 6: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p'],
            ['s1c'],
            name='',
            domain='',
            dtype=2,
            seed=-92832960.0,
        ),
        # 7: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p065'],
            ['s2c'],
            name='',
            domain='',
            dtype=2,
            seed=-7715342.0,
        ),
        # 8: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p'],
            ['s0d'],
            name='',
            domain='',
            dtype=2,
            seed=388760960.0,
        ),
        # 9: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p'],
            ['s1d'],
            name='',
            domain='',
            dtype=2,
            seed=1015199424.0,
        ),
        # 10: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p'],
            ['s2d'],
            name='',
            domain='',
            dtype=2,
            seed=1023897280.0,
        ),
        # 11: Sub inputs=2 outputs=1
        _node(
            'Sub',
            ['s0a', 's0b'],
            ['nc0_sub'],
            name='',
            domain='',
        ),
        # 12: Add inputs=2 outputs=1
        _node(
            'Add',
            ['s0c', 's0d'],
            ['nc0_add'],
            name='',
            domain='',
        ),
        # 13: BitShift inputs=2 outputs=1
        _node(
            'BitShift',
            ['nc0_add', 's0d'],
            ['nc0_sh'],
            name='',
            domain='',
            direction='LEFT',
        ),
        # 14: BitwiseXor inputs=2 outputs=1
        _node(
            'BitwiseXor',
            ['nc0_sub', 'nc0_sh'],
            ['nc0_lab'],
            name='',
            domain='',
        ),
        # 15: Sub inputs=2 outputs=1
        _node(
            'Sub',
            ['s0b', 's1b'],
            ['nc1_sub'],
            name='',
            domain='',
        ),
        # 16: Add inputs=2 outputs=1
        _node(
            'Add',
            ['s1c', 's1d'],
            ['nc1_add'],
            name='',
            domain='',
        ),
        # 17: BitShift inputs=2 outputs=1
        _node(
            'BitShift',
            ['nc1_add', 's1d'],
            ['nc1_sh'],
            name='',
            domain='',
            direction='LEFT',
        ),
        # 18: BitwiseXor inputs=2 outputs=1
        _node(
            'BitwiseXor',
            ['nc1_sub', 'nc1_sh'],
            ['nc1_lab'],
            name='',
            domain='',
        ),
        # 19: Sub inputs=2 outputs=1
        _node(
            'Sub',
            ['s2a', 's2b'],
            ['nc2_sub'],
            name='',
            domain='',
        ),
        # 20: Add inputs=2 outputs=1
        _node(
            'Add',
            ['s2c', 's2d'],
            ['nc2_add'],
            name='',
            domain='',
        ),
        # 21: BitShift inputs=2 outputs=1
        _node(
            'BitShift',
            ['nc2_add', 's2d'],
            ['nc2_sh'],
            name='',
            domain='',
            direction='LEFT',
        ),
        # 22: BitwiseXor inputs=2 outputs=1
        _node(
            'BitwiseXor',
            ['nc2_sub', 'nc2_sh'],
            ['nc2_lab'],
            name='',
            domain='',
        ),
        # 23: Concat inputs=3 outputs=1
        _node(
            'Concat',
            ['nc0_lab', 'nc1_lab', 'nc2_lab'],
            ['labels'],
            name='',
            domain='',
            axis=3,
        ),
        # 24: Sub inputs=2 outputs=1
        _node(
            'Sub',
            ['colors', 'labels'],
            ['diffw'],
            name='',
            domain='',
        ),
        # 25: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p065'],
            ['d3'],
            name='',
            domain='',
            dtype=2,
            seed=5914772.0,
        ),
        # 26: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p'],
            ['x5'],
            name='',
            domain='',
            dtype=2,
            seed=-24597792.0,
        ),
        # 27: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p'],
            ['x6'],
            name='',
            domain='',
            dtype=2,
            seed=4775929.0,
        ),
        # 28: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p'],
            ['x4'],
            name='',
            domain='',
            dtype=2,
            seed=2670092032.0,
        ),
        # 29: Sub inputs=2 outputs=1
        _node(
            'Sub',
            ['d3', 'd3'],
            ['z'],
            name='',
            domain='',
        ),
        # 30: Concat inputs=9 outputs=1
        _node(
            'Concat',
            ['z', 'z', 'z', 'd3', 'x4', 'x5', 'x6', 'x6', 'x6'],
            ['rows'],
            name='',
            domain='',
            axis=2,
        ),
        # 31: ConvInteger inputs=4 outputs=1
        _node(
            'ConvInteger',
            ['rows', 'diffw', 'one', 'one'],
            ['output'],
            name='',
            domain='',
            kernel_shape=[1, 3],
            pads=[0, 2, 21, 29],
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
            _tensor('colors', TensorProto.UINT8, (10, 1, 1, 1), INIT_COLORS_0),
            _tensor('one', TensorProto.UINT8, (), INIT_ONE_1),
            _tensor('p', TensorProto.FLOAT, (1, 1, 1, 1), INIT_P_2),
            _tensor('p065', TensorProto.FLOAT, (1, 1, 1, 1), INIT_P065_3),
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
