from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task071'
TASK_NUM = 71
KAGGLE = {'score': 19.686794, 'date': '2026-07-15'}
MEMORY_BYTES = 168
PARAMS = 35
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = ''
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task071_stateful_rng_gray_comp3_mirror_repair'
OPSETS = [('', 20)]


# p_mid: FLOAT[1], 1 value(s)
INIT_P_MID_0 = [0.49880000948905945]

# j4_base: UINT8[1, 4], 4 value(s)
INIT_J4_BASE_1 = [4, 6, 8, 10]

# base_src: INT32[30], 30 value(s)
INIT_BASE_SRC_2 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16]


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
            ['p_mid'],
            ['hb0_u8'],
            name='',
            domain='',
            dtype=2,
            seed=-1496506496.0,
        ),
        # 1: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p_mid'],
            ['hb1_u8'],
            name='',
            domain='',
            dtype=2,
            seed=105144824.0,
        ),
        # 2: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p_mid'],
            ['hb2_u8'],
            name='',
            domain='',
            dtype=2,
            seed=431527648.0,
        ),
        # 3: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p_mid'],
            ['qbit_u8'],
            name='',
            domain='',
            dtype=2,
            seed=202184688.0,
        ),
        # 4: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p_mid'],
            ['sidebit_u8'],
            name='',
            domain='',
            dtype=2,
            seed=216278384.0,
        ),
        # 5: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p_mid'],
            ['comp_u8'],
            name='',
            domain='',
            dtype=2,
            seed=-57665984.0,
        ),
        # 6: Shrink inputs=1 outputs=1
        _node(
            'Shrink',
            ['comp_u8'],
            ['comp3_u8'],
            name='',
            domain='',
            bias=-2.0,
            lambd=0.0,
        ),
        # 7: BitwiseXor inputs=2 outputs=1
        _node(
            'BitwiseXor',
            ['comp3_u8', 'qbit_u8'],
            ['zpc_u8'],
            name='',
            domain='',
        ),
        # 8: Shrink inputs=1 outputs=1
        _node(
            'Shrink',
            ['hb2_u8'],
            ['hhigh_u8'],
            name='hhigh_u8',
            domain='',
            bias=-249.0,
            lambd=-5.0,
        ),
        # 9: Add inputs=2 outputs=1
        _node(
            'Add',
            ['hb0_u8', 'hb1_u8'],
            ['hsum_u8'],
            name='hsum_u8',
            domain='',
        ),
        # 10: BitwiseXor inputs=2 outputs=1
        _node(
            'BitwiseXor',
            ['hhigh_u8', 'hsum_u8'],
            ['half_base'],
            name='half_base',
            domain='',
        ),
        # 11: Shrink inputs=1 outputs=1
        _node(
            'Shrink',
            ['sidebit_u8'],
            ['invside_u8'],
            name='',
            domain='',
            bias=1.0,
            lambd=-1.0,
        ),
        # 12: QLinearMatMul inputs=8 outputs=1
        _node(
            'QLinearMatMul',
            ['sidebit_u8', 'p_mid', 'invside_u8', 'j4_base', 'p_mid', 'comp_u8', 'p_mid', 'half_base'],
            ['pos_u8'],
            name='',
            domain='',
        ),
        # 13: Cast inputs=1 outputs=1
        _node(
            'Cast',
            ['pos_u8'],
            ['pos_i32'],
            name='',
            domain='',
            to=6,
        ),
        # 14: MatMulInteger inputs=4 outputs=1
        _node(
            'MatMulInteger',
            ['invside_u8', 'j4_base', 'sidebit_u8', 'zpc_u8'],
            ['delta_i32'],
            name='',
            domain='',
        ),
        # 15: ScatterElements inputs=3 outputs=1
        _node(
            'ScatterElements',
            ['base_src', 'pos_i32', 'delta_i32'],
            ['src'],
            name='',
            domain='',
            axis=0,
            reduction='add',
        ),
        # 16: Gather inputs=2 outputs=1
        _node(
            'Gather',
            ['input', 'src'],
            ['output'],
            name='',
            domain='',
            axis=3,
        ),
    ]

    graph = helper.make_graph(
        nodes,
        GRAPH_NAME,
        [
            _vi('input', TensorProto.FLOAT, [1, 10, 30, 30]),
        ],
        [
            _vi('output', TensorProto.FLOAT, [1, 10, 30, 30]),
        ],
        initializer=[
            _tensor('p_mid', TensorProto.FLOAT, (1,), INIT_P_MID_0),
            _tensor('j4_base', TensorProto.UINT8, (1, 4), INIT_J4_BASE_1),
            _tensor('base_src', TensorProto.INT32, (30,), INIT_BASE_SRC_2),
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
