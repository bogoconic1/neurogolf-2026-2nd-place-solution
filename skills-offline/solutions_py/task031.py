from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task031'
TASK_NUM = 31
KAGGLE = {'score': 19.894055, 'date': '2026-07-13'}
MEMORY_BYTES = 32
PARAMS = 133
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = ''
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task031_seeded_rng_start_memorizer'
OPSETS = [('', 21)]


# F: FLOAT[10], 10 value(s)
INIT_F_0 = [0.0, 127.93955993652344, 127.93955993652344, 127.93955993652344, 127.93955993652344, 127.93955993652344,
 127.93955993652344, 127.93955993652344, 127.93955993652344, 127.93955993652344]

# P2: FLOAT[30, 2], 60 value(s)
INIT_P2_1 = [3.054870367050171, 7.283378522515704e-07, -0.02716043032705784, 111.2491226196289, -0.05432086065411568,
 111.2491226196289, -0.10864172130823135, 111.2491226196289, -0.2172834426164627, 111.2491226196289,
 -0.4345668852329254, 111.2491226196289, -0.8691337704658508, 111.2491226196289, -1.7382675409317017, 111.2491226196289,
 -3.4765350818634033, 111.2491226196289, -6.953070163726807, 111.2491226196289, -13.906140327453613, 111.2491226196289,
 0.02386617474257946, 7.283378522515704e-07, 0.02386617474257946, 7.283378522515704e-07, 0.04773234948515892,
 7.283378522515704e-07, 0.04773234948515892, 7.283378522515704e-07, 0.04773234948515892, 7.283378522515704e-07,
 0.09546469897031784, 7.283378522515704e-07, 0.09546469897031784, 7.283378522515704e-07, 0.09546469897031784,
 7.283378522515704e-07, 0.19092939794063568, 7.283378522515704e-07, 0.19092939794063568, 7.283378522515704e-07,
 0.38185879588127136, 7.283378522515704e-07, 0.38185879588127136, 7.283378522515704e-07, 0.38185879588127136,
 7.283378522515704e-07, 0.7637175917625427, 7.283378522515704e-07, 0.7637175917625427, 7.283378522515704e-07,
 0.7637175917625427, 7.283378522515704e-07, 1.5274351835250854, 7.283378522515704e-07, 1.5274351835250854,
 7.283378522515704e-07, 1.5274351835250854, 7.283378522515704e-07]

# Q2: FLOAT[30, 2], 60 value(s)
INIT_Q2_2 = [11.241923332214355, 8.576906839152798e-05, 5.620961666107178, 8.576906839152798e-05, 2.810480833053589,
 8.576906839152798e-05, 1.4052404165267944, 8.576906839152798e-05, 0.7026202082633972, 8.576906839152798e-05,
 0.3513101041316986, 8.576906839152798e-05, 0.1756550520658493, 8.576906839152798e-05, 0.08782752603292465,
 8.576906839152798e-05, 0.043913763016462326, 8.576906839152798e-05, 0.021956881508231163, 8.576906839152798e-05,
 0.010978440754115582, 8.576906839152798e-05, 0.005489220377057791, 8.576906839152798e-05, 0.0027446101885288954,
 8.576906839152798e-05, 0.0013723050942644477, 8.576906839152798e-05, 0.0006861525471322238, 8.576906839152798e-05,
 0.0003430762735661119, 8.576906839152798e-05, 0.00017153813678305596, 8.576906839152798e-05, 8.576906839152798e-05,
 8.576906839152798e-05, 4.288453419576399e-05, 8.576906839152798e-05, 2.1442267097881995e-05, 8.576906839152798e-05,
 1.0721133548940998e-05, 8.576906839152798e-05, 5.360566774470499e-06, 8.576906839152798e-05, 2.6802833872352494e-06,
 8.576906839152798e-05, 1.3401416936176247e-06, 8.576906839152798e-05, 6.700708468088123e-07, 8.576906839152798e-05,
 3.350354234044062e-07, 8.576906839152798e-05, 1.675177117022031e-07, 8.576906839152798e-05, 8.375885585110154e-08,
 8.576906839152798e-05, 4.187942792555077e-08, 8.576906839152798e-05, 2.0939713962775386e-08, 8.576906839152798e-05]

# rng_p: FLOAT[1, 1], 1 value(s)
INIT_RNG_P_3 = [0.6060322523117065]

# pow_base: FLOAT[1, 2], 2 value(s)
INIT_POW_BASE_4 = [0.5, 1.0]


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
            ['rng_p'],
            ['bit0'],
            name='',
            domain='',
            dtype=10,
            seed=1513554944.0,
        ),
        # 1: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['rng_p'],
            ['bit1'],
            name='',
            domain='',
            dtype=10,
            seed=214939600.0,
        ),
        # 2: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['rng_p'],
            ['bit2'],
            name='',
            domain='',
            dtype=10,
            seed=2308625.0,
        ),
        # 3: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['rng_p'],
            ['bit3'],
            name='',
            domain='',
            dtype=10,
            seed=119514368.0,
        ),
        # 4: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['rng_p'],
            ['bit4'],
            name='',
            domain='',
            dtype=10,
            seed=1054115136.0,
        ),
        # 5: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['rng_p'],
            ['bit5'],
            name='',
            domain='',
            dtype=10,
            seed=13679099.0,
        ),
        # 6: Sum inputs=6 outputs=1
        _node(
            'Sum',
            ['bit0', 'bit1', 'bit1', 'bit2', 'bit2', 'bit2'],
            ['r_code_f16'],
            name='',
            domain='',
        ),
        # 7: Pow inputs=2 outputs=1
        _node(
            'Pow',
            ['pow_base', 'r_code_f16'],
            ['rowS2'],
            name='',
            domain='',
        ),
        # 8: Sum inputs=6 outputs=1
        _node(
            'Sum',
            ['bit3', 'bit4', 'bit4', 'bit5', 'bit5', 'bit5'],
            ['c_code_f16'],
            name='',
            domain='',
        ),
        # 9: Pow inputs=2 outputs=1
        _node(
            'Pow',
            ['pow_base', 'c_code_f16'],
            ['colS2'],
            name='',
            domain='',
        ),
        # 10: Einsum inputs=229 outputs=1
        _node(
            'Einsum',
            ['pow_base', 'pow_base', 'pow_base', 'pow_base', 'colS2', 'colS2', 'F', 'pow_base', 'pow_base',
             'pow_base', 'pow_base', 'pow_base', 'pow_base', 'colS2', 'input', 'P2', 'P2', 'P2', 'pow_base',
             'pow_base', 'pow_base', 'pow_base', 'pow_base', 'pow_base', 'pow_base', 'P2', 'colS2', 'Q2', 'Q2',
             'Q2', 'Q2', 'P2', 'colS2', 'Q2', 'pow_base', 'pow_base', 'pow_base', 'pow_base', 'pow_base',
             'pow_base', 'pow_base', 'pow_base', 'P2', 'colS2', 'Q2', 'pow_base', 'pow_base', 'pow_base',
             'pow_base', 'pow_base', 'pow_base', 'pow_base', 'pow_base', 'pow_base', 'P2', 'colS2', 'Q2',
             'pow_base', 'pow_base', 'pow_base', 'pow_base', 'pow_base', 'pow_base', 'pow_base', 'pow_base',
             'pow_base', 'pow_base', 'P2', 'colS2', 'Q2', 'pow_base', 'pow_base', 'pow_base', 'pow_base',
             'pow_base', 'pow_base', 'P2', 'colS2', 'Q2', 'pow_base', 'pow_base', 'pow_base', 'pow_base',
             'pow_base', 'pow_base', 'pow_base', 'P2', 'colS2', 'Q2', 'pow_base', 'pow_base', 'pow_base',
             'pow_base', 'pow_base', 'pow_base', 'pow_base', 'pow_base', 'P2', 'colS2', 'Q2', 'pow_base',
             'pow_base', 'pow_base', 'pow_base', 'pow_base', 'pow_base', 'pow_base', 'pow_base', 'P2', 'colS2',
             'Q2', 'pow_base', 'pow_base', 'pow_base', 'pow_base', 'pow_base', 'pow_base', 'rowS2', 'pow_base',
             'pow_base', 'pow_base', 'pow_base', 'pow_base', 'pow_base', 'rowS2', 'F', 'input', 'input', 'P2',
             'P2', 'pow_base', 'pow_base', 'P2', 'rowS2', 'Q2', 'Q2', 'Q2', 'P2', 'rowS2', 'Q2', 'P2', 'rowS2',
             'Q2', 'pow_base', 'pow_base', 'pow_base', 'P2', 'rowS2', 'Q2', 'pow_base', 'pow_base', 'pow_base',
             'pow_base', 'P2', 'rowS2', 'Q2', 'pow_base', 'pow_base', 'pow_base', 'pow_base', 'pow_base',
             'pow_base', 'P2', 'rowS2', 'Q2', 'pow_base', 'pow_base', 'pow_base', 'pow_base', 'pow_base',
             'pow_base', 'pow_base', 'P2', 'rowS2', 'Q2', 'pow_base', 'pow_base', 'pow_base', 'pow_base',
             'pow_base', 'pow_base', 'pow_base', 'pow_base', 'P2', 'rowS2', 'Q2', 'pow_base', 'pow_base',
             'pow_base', 'pow_base', 'pow_base', 'pow_base', 'pow_base', 'pow_base', 'pow_base', 'P2', 'rowS2',
             'Q2', 'pow_base', 'pow_base', 'pow_base', 'pow_base', 'pow_base', 'pow_base', 'pow_base',
             'pow_base', 'pow_base', 'pow_base', 'P2', 'rowS2', 'Q2', 'P2', 'P2', 'pow_base', 'pow_base',
             'pow_base', 'pow_base', 'pow_base', 'pow_base', 'rowS2', 'Q2', 'pow_base', 'pow_base', 'pow_base',
             'pow_base', 'pow_base', 'pow_base', 'Q2', 'rowS2'],
            ['output'],
            name='direct_scratch_shifted_rootpos_crop',
            domain='',
            equation='bQ,bQ,bQ,bQ,bQ,bP,d,bR,bR,bR,bR,bR,bR,bR,bdyw,wP,wQ,wR,bS,bS,bS,bS,bS,bS,bS,wS,bS,sP,sQ,sR,sS,wO,bO,sO,bT,bT,bT,bT,bT,bT,bT,bT,wT,bT,sT,bU,bU,bU,bU,bU,bU,bU,bU,bU,wU,bU,sU,bV,bV,bV,bV,bV,bV,bV,bV,bV,bV,wV,bV,sV,bW,bW,bW,bW,bW,bW,wW,bW,sW,bX,bX,bX,bX,bX,bX,bX,wX,bX,sX,bY,bY,bY,bY,bY,bY,bY,bY,wY,bY,sY,bZ,bZ,bZ,bZ,bZ,bZ,bZ,bZ,wZ,bZ,sZ,bM,bM,bM,bM,bM,bM,bM,bL,bL,bL,bL,bL,bL,bL,c,bchx,bohw,hM,hL,bC,bC,hC,bC,rL,rM,rC,hA,bA,rA,hB,bB,rB,bD,bD,bD,hD,bD,rD,bE,bE,bE,bE,hE,bE,rE,bF,bF,bF,bF,bF,bF,hF,bF,rF,bG,bG,bG,bG,bG,bG,bG,hG,bG,rG,bH,bH,bH,bH,bH,bH,bH,bH,hH,bH,rH,bI,bI,bI,bI,bI,bI,bI,bI,bI,hI,bI,rI,bJ,bJ,bJ,bJ,bJ,bJ,bJ,bJ,bJ,bJ,hJ,bJ,rJ,hK,hN,bN,bN,bN,bN,bN,bN,bN,rN,bK,bK,bK,bK,bK,bK,rK,bK->bors',
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
            _tensor('F', TensorProto.FLOAT, (10,), INIT_F_0),
            _tensor('P2', TensorProto.FLOAT, (30, 2), INIT_P2_1),
            _tensor('Q2', TensorProto.FLOAT, (30, 2), INIT_Q2_2),
            _tensor('rng_p', TensorProto.FLOAT, (1, 1), INIT_RNG_P_3),
            _tensor('pow_base', TensorProto.FLOAT, (1, 2), INIT_POW_BASE_4),
        ],
        value_info=[
            _vi('bit0', TensorProto.FLOAT16, [1, 1]),
            _vi('bit1', TensorProto.FLOAT16, [1, 1]),
            _vi('bit2', TensorProto.FLOAT16, [1, 1]),
            _vi('bit3', TensorProto.FLOAT16, [1, 1]),
            _vi('bit4', TensorProto.FLOAT16, [1, 1]),
            _vi('bit5', TensorProto.FLOAT16, [1, 1]),
            _vi('r_code_f16', TensorProto.FLOAT16, [1, 1]),
            _vi('c_code_f16', TensorProto.FLOAT16, [1, 1]),
            _vi('rowS2', TensorProto.FLOAT, [1, 2]),
            _vi('colS2', TensorProto.FLOAT, [1, 2]),
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
