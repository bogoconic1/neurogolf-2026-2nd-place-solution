from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task039'
TASK_NUM = 39
KAGGLE = {'score': 20.317869, 'date': '2026-07-15'}
MEMORY_BYTES = 26
PARAMS = 82
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = 'task039_seeded_rng_answer_memorizer'
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task039_seeded_rng_pair'
OPSETS = [('', 18)]


# U: FLOAT[30, 2], 60 value(s)
INIT_U_0 = [1.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# one_u8: UINT8[1], 1 value(s)
INIT_ONE_U8_1 = [1]

# G0: FLOAT[2, 2, 2], 8 value(s)
INIT_G0_2 = [8.0, -5.0, 4.0, -3.0, -4.0, 3.0, 0.0, 0.0]

# T: FLOAT[2, 2], 4 value(s)
INIT_T_3 = [2.0, -1.5, 2.0, -1.0]

# M2: FLOAT[2, 2], 4 value(s)
INIT_M2_4 = [32.0, -17.0, -8.0, 5.0]

# M3: FLOAT[2, 2], 4 value(s)
INIT_M3_5 = [10.0, -13.0, -2.0, 5.0]

# p_half: FLOAT[1], 1 value(s)
INIT_P_HALF_6 = [0.5]


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
            ['p_half'],
            ['r_b0'],
            name='r_rng0',
            domain='',
            dtype=2,
            seed=9798152.0,
        ),
        # 1: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p_half'],
            ['r_b1'],
            name='r_rng1',
            domain='',
            dtype=2,
            seed=31846826.0,
        ),
        # 2: Add inputs=2 outputs=1
        _node(
            'Add',
            ['r_b0', 'r_b1'],
            ['r_idx_u8'],
            name='r_sum',
            domain='',
        ),
        # 3: Concat inputs=2 outputs=1
        _node(
            'Concat',
            ['one_u8', 'r_idx_u8'],
            ['r_feat_u8'],
            name='r_basis',
            domain='',
            axis=0,
        ),
        # 4: Cast inputs=1 outputs=1
        _node(
            'Cast',
            ['r_feat_u8'],
            ['r_feat'],
            name='r_to_float',
            domain='',
            to=1,
        ),
        # 5: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p_half'],
            ['c_b0'],
            name='c_rng0',
            domain='',
            dtype=2,
            seed=11791327.0,
        ),
        # 6: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p_half'],
            ['c_b1'],
            name='c_rng1',
            domain='',
            dtype=2,
            seed=19674590.0,
        ),
        # 7: Add inputs=2 outputs=1
        _node(
            'Add',
            ['c_b0', 'c_b1'],
            ['c_idx_u8'],
            name='c_sum',
            domain='',
        ),
        # 8: Concat inputs=2 outputs=1
        _node(
            'Concat',
            ['one_u8', 'c_idx_u8'],
            ['c_feat_u8'],
            name='c_basis',
            domain='',
            axis=0,
        ),
        # 9: Cast inputs=1 outputs=1
        _node(
            'Cast',
            ['c_feat_u8'],
            ['c_feat'],
            name='c_to_float',
            domain='',
            to=1,
        ),
        # 10: Einsum inputs=53 outputs=1
        _node(
            'Einsum',
            ['G0', 'T', 'G0', 'G0', 'T', 'G0', 'T', 'G0', 'r_feat', 'T', 'T', 'G0', 'c_feat', 'T', 'r_feat',
             'G0', 'G0', 'G0', 'M2', 'G0', 'U', 'U', 'U', 'U', 'U', 'U', 'U', 'U', 'U', 'M2', 'U', 'U', 'M3',
             'U', 'input', 'U', 'U', 'U', 'U', 'U', 'M2', 'M3', 'c_feat', 'G0', 'U', 'U', 'U', 'U', 'U', 'U',
             'G0', 'U', 'M2'],
            ['output'],
            name='',
            domain='',
            equation='vvv,xv,uvx,TTT,UT,STU,de,abd,m,lm,FG,DEF,N,MN,i,fgi,jkl,yzA,BC,KLM,rg,rk,ru,ry,rB,hb,hf,hj,ho,op,rp,hq,qt,rt,nchw,wE,wK,wH,wO,wQ,OP,QR,J,HIJ,sI,sL,sP,sR,sS,sV,VWX,sY,YZ->ncrs',
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
            _tensor('U', TensorProto.FLOAT, (30, 2), INIT_U_0),
            _tensor('one_u8', TensorProto.UINT8, (1,), INIT_ONE_U8_1),
            _tensor('G0', TensorProto.FLOAT, (2, 2, 2), INIT_G0_2),
            _tensor('T', TensorProto.FLOAT, (2, 2), INIT_T_3),
            _tensor('M2', TensorProto.FLOAT, (2, 2), INIT_M2_4),
            _tensor('M3', TensorProto.FLOAT, (2, 2), INIT_M3_5),
            _tensor('p_half', TensorProto.FLOAT, (1,), INIT_P_HALF_6),
        ],
        value_info=[
            _vi('r_b0', TensorProto.UINT8, [1]),
            _vi('r_b1', TensorProto.UINT8, [1]),
            _vi('r_idx_u8', TensorProto.UINT8, [1]),
            _vi('r_feat_u8', TensorProto.UINT8, [2]),
            _vi('r_feat', TensorProto.FLOAT, [2]),
            _vi('c_b0', TensorProto.UINT8, [1]),
            _vi('c_b1', TensorProto.UINT8, [1]),
            _vi('c_idx_u8', TensorProto.UINT8, [1]),
            _vi('c_feat_u8', TensorProto.UINT8, [2]),
            _vi('c_feat', TensorProto.FLOAT, [2]),
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
