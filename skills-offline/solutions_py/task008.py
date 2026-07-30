from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task008'
TASK_NUM = 8
KAGGLE = {'score': 20.044173, 'date': '2026-07-14'}
MEMORY_BYTES = 26
PARAMS = 116
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 8
PRODUCER_NAME = 'repeated-power-homotopy-factorization+expand-simplify-recompress'
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task008_m42_p108_tied_core'
OPSETS = [('', 21)]


# F: FLOAT[2, 2, 3, 2], 24 value(s)
INIT_F_0 = [2.2360680103302, -1.4142135381698608, -1.4142135381698608, 0.0, -1.9253736734390259, 1.4142135381698608,
 1.4142135381698608, 0.0, 0.0, 0.0, -1.4142135381698608, 0.0, 1.781399130821228, 0.05640333518385887, 0.0, 0.0,
 -1.8584187030792236, -1.0053974390029907, 0.0, 0.0, 0.0, 0.0, -0.6661007404327393, 0.0]

# V: FLOAT[2, 30], 60 value(s)
INIT_V_1 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0714285373687744,
 -1.0714285373687744, -1.0714285373687744, -1.0714285373687744, -1.0714285373687744, -1.0714285373687744,
 -1.0714285373687744, -1.0714285373687744, -1.0714285373687744, -1.0714285373687744, -1.0714285373687744,
 -1.0714285373687744, -1.0714285373687744, -1.0714285373687744, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0,
 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, -31.5, -31.5, -31.5, -31.5, -31.5, -31.5, -31.5, -31.5, -31.5, -31.5, -31.5,
 -31.5, -31.5, -31.5]

# U: FLOAT[2, 10], 20 value(s)
INIT_U_2 = [-8.0, 0.0, 16.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.002470348495990038, 0.0, 0.5, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 -0.6324092149734497, 0.0]

# p: FLOAT[1], 1 value(s)
INIT_P_3 = [0.351500004529953]

# C: FLOAT[2], 2 value(s)
INIT_C_4 = [0.5307174324989319, -0.3996802568435669]

# P: FLOAT[3, 3], 9 value(s)
INIT_P_5 = [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0]


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
            ['o'],
            name='rng_o',
            domain='',
            dtype=3,
            seed=5875895.0,
        ),
        # 1: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p'],
            ['a'],
            name='rng_a',
            domain='',
            dtype=3,
            seed=5128026.0,
        ),
        # 2: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p'],
            ['b'],
            name='rng_b',
            domain='',
            dtype=3,
            seed=253834352.0,
        ),
        # 3: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p'],
            ['c'],
            name='rng_c',
            domain='',
            dtype=3,
            seed=306401920.0,
        ),
        # 4: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p'],
            ['d4'],
            name='rng_d4',
            domain='',
            dtype=3,
            seed=82264888.0,
        ),
        # 5: Sub inputs=2 outputs=1
        _node(
            'Sub',
            ['b', 'a'],
            ['x'],
            name='',
            domain='',
        ),
        # 6: Add inputs=2 outputs=1
        _node(
            'Add',
            ['x', 'x'],
            ['y'],
            name='',
            domain='',
        ),
        # 7: Add inputs=2 outputs=1
        _node(
            'Add',
            ['y', 'd4'],
            ['z'],
            name='',
            domain='',
        ),
        # 8: Add inputs=2 outputs=1
        _node(
            'Add',
            ['z', 'z'],
            ['w'],
            name='',
            domain='',
        ),
        # 9: Add inputs=2 outputs=1
        _node(
            'Add',
            ['c', 'w'],
            ['d'],
            name='',
            domain='',
        ),
        # 10: Shrink inputs=1 outputs=1
        _node(
            'Shrink',
            ['o'],
            ['nego'],
            name='sub_nego',
            domain='',
            bias=1.0,
            lambd=-0.10000000149011612,
        ),
        # 11: Concat inputs=3 outputs=1
        _node(
            'Concat',
            ['o', 'd', 'nego'],
            ['phase_ri'],
            name='phase_ri_concat',
            domain='',
            axis=0,
        ),
        # 12: Cast inputs=1 outputs=1
        _node(
            'Cast',
            ['phase_ri'],
            ['phase_r'],
            name='cast_phase_r',
            domain='',
            to=1,
        ),
        # 13: Einsum inputs=46 outputs=1
        _node(
            'Einsum',
            ['F', 'F', 'F', 'C', 'phase_r', 'P', 'V', 'F', 'V', 'phase_r', 'P', 'F', 'V', 'V', 'F', 'F', 'F',
             'U', 'U', 'input', 'V', 'V', 'F', 'F', 'F', 'C', 'phase_r', 'F', 'V', 'phase_r', 'F', 'V', 'V',
             'V', 'F', 'F', 'F', 'U', 'U', 'input', 'V', 'V', 'input', 'U', 'U', 'U'],
            ['output'],
            name='',
            domain='',
            equation='fjTo,rsTs,jrTr,f,H,TH,Iz,lITX,Xy,J,tJ,litx,Xv,xv,lRSY,FRSY,ORSY,Fe,le,behv,Ah,Am,BCPD,EGPG,CEPE,B,P,gAPQ,Qn,p,gapq,ah,qu,Qu,gUVW,KUVW,ZUVW,Kd,gd,bduw,iw,Iw,bchw,Nk,Nc,Nc->bkhw',
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
            _tensor('F', TensorProto.FLOAT, (2, 2, 3, 2), INIT_F_0),
            _tensor('V', TensorProto.FLOAT, (2, 30), INIT_V_1),
            _tensor('U', TensorProto.FLOAT, (2, 10), INIT_U_2),
            _tensor('p', TensorProto.FLOAT, (1,), INIT_P_3),
            _tensor('C', TensorProto.FLOAT, (2,), INIT_C_4),
            _tensor('P', TensorProto.FLOAT, (3, 3), INIT_P_5),
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
