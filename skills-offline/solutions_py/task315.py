from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task315'
TASK_NUM = 315
KAGGLE = {'score': 20.355609, 'date': '2026-07-09'}
MEMORY_BYTES = 0
PARAMS = 104
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = 'task315_m0_p104_RcarriesA'
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task315_p104_R_carries_A'
OPSETS = [('', 13)]


# C: FLOAT[30, 2], 60 value(s)
INIT_C_0 = [0.8543173670768738, -1.8590847253799438, 1.423943042755127, -2.5805394649505615, 1.0459507703781128,
 -1.8162527084350586, 2.58740496635437, -0.025411104783415794, 3.6605172157287598, -0.04107566550374031,
 2.5893468856811523, -0.03267073631286621, 2.6946160793304443, 2.7961180210113525, 3.8108320236206055,
 3.956369161605835, 2.694714307785034, 2.7990305423736572, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# E: FLOAT[10, 2], 20 value(s)
INIT_E_1 = [-9.794901847839355, 21.884004592895508, -11.93203353881836, 22.13102149963379, -2.126046895980835, 9.429862976074219,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# B: FLOAT[2, 2, 2, 2], 16 value(s)
INIT_B_2 = [0.005636807531118393, 0.0059846555814146996, -0.01924227550625801, 0.012415614910423756, 0.005636757239699364,
 0.00597791001200676, -0.0016587991267442703, -0.0004335376142989844, 0.005636807531118393, 0.0059846555814146996,
 -0.01924227550625801, 0.012415614910423756, -0.039072588086128235, -3.978710889816284, 2.7957541942596436,
 0.900391161441803]

# R: FLOAT[2, 2, 2], 8 value(s)
INIT_R_3 = [1.0262794494628906, -0.507101833820343, 2.9737205505371094, -3.4928982257843018, -2.1542296409606934,
 3.079746723175049, 0.15422958135604858, -0.07974664866924286]


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
        # 0: Einsum inputs=51 outputs=1
        _node(
            'Einsum',
            ['input', 'E', 'C', 'C', 'C', 'C', 'B', 'C', 'C', 'C', 'C', 'B', 'C', 'C', 'C', 'C', 'B', 'C', 'C',
             'C', 'C', 'B', 'R', 'R', 'R', 'R', 'R', 'R', 'C', 'C', 'C', 'C', 'B', 'C', 'C', 'C', 'C', 'B', 'C',
             'C', 'C', 'C', 'B', 'C', 'C', 'C', 'C', 'B', 'R', 'E', 'E'],
            ['output'],
            name='task315_p104_R_carries_A',
            domain='',
            equation='barc,at,rh,rh,rh,ri,xghi,rk,rk,rk,rl,xekl,co,co,co,cw,xEow,cz,cz,cz,cA,xGzA,gUf,dVe,xWy,GZF,DTE,pMt,qB,qB,qB,qC,pfBC,qH,qH,qH,qI,pdHI,sJ,sJ,sJ,sK,pDJK,sL,sL,sL,sQ,pFLQ,OPt,jP,jO->bjqs',
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
            _tensor('C', TensorProto.FLOAT, (30, 2), INIT_C_0),
            _tensor('E', TensorProto.FLOAT, (10, 2), INIT_E_1),
            _tensor('B', TensorProto.FLOAT, (2, 2, 2, 2), INIT_B_2),
            _tensor('R', TensorProto.FLOAT, (2, 2, 2), INIT_R_3),
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
