from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task114'
TASK_NUM = 114
KAGGLE = {'score': 20.355609, 'date': '2026-07-15'}
MEMORY_BYTES = 0
PARAMS = 104
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = ''
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'g'
OPSETS = [('', 20)]


# P: FLOAT[30, 2], 60 value(s)
INIT_P_0 = [2.0, 0.5, 0.0, -0.5, -2.0, -1.5, -4.0, -2.5, -6.0, -3.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# S1: FLOAT[2, 2], 4 value(s)
INIT_S1_1 = [-1.0, 1.0, 1.0, 0.0]

# L2: FLOAT[2, 2, 2], 8 value(s)
INIT_L2_2 = [-63.0, 108.0, 28.0, -48.0, 72.0, -120.0, -120.0, 192.0]

# BY: FLOAT[2, 2, 2], 8 value(s)
INIT_BY_3 = [1.0, -2.0, 1.0, -2.0, 1.25, -2.0, 2.3333332538604736, -4.0]

# E: FLOAT[10, 2], 20 value(s)
INIT_E_4 = [1.9358691133675165e-05, 9.198346560879145e-06, -4.900000021734741e-06, 6.000769232965652e-22, -3.1713464068161556e-06,
 -2.30411819757137e-06, 1.5444669543285272e-06, 4.753380835609278e-06, 1.5141832818699186e-06, -4.66017672806629e-06,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.558810815069592e-06, -3.3121698379545705e-06, 0.0, 0.0]

# M: FLOAT[2, 2], 4 value(s)
INIT_M_5 = [0.9510565400123596, -0.30901700258255005, 0.30901700258255005, 0.9510565400123596]


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
        # 0: Einsum inputs=64 outputs=1
        _node(
            'Einsum',
            ['input', 'P', 'P', 'S1', 'L2', 'P', 'P', 'S1', 'P', 'P', 'L2', 'P', 'P', 'L2', 'P', 'P', 'BY',
             'BY', 'P', 'input', 'input', 'P', 'P', 'S1', 'L2', 'P', 'P', 'P', 'S1', 'P', 'P', 'L2', 'P', 'L2',
             'P', 'P', 'BY', 'BY', 'P', 'S1', 'E', 'E', 'E', 'M', 'E', 'E', 'M', 'E', 'E', 'M', 'M', 'M', 'E',
             'E', 'M', 'M', 'M', 'E', 'S1', 'S1', 'M', 'M', 'L2', 'L2'],
            ['output'],
            name='e',
            domain='',
            equation='...eij,iq,xh,hn,nnH,RH,xI,IJ,RJ,xK,qKL,RL,xM,NMM,RN,RY,lqY,ZlZ,RZ,...sxy,...fuv,vk,yO,OG,GGP,yV,yQ,yT,QS,WP,WS,kTU,WU,XVV,WX,Wg,mkg,omo,Wo,lm,sz,cw,sa,ab,cb,sd,pd,cp,sr,rt,tA,AB,cB,sC,DC,ED,FE,cF,nG,nG,nG,Gn,nGG,Gnn->...cRW',
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
            _tensor('P', TensorProto.FLOAT, (30, 2), INIT_P_0),
            _tensor('S1', TensorProto.FLOAT, (2, 2), INIT_S1_1),
            _tensor('L2', TensorProto.FLOAT, (2, 2, 2), INIT_L2_2),
            _tensor('BY', TensorProto.FLOAT, (2, 2, 2), INIT_BY_3),
            _tensor('E', TensorProto.FLOAT, (10, 2), INIT_E_4),
            _tensor('M', TensorProto.FLOAT, (2, 2), INIT_M_5),
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
