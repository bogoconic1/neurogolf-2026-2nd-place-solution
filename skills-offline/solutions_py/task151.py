from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task151'
TASK_NUM = 151
KAGGLE = {'score': 20.569183, 'date': '2026-07-09'}
MEMORY_BYTES = 0
PARAMS = 84
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 8
PRODUCER_NAME = ''
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task151_direct84'
OPSETS = [('', 13)]


# B: FLOAT[2, 30], 60 value(s)
INIT_B_0 = [0.25, 0.24969999492168427, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 1.4607045650482178,
 0.11367762833833694, -1.0170783996582031, 0.7777647972106934, 1.7013499736785889, -1.3947631120681763,
 -1.4116723537445068, -1.7383151054382324, 0.9530304670333862, 1.0577635765075684, 1.064226508140564,
 1.6024823188781738, -1.3581066131591797, -1.3426365852355957, -1.2322909832000732, 1.4795897006988525,
 1.2860002517700195, -1.516330361366272, -1.0, -0.8181818127632141, -0.6363636255264282, -0.4545454680919647,
 -0.27272728085517883, -0.09090909361839294, 0.09090909361839294, 0.27272728085517883, 0.4545454680919647,
 0.6363636255264282, 0.8181818127632141, 1.0, -0.8128068447113037, 1.2921960353851318, -1.4999432563781738,
 -1.036711573600769, -0.030801141634583473, 0.7548307776451111, 0.642178475856781, -0.36604684591293335,
 -0.09491748362779617, 0.18977706134319305, 0.233682781457901, 0.8640120029449463, -0.3800659477710724,
 -0.36912864446640015, -0.6695870161056519, 0.8162388801574707, 1.5020606517791748, -1.271790623664856]

# E: FLOAT[2, 10], 20 value(s)
INIT_E_1 = [-10.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

# M: FLOAT[2, 2], 4 value(s)
INIT_M_2 = [-1.0, 1.0, -1.0, 0.0]


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
        # 0: Einsum inputs=252 outputs=1
        _node(
            'Einsum',
            ['B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B',
             'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B',
             'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B',
             'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B',
             'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'input', 'E', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B',
             'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B',
             'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B',
             'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B',
             'B', 'B', 'B', 'B', 'B', 'E', 'input', 'B', 'B', 'B', 'B', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M',
             'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M',
             'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M',
             'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'M', 'M', 'M', 'M', 'M',
             'E', 'input', 'E', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'E', 'E', 'E', 'E'],
            ['output'],
            name='',
            domain='',
            equation='tW,tW,tW,tW,qW,qW,qW,sW,sW,qQ,qQ,qQ,vQ,vQ,vQ,vQ,uQ,uQ,kx,kx,kx,kx,kx,kx,kx,kx,kx,kx,kx,kx,kx,kx,kx,kx,kx,kx,kx,kx,kx,kx,kx,kx,kx,kx,kx,kx,kx,kx,kx,kx,kx,kx,kx,kx,kx,kx,kx,kx,kx,kx,kx,kx,kx,kx,kx,kx,kx,kx,kx,kx,kx,kx,kx,kx,kx,kx,kx,kx,kx,kx,kx,kx,kx,kx,kx,kx,kx,kx,nlrx,kl,Or,Fr,Cr,Rr,kz,kz,kz,kz,kz,kz,kz,kz,kz,kz,kz,kz,kz,kz,kz,kz,kz,kz,kz,kz,kz,kz,kz,kz,kz,kz,kz,kz,kz,kz,kz,kz,kz,kz,kz,kz,kz,kz,kz,kz,kz,kz,kz,kz,kz,kz,kz,kz,kz,kz,kz,kz,kz,kz,kz,kz,kz,kz,kz,kz,kz,kz,kz,kz,kz,kz,kz,kz,kz,kz,km,nmzp,Up,Xp,Ip,Lp,uC,uF,uf,Ff,DF,Df,uD,Ca,ua,AC,Aa,uA,vL,vI,LB,vB,JB,JL,vJ,GI,Ij,vj,vG,Gj,tX,tU,SU,UK,SK,tK,tS,XN,tN,VN,tV,VX,sO,sR,RH,PR,PH,sH,sP,OE,sE,ME,MO,sM,Jw,Vw,Sw,Gw,Mh,Ah,Dh,Ph,qY,yY,yY,yY,yY,eY,eY,eZ,gZ,iZ,ei,ig,id,ndhw,cd,eg,ec,eb,cb,eT,cT,bT,go,bo,yo,yo->nohw',
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
            _tensor('B', TensorProto.FLOAT, (2, 30), INIT_B_0),
            _tensor('E', TensorProto.FLOAT, (2, 10), INIT_E_1),
            _tensor('M', TensorProto.FLOAT, (2, 2), INIT_M_2),
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
