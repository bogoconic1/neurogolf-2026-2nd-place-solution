from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task335'
TASK_NUM = 335
KAGGLE = {'score': 20.569183, 'date': '2026-07-11'}
MEMORY_BYTES = 0
PARAMS = 84
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 8
PRODUCER_NAME = 'task335_p84_pow31'
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task335_p84_pow31'
OPSETS = [('', 20)]


# basis: FLOAT[2, 30], 60 value(s)
INIT_BASIS_0 = [0.10526315867900848, 0.11764705926179886, 0.13333334028720856, 0.1538461595773697, 0.1818181872367859,
 0.2222222238779068, 0.2857142984867096, 0.4000000059604645, 0.6666666865348816, 1.0, 1.0, 0.6666666865348816,
 0.4000000059604645, 0.2857142984867096, 0.2222222238779068, 0.1818181872367859, 0.1538461595773697,
 0.13333334028720856, 0.11764705926179886, 0.10526315867900848, -0.8958173394203186, -0.8958173394203186,
 -1.0925084352493286, -1.0925084352493286, -0.7781457901000977, -0.7782249450683594, 0.0, 0.0, 0.0, 0.0, -1.0, -1.0,
 -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
 -0.00010098598431795835, 0.00010098598431795835, -1.8155680894851685, 1.8155680894851685, 0.0, 0.0,
 -0.28933754563331604, 1.2993375062942505, 0.0, 0.0]

# E: FLOAT[2, 10], 20 value(s)
INIT_E_1 = [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0, 0.0]

# H: FLOAT[2, 2], 4 value(s)
INIT_H_2 = [1.0, 1.0, 1.0, -1.0]


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
        # 0: Einsum inputs=122 outputs=1
        _node(
            'Einsum',
            ['basis', 'basis', 'basis', 'basis', 'basis', 'basis', 'H', 'H', 'H', 'basis', 'basis', 'basis',
             'H', 'H', 'H', 'basis', 'basis', 'input', 'basis', 'E', 'E', 'H', 'E', 'E', 'basis', 'basis',
             'input', 'basis', 'H', 'H', 'H', 'H', 'H', 'H', 'basis', 'H', 'H', 'basis', 'basis', 'basis',
             'basis', 'H', 'H', 'H', 'H', 'basis', 'basis', 'basis', 'basis', 'H', 'H', 'H', 'basis', 'basis',
             'basis', 'H', 'H', 'H', 'H', 'basis', 'basis', 'basis', 'basis', 'H', 'H', 'H', 'H', 'basis', 'H',
             'H', 'H', 'H', 'basis', 'H', 'H', 'basis', 'basis', 'basis', 'H', 'H', 'H', 'H', 'basis', 'basis',
             'basis', 'basis', 'basis', 'basis', 'basis', 'basis', 'basis', 'basis', 'basis', 'basis', 'basis',
             'basis', 'basis', 'basis', 'basis', 'basis', 'basis', 'basis', 'basis', 'basis', 'basis', 'basis',
             'basis', 'basis', 'basis', 'basis', 'basis', 'basis', 'basis', 'input', 'E', 'E', 'H', 'H', 'H',
             'H', 'E', 'E'],
            ['output'],
            name='p84_pow31_rowsum_from_basis',
            domain='',
            equation='rc,Bc,rc,lg,lg,Lg,sl,sl,sl,sq,sq,Dq,tt,ot,oo,BR,FR,...MRb,Jb,oM,wM,xx,oO,xO,LX,HX,...OeX,De,BA,aB,Aa,ar,fD,fs,Ai,DC,Cf,Ci,Fd,ld,ld,FE,pF,Ep,pl,Ei,sS,rS,rS,Ts,lT,Tr,HV,uV,uV,HG,UH,GU,Uu,Gj,Jn,vn,vn,JI,WJ,IW,Wv,Ij,LK,YL,KY,Yl,Kj,vl,vl,vh,uh,uh,vl,Zv,lZ,Zu,lQ,lQ,lQ,lQ,lQ,lQ,lQ,lQ,lQ,lQ,lQ,lQ,lQ,lQ,lQ,lQ,lQ,lQ,lQ,lQ,lQ,lQ,lQ,lQ,lQ,lQ,lQ,lQ,lQ,lQ,lQ,...mij,Nm,ym,yy,yz,zz,NP,zk,Pk->...kij',
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
            _tensor('basis', TensorProto.FLOAT, (2, 30), INIT_BASIS_0),
            _tensor('E', TensorProto.FLOAT, (2, 10), INIT_E_1),
            _tensor('H', TensorProto.FLOAT, (2, 2), INIT_H_2),
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
