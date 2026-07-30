from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task246'
TASK_NUM = 246
KAGGLE = {'score': 20.569183, 'date': '2026-07-13'}
MEMORY_BYTES = 0
PARAMS = 84
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 8
PRODUCER_NAME = 'golf'
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'g'
OPSETS = [('', 20)]


# basis: FLOAT[2, 30], 60 value(s)
INIT_BASIS_0 = [0.07894736528396606, 0.0882352963089943, 0.10000000894069672, 0.11538462340831757, 0.13636364042758942,
 0.1666666716337204, 0.2142857313156128, 0.30000001192092896, 0.5, 0.75, 0.75, 0.5, 0.30000001192092896,
 0.2142857313156128, 0.1666666716337204, 0.13636364042758942, 0.11538462340831757, 0.10000000894069672,
 0.0882352963089943, 0.07894736528396606, -0.6718630194664001, -0.6718630194664001, -0.8193813562393188,
 -0.8193813562393188, -0.5836093425750732, -0.5836687088012695, 0.0, 0.0, 0.0, 0.0, -0.75, -0.75, -0.75, -0.75, -0.75,
 -0.75, -0.75, -0.75, -0.75, -0.375, 0.375, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75,
 -7.573948823846877e-05, 7.573948823846877e-05, -1.3616760969161987, 1.3616760969161987, 0.0, 0.0, -0.21700316667556763,
 0.9745031595230103, 0.0, 0.0]

# E: FLOAT[2, 10], 20 value(s)
INIT_E_1 = [1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]

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
        # 0: Einsum inputs=101 outputs=1
        _node(
            'Einsum',
            ['basis', 'basis', 'basis', 'basis', 'basis', 'basis', 'basis', 'basis', 'basis', 'H', 'H', 'H',
             'E', 'E', 'basis', 'input', 'basis', 'basis', 'E', 'E', 'H', 'basis', 'input', 'basis', 'basis',
             'H', 'H', 'H', 'H', 'basis', 'H', 'H', 'H', 'H', 'basis', 'basis', 'basis', 'basis', 'H', 'H', 'H',
             'H', 'basis', 'basis', 'basis', 'basis', 'H', 'H', 'H', 'H', 'basis', 'basis', 'basis', 'H', 'H',
             'H', 'H', 'basis', 'basis', 'basis', 'H', 'H', 'H', 'H', 'H', 'basis', 'basis', 'H', 'H', 'H', 'H',
             'basis', 'H', 'H', 'H', 'H', 'H', 'basis', 'basis', 'basis', 'basis', 'basis', 'basis', 'basis',
             'basis', 'basis', 'basis', 'basis', 'basis', 'basis', 'basis', 'basis', 'E', 'H', 'H', 'H',
             'input', 'E', 'H', 'E', 'E'],
            ['output'],
            name='output',
            domain='',
            equation='rc,Bc,rc,lg,lg,Lg,sq,sq,Dq,oo,ot,tt,oM,wM,Jb,...MRb,BR,FR,oO,xO,xx,De,...OeX,HX,LX,Aa,BA,aB,ar,Ai,Cf,DC,fD,fs,Ci,Fd,ld,ld,Ep,FE,pF,pl,Ei,sS,rS,rS,lT,sl,Ts,Tr,HV,uV,uV,GU,HG,UH,Uu,Jn,vn,vn,vl,IW,JI,WJ,Wv,Gj,Ij,LK,YL,KY,Yl,Kj,Zu,lZ,Zv,lQ,QQ,vh,uh,uh,Qh,Qh,Qh,Qh,uh,uh,uh,uh,uh,uh,uh,uh,ym,yy,yz,zz,...mij,Nm,NP,Pk,zk->...kij',
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
