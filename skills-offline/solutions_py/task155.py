from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task155'
TASK_NUM = 155
KAGGLE = {'score': 20.780492, 'date': '2026-07-09'}
MEMORY_BYTES = 8
PARAMS = 60
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 8
PRODUCER_NAME = ''
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'K_first_reverseblocks'
OPSETS = [('', 13)]


# E: FLOAT[30, 2], 60 value(s)
INIT_E_0 = [-0.7660444378852844, -0.6427876353263855, -0.1736481785774231, -0.9848077297210693, 0.5, -0.8660253882408142,
 0.9396926164627075, -0.3420201539993286, 0.9396926164627075, 0.3420201539993286, 0.5, 0.8660253882408142,
 -0.1736481785774231, 0.9848077297210693, -0.7660444378852844, 0.6427876353263855, -0.7271867990493774,
 1.0955666303634644, 0.923577606678009, -1.3494689464569092, -1.0321025848388672, 1.5627565383911133,
 -0.2757995128631592, -0.7836671471595764, -0.9293404817581177, 0.7698930501937866, 0.9230587482452393,
 -1.3453625440597534, 0.7147749662399292, 0.040161870419979095, -0.049001459032297134, -0.8312610387802124,
 0.9941592216491699, -0.9002727270126343, -0.954062819480896, -0.44699957966804504, 0.9325432181358337,
 -1.379802942276001, 0.10679630935192108, -0.8911112546920776, 0.6285431385040283, -1.45496666431427,
 0.6019428372383118, 0.9113245606422424, 0.47983330488204956, 0.12090100347995758, -0.9873945116996765,
 1.082527756690979, 0.9506722688674927, 0.7528555393218994, -0.551175594329834, 1.2404073476791382, -0.8634371161460876,
 -0.021634558215737343, -0.507269561290741, 1.2607207298278809, -0.5586060881614685, 1.2365546226501465,
 -0.8658601641654968, -0.8720240592956543]


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
        # 0: Einsum inputs=7 outputs=1
        _node(
            'Einsum',
            ['input', 'E', 'input', 'E', 'E', 'E', 'E'],
            ['ncode'],
            name='',
            domain='',
            equation='ncuv,up,ndxy,xq,zp,zq,zs->ns',
        ),
        # 1: Einsum inputs=65 outputs=1
        _node(
            'Einsum',
            ['E', 'E', 'ncode', 'E', 'E', 'E', 'ncode', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E',
             'E', 'ncode', 'E', 'E', 'E', 'E', 'E', 'ncode', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'ncode', 'E',
             'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'ncode', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E',
             'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'ncode', 'input', 'input'],
            ['output'],
            name='K_first_reverseblocks',
            domain='',
            equation='DE,Dz,nz,DC,DA,DB,nL,PL,PQ,PN,PM,PO,XC,XO,Xv,XI,Xo,Xe,XU,na,fe,fa,fg,fd,fb,nF,JF,JI,JK,JG,JH,VU,VW,nR,VR,VS,VT,hA,hb,hG,hM,hS,xv,xy,ns,xs,xt,xu,ht,rd,rB,rN,ru,rT,rH,rm,pm,po,hj,pj,pq,pi,ni,nkrl,nchw->ncrw',
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
            _tensor('E', TensorProto.FLOAT, (30, 2), INIT_E_0),
        ],
        value_info=[
            _vi('ncode', TensorProto.FLOAT, [1, 2]),
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
