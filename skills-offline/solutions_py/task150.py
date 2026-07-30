from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task150'
TASK_NUM = 150
KAGGLE = {'score': 20.780492, 'date': '2026-07-11'}
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
 -0.1736481785774231, 0.9848077297210693, -0.7660444378852844, 0.6427876353263855, -0.912285327911377,
 0.03151022642850876, 0.942584753036499, -1.349584937095642, -1.0423614978790283, 1.5615251064300537,
 -0.1461121290922165, -0.7171170115470886, -0.872713029384613, 0.9386563897132874, 0.9425470232963562,
 -1.3454118967056274, 0.7720032930374146, -0.1938549131155014, -0.13771256804466248, -0.6795130372047424,
 0.9609859585762024, -0.8208826184272766, -0.9332082867622375, -0.48462703824043274, 0.9464634656906128,
 -1.380350947380066, -0.17863571643829346, -0.8658325672149658, 0.6509496569633484, -1.4549107551574707,
 0.5818239450454712, 0.9067425727844238, 0.6282351613044739, 0.27316680550575256, -0.9882658123970032,
 1.0982879400253296, 0.9595871567726135, 0.7900806665420532, -0.5864686369895935, 1.250027060508728,
 -0.8190335631370544, 0.509006679058075, -0.591097891330719, 1.269604206085205, -0.5848987698554993, 1.2463502883911133,
 -0.8898242712020874, -0.8780783414840698]


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
            equation='DE,Dz,nz,DC,DA,DB,nL,PL,PQ,PN,PM,PO,XC,XO,Xv,XI,Xo,Xe,XU,na,fe,fa,fg,fd,fb,nF,JF,JI,JK,JG,JH,VU,VW,nR,VR,VS,VT,hA,hb,hG,hM,hS,xv,xy,ns,xs,xt,xu,ht,rd,rB,rN,ru,rT,rH,rm,pm,po,hj,pj,pq,pi,ni,nkrl,ncwh->ncwr',
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
