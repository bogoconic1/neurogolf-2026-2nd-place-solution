from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task117'
TASK_NUM = 117
KAGGLE = {'score': 19.5839, 'date': '2026-07-09'}
MEMORY_BYTES = 24
PARAMS = 201
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 9
PRODUCER_NAME = 'ng-task117-updated-v10'
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task117_updated_v10'
OPSETS = [('', 18)]


# P: FLOAT[3, 30], 90 value(s)
INIT_P_0 = [0.6987918615341187, 0.6987918615341187, 0.6987918615341187, 0.6987918615341187, 0.6987918615341187, 0.6987918615341187,
 0.6987918615341187, 0.6987918615341187, 0.6987918615341187, 0.6987918615341187, 0.6987918615341187, 0.6987918615341187,
 0.6987918615341187, 0.6987918615341187, 0.6987918615341187, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 -17.48187828063965, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.6234897971153259, -0.22252093255519867, -0.9009688496589661,
 -0.9009688496589661, -0.22252093255519867, 0.6234897971153259, 1.0, 0.6234897971153259, -0.22252093255519867,
 -0.9009688496589661, -0.9009688496589661, -0.22252093255519867, 0.6234897971153259, 1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0,
 0.0, 0.0, 0.0, 0.0, -6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7818315029144287, 0.9749279022216797, 0.4338837265968323,
 -0.4338837265968323, -0.9749279022216797, -0.7818315029144287, -2.4492937051703357e-16, 0.7818315029144287,
 0.9749279022216797, 0.4338837265968323, -0.4338837265968323, -0.9749279022216797, -0.7818315029144287,
 -4.898587410340671e-16, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0, -5.0, 0.0, 0.0, 0.0]

# H: FLOAT[2, 30], 60 value(s)
INIT_H_1 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -17.669496536254883, 4.5625104904174805,
 -0.8536304831504822, 17.5, -13.999999046325684, 3.499999761581421, 18.5, -14.0, 3.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
 -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -3.022501230239868, 1.1719733476638794,
 -0.21457569301128387, 0.0, 0.0, -0.0, -1.0, -3.3306690738754696e-16, 7.401487051415874e-17, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0]

# K: FLOAT[3, 3, 3], 27 value(s)
INIT_K_2 = [0.40957584977149963, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.28620827198028564, 0.0, 0.0, 0.0,
 0.28620827198028564, 0.0, 0.0, 0.0, 0.0, 0.0, -0.28620827198028564, 0.0, 0.28620827198028564, 0.0]

# E: FLOAT[2, 10], 20 value(s)
INIT_E_3 = [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -0.18261906504631042, 2.285356044769287, 0.5723408460617065,
 -3.854494333267212, -0.593226432800293, -2.6350741386413574, -1.5854780673980713, -2.041794538497925,
 -1.0880839824676514, 1.3260893821716309]

# A: FLOAT[2, 2], 4 value(s)
INIT_A_4 = [1.9174021482467651, -3.133728265762329, 3.620553731918335, 0.14394594728946686]


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
        # 0: Einsum inputs=234 outputs=1
        _node(
            'Einsum',
            ['P', 'input', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K',
             'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'A', 'A', 'A',
             'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A',
             'A', 'A', 'A', 'A', 'E', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K',
             'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'A',
             'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A',
             'A', 'A', 'A', 'A', 'A', 'A', 'E', 'E', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'K', 'K', 'K', 'K', 'K',
             'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K',
             'K', 'K', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'E', 'E', 'A', 'A', 'A',
             'A', 'A', 'A', 'A', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K',
             'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A',
             'A', 'A', 'A', 'A', 'E', 'P', 'P', 'input', 'P', 'input', 'input', 'P', 'P', 'P', 'input'],
            ['Cr'],
            name='Cr',
            domain='',
            equation='qM,naMN,de,de,gd,gc,cc,ee,gd,fhi,fhi,fhi,fhi,fhi,fhi,fhi,fhi,fhi,fhi,fhi,fhi,fhi,fhi,fhi,fhi,fhi,fhi,fhi,fhi,fhi,fhi,fhi,fhi,fhi,fhi,fhi,ll,ll,ll,ll,ll,ll,ll,ll,ll,ll,ll,ll,ll,ll,ll,ll,ll,ll,ll,ll,ll,ll,ll,ll,ll,ll,ll,la,ko,ko,gk,gj,jj,oo,gk,rst,rst,rst,rst,rst,rst,rst,rst,rst,rst,rst,rst,rst,rst,rst,rst,rst,rst,rst,rst,rst,rst,rst,rst,rst,rst,rst,mm,mm,mm,mm,mm,mm,mm,mm,mm,mm,mm,mm,mm,mm,mm,mm,mm,mm,mm,mm,mm,mm,mm,mm,mm,mm,mm,mb,ua,AB,AB,uA,uw,ww,BB,uA,CDE,CDE,CDE,CDE,CDE,CDE,CDE,CDE,CDE,CDE,CDE,CDE,CDE,CDE,CDE,CDE,CDE,CDE,CDE,CDE,CDE,CDE,CDE,CDE,CDE,CDE,CDE,uv,uv,uv,uv,uv,uv,uv,vu,vu,vu,vu,vu,vu,vb,xa,GH,GH,xG,xF,FF,HH,xG,IJK,IJK,IJK,IJK,IJK,IJK,IJK,IJK,IJK,IJK,IJK,IJK,IJK,IJK,IJK,IJK,IJK,IJK,IJK,IJK,IJK,IJK,IJK,IJK,IJK,IJK,IJK,xy,xy,xy,xy,xy,xy,xy,yx,yx,yx,yx,yx,yx,yb,pP,pR,nbRS,pT,nbTU,nbVW,zV,zZ,zX,nbXY->nq',
        ),
        # 1: Einsum inputs=234 outputs=1
        _node(
            'Einsum',
            ['P', 'input', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K',
             'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'A', 'A', 'A',
             'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A',
             'A', 'A', 'A', 'A', 'E', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K',
             'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'A',
             'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A',
             'A', 'A', 'A', 'A', 'A', 'A', 'E', 'E', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'K', 'K', 'K', 'K', 'K',
             'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K',
             'K', 'K', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'E', 'E', 'A', 'A', 'A',
             'A', 'A', 'A', 'A', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K',
             'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A',
             'A', 'A', 'A', 'A', 'E', 'P', 'P', 'input', 'P', 'input', 'input', 'P', 'P', 'P', 'input'],
            ['Cc'],
            name='Cc',
            domain='',
            equation='qN,naMN,de,de,gd,gc,cc,ee,gd,fhi,fhi,fhi,fhi,fhi,fhi,fhi,fhi,fhi,fhi,fhi,fhi,fhi,fhi,fhi,fhi,fhi,fhi,fhi,fhi,fhi,fhi,fhi,fhi,fhi,fhi,fhi,ll,ll,ll,ll,ll,ll,ll,ll,ll,ll,ll,ll,ll,ll,ll,ll,ll,ll,ll,ll,ll,ll,ll,ll,ll,ll,ll,la,ko,ko,gk,gj,jj,oo,gk,rst,rst,rst,rst,rst,rst,rst,rst,rst,rst,rst,rst,rst,rst,rst,rst,rst,rst,rst,rst,rst,rst,rst,rst,rst,rst,rst,mm,mm,mm,mm,mm,mm,mm,mm,mm,mm,mm,mm,mm,mm,mm,mm,mm,mm,mm,mm,mm,mm,mm,mm,mm,mm,mm,mb,ua,AB,AB,uA,uw,ww,BB,uA,CDE,CDE,CDE,CDE,CDE,CDE,CDE,CDE,CDE,CDE,CDE,CDE,CDE,CDE,CDE,CDE,CDE,CDE,CDE,CDE,CDE,CDE,CDE,CDE,CDE,CDE,CDE,uv,uv,uv,uv,uv,uv,uv,vu,vu,vu,vu,vu,vu,vb,xa,GH,GH,xG,xF,FF,HH,xG,IJK,IJK,IJK,IJK,IJK,IJK,IJK,IJK,IJK,IJK,IJK,IJK,IJK,IJK,IJK,IJK,IJK,IJK,IJK,IJK,IJK,IJK,IJK,IJK,IJK,IJK,IJK,xy,xy,xy,xy,xy,xy,xy,yx,yx,yx,yx,yx,yx,yb,pP,pR,nbRS,pT,nbTU,nbVW,zV,zZ,zX,nbXY->nq',
        ),
        # 2: Einsum inputs=66 outputs=1
        _node(
            'Einsum',
            ['P', 'P', 'H', 'H', 'P', 'P', 'P', 'H', 'P', 'Cc', 'K', 'Cc', 'K', 'Cc', 'Cc', 'K', 'K', 'P', 'P',
             'Cc', 'K', 'P', 'Cc', 'K', 'P', 'P', 'P', 'H', 'H', 'input', 'E', 'E', 'A', 'H', 'P', 'P', 'P',
             'Cr', 'K', 'K', 'Cr', 'Cr', 'K', 'P', 'P', 'P', 'H', 'H', 'P', 'H', 'P', 'P', 'K', 'Cr', 'Cr', 'K',
             'Cr', 'K', 'P', 'P', 'P', 'H', 'input', 'A', 'E', 'E'],
            ['output'],
            name='output',
            domain='',
            equation='FT,FT,JT,JS,LS,LS,LS,JR,AR,...P,LOP,...N,LMN,...K,...H,FIK,FGH,Gs,Ms,...C,ABC,Bs,...E,ADE,Ow,Iw,Dw,js,jw,...crs,Yc,zc,XY,er,nr,ur,fr,...o,mno,tuv,...v,...g,dfg,tW,tW,tW,bW,bV,dV,bU,mU,mU,mpq,...q,...y,txy,...l,dil,xh,ih,ph,eh,...ahw,zQ,Xk,Qk->...khw',
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
            _tensor('P', TensorProto.FLOAT, (3, 30), INIT_P_0),
            _tensor('H', TensorProto.FLOAT, (2, 30), INIT_H_1),
            _tensor('K', TensorProto.FLOAT, (3, 3, 3), INIT_K_2),
            _tensor('E', TensorProto.FLOAT, (2, 10), INIT_E_3),
            _tensor('A', TensorProto.FLOAT, (2, 2), INIT_A_4),
        ],
        value_info=[
            _vi('Cr', TensorProto.FLOAT, [1, 3]),
            _vi('Cc', TensorProto.FLOAT, [1, 3]),
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
