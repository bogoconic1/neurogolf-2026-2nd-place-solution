from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task035'
TASK_NUM = 35
KAGGLE = {'score': 20.080019, 'date': '2026-07-15'}
MEMORY_BYTES = 16
PARAMS = 121
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = 'gpt-5.6-sol-swarm-20plus'
PRODUCER_VERSION = 'task035-tie-v1'
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task035_m16_p131_cw_split_factor_drop_zl'
OPSETS = [('', 21)]


# E: FLOAT[10, 3], 30 value(s)
INIT_E_0 = [1.0, 0.22123174369335175, 1.396802306175232, 1.0, -0.6420395374298096, 1.2600735425949097, 1.0, -1.2600735425949097,
 0.6420395374298096, 1.0, -1.396802306175232, -0.22123174369335175, 1.0, -1.0, -1.0, 1.0, -0.22123174369335175,
 -1.396802306175232, 1.0, 0.6420395374298096, -1.2600735425949097, 1.0, 1.2600735425949097, -0.6420395374298096, 1.0,
 1.0, 1.0, 1.0, 1.396802306175232, 0.22123174369335175]

# D: FLOAT[2, 3], 6 value(s)
INIT_D_1 = [1.5, 0.0, 0.0, -16.19999885559082, 9.0, 9.0]

# Gate: FLOAT[2, 30], 60 value(s)
INIT_GATE_2 = [0.6666666865348816, 0.6666666865348816, 0.6666666865348816, 0.6666666865348816, 0.6666666865348816, 0.6666666865348816,
 0.6666666865348816, 0.6666666865348816, 0.6666666865348816, 0.6666666865348816, -0.3333333432674408,
 -0.3333333432674408, -0.3333333432674408, -0.3333333432674408, -0.3333333432674408, -0.3333333432674408,
 -0.3333333432674408, -0.3333333432674408, -0.3333333432674408, -0.3333333432674408, -0.3333333432674408,
 -0.3333333432674408, -0.3333333432674408, -0.3333333432674408, -0.3333333432674408, -0.3333333432674408,
 -0.3333333432674408, -0.3333333432674408, -0.3333333432674408, -0.3333333432674408, 0.3333333432674408,
 0.2222222238779068, 0.1111111119389534, 0.0, -0.1111111119389534, -0.2222222238779068, -0.3333333432674408,
 -0.4444444477558136, -0.5555555820465088, -0.6666666865348816, 0.08888889104127884, 0.08888889104127884,
 0.08888889104127884, 0.08888889104127884, 0.08888889104127884, 0.08888889104127884, 0.08888889104127884,
 0.08888889104127884, 0.08888889104127884, 0.08888889104127884, 0.08888889104127884, 0.08888889104127884,
 0.08888889104127884, 0.08888889104127884, 0.08888889104127884, 0.08888889104127884, 0.08888889104127884,
 0.08888889104127884, 0.08888889104127884, 0.08888889104127884]

# G: FLOAT[2, 2, 2], 8 value(s)
INIT_G_3 = [0.3333333432674408, 0.0, 3430.86376953125, 0.0, 0.20297469198703766, 1948.55712890625, 2192.126708984375,
 -82.2118148803711]

# Rbottom: FLOAT[2, 2, 2], 8 value(s)
INIT_RBOTTOM_4 = [1.5, 0.0, 0.0, -0.00015205933596007526, 0.0, 3.0, -1.5, 0.01216370239853859]

# Bs: FLOAT[3], 3 value(s)
INIT_BS_5 = [3.3711936473846436, 2.048243761062622, 4.019904136657715]

# A: FLOAT[3, 2], 6 value(s)
INIT_A_6 = [1.0, 0.00293422001414001, -0.5019952058792114, 0.005233299918472767, -0.49818024039268494, 0.0037612300366163254]


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
        # 0: Einsum inputs=48 outputs=1
        _node(
            'Einsum',
            ['E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E',
             'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'A', 'A', 'E', 'A', 'A', 'E', 'A', 'A', 'E', 'A', 'A', 'E',
             'A', 'A', 'E', 'A', 'A', 'Bs', 'input', 'Gate'],
            ['hfeat'],
            name='',
            domain='',
            equation='ab,ac,ad,ae,af,ag,ai,ak,al,am,ao,ap,aq,ar,as,at,au,av,ax,ay,az,aA,aB,aC,aD,aE,aF,aG,GH,IH,aJ,JK,LK,aM,MN,ON,aP,PQ,RQ,aS,ST,UT,aV,VW,XW,Y,nahw,jh->nj',
        ),
        # 1: Einsum inputs=50 outputs=1
        _node(
            'Einsum',
            ['E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E',
             'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'A', 'A', 'E', 'A', 'A', 'E', 'A', 'A', 'E', 'A', 'A', 'E',
             'A', 'A', 'E', 'A', 'A', 'Bs', 'input', 'Gate', 'Rbottom', 'G'],
            ['wfeat'],
            name='',
            domain='',
            equation='ab,ac,ad,ae,af,ag,ai,ak,am,ao,ap,aq,ar,as,at,au,av,ax,ay,az,aA,aB,aC,aD,aE,aF,aG,aH,HI,JI,aK,KL,ML,aN,NO,PO,aQ,QR,SR,aT,TU,VU,aW,WX,YX,Z,nahw,lw,ljj,jjl->nj',
        ),
        # 2: Einsum inputs=94 outputs=1
        _node(
            'Einsum',
            ['Rbottom', 'Rbottom', 'Rbottom', 'G', 'Rbottom', 'G', 'Bs', 'D', 'E', 'Gate', 'input', 'E', 'D',
             'Gate', 'Rbottom', 'G', 'wfeat', 'Rbottom', 'wfeat', 'Rbottom', 'Rbottom', 'G', 'Gate', 'Gate',
             'input', 'E', 'E', 'Bs', 'D', 'G', 'Rbottom', 'Rbottom', 'Rbottom', 'D', 'E', 'E', 'D', 'E', 'E',
             'D', 'E', 'D', 'Rbottom', 'G', 'G', 'G', 'Rbottom', 'Rbottom', 'G', 'Gate', 'G', 'Rbottom', 'G',
             'Rbottom', 'Gate', 'G', 'Rbottom', 'D', 'E', 'Rbottom', 'Rbottom', 'G', 'hfeat', 'Rbottom',
             'hfeat', 'Rbottom', 'D', 'E', 'E', 'E', 'A', 'input', 'Gate', 'Gate', 'Gate', 'Gate', 'D', 'E',
             'D', 'Bs', 'E', 'E', 'input', 'Gate', 'E', 'D', 'D', 'Bs', 'E', 'E', 'input', 'Gate', 'G',
             'Rbottom'],
            ['output'],
            name='',
            domain='',
            equation='nnn,nnn,nnn,nnn,sss,sss,u,nu,qu,sr,...qrw,qt,nt,Lw,LKK,KKL,...M,CKM,...P,CNP,ONN,NNO,Ow,gf,...efw,ej,ei,j,dj,ddd,ddd,ddd,ddd,di,oi,ot,nv,ov,ok,dk,oJ,DJ,DDD,cCD,cbd,DDD,DDD,DDD,cmn,Zw,QQZ,ZQQ,QQY,YQQ,Yw,cQR,RRR,RX,oX,RRR,RRR,RRR,...y,mxy,...B,mAB,zp,op,ap,al,lc,...ahw,Ah,xh,bh,bh,DH,oH,DI,I,EI,EH,...EhF,GF,oV,RV,RW,W,SV,SW,...ShT,UT,UUU,UUU->...ohw',
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
            _tensor('E', TensorProto.FLOAT, (10, 3), INIT_E_0),
            _tensor('D', TensorProto.FLOAT, (2, 3), INIT_D_1),
            _tensor('Gate', TensorProto.FLOAT, (2, 30), INIT_GATE_2),
            _tensor('G', TensorProto.FLOAT, (2, 2, 2), INIT_G_3),
            _tensor('Rbottom', TensorProto.FLOAT, (2, 2, 2), INIT_RBOTTOM_4),
            _tensor('Bs', TensorProto.FLOAT, (3,), INIT_BS_5),
            _tensor('A', TensorProto.FLOAT, (3, 2), INIT_A_6),
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
