from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task278'
TASK_NUM = 278
KAGGLE = {'score': 20.187816, 'date': '2026-07-15'}
MEMORY_BYTES = 0
PARAMS = 123
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = 'sol8-r6-golfer-1'
PRODUCER_VERSION = 'task278-p144-fast-op13'
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task278_col_base10_cost123'
OPSETS = [('', 13)]


# Psel: FLOAT[2, 10], 20 value(s)
INIT_PSEL_0 = [1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# BAS: FLOAT[3, 30], 90 value(s)
INIT_BAS_1 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.945817232131958, 0.789140522480011, 0.5469481348991394, 0.24548548460006714,
 -0.0825793445110321, -0.4016954302787781, -0.6772815585136414, -0.8794737458229065, -0.9863613247871399,
 -0.9863613247871399, -0.8794737458229065, -0.6772815585136414, -0.4016954302787781, -0.0825793445110321,
 0.24548548460006714, 0.5469481348991394, 0.789140522480011, 0.945817232131958, 1.0, 0.945817232131958,
 0.789140522480011, 0.5469481348991394, 0.24548548460006714, -0.0825793445110321, -0.4016954302787781,
 -0.6772815585136414, -0.8794737458229065, -0.9863613247871399, -0.9863613247871399, 0.0, 0.3246994614601135,
 0.614212691783905, 0.8371664881706238, 0.9694002866744995, 0.9965844750404358, 0.915773332118988, 0.7357239127159119,
 0.47594738006591797, 0.1645945906639099, -0.1645945906639099, -0.47594738006591797, -0.7357239127159119,
 -0.915773332118988, -0.9965844750404358, -0.9694002866744995, -0.8371664881706238, -0.614212691783905,
 -0.3246994614601135, -2.4492937051703357e-16, 0.3246994614601135, 0.614212691783905, 0.8371664881706238,
 0.9694002866744995, 0.9965844750404358, 0.915773332118988, 0.7357239127159119, 0.47594738006591797, 0.1645945906639099,
 -0.1645945906639099]

# T: FLOAT[2, 2], 4 value(s)
INIT_T_2 = [1.0, -64.0, 2.0, -1.0]

# F: FLOAT[2, 3], 6 value(s)
INIT_F_3 = [0.027091380208730698, 0.0, 0.0, -0.972908616065979, 1.0, 1.0]

# F10Col: FLOAT[3], 3 value(s)
INIT_F10COL_4 = [0.7598370909690857, 1.0, 1.0]


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
        # 0: Einsum inputs=243 outputs=1
        _node(
            'Einsum',
            ['BAS', 'BAS', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'BAS', 'F10Col', 'F10Col', 'F', 'BAS',
             'BAS', 'BAS', 'F10Col', 'F10Col', 'F10Col', 'F10Col', 'F10Col', 'F', 'BAS', 'BAS', 'F10Col',
             'F10Col', 'F10Col', 'F10Col', 'F10Col', 'F10Col', 'F10Col', 'F10Col', 'F10Col', 'BAS', 'BAS',
             'F10Col', 'F10Col', 'F10Col', 'F', 'F', 'BAS', 'BAS', 'F10Col', 'F', 'F', 'F', 'F', 'BAS', 'BAS',
             'F', 'F', 'F', 'F', 'BAS', 'BAS', 'BAS', 'F', 'input', 'BAS', 'Psel', 'Psel', 'input', 'BAS',
             'BAS', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'BAS', 'BAS', 'F10Col', 'F10Col', 'F', 'BAS',
             'BAS', 'F10Col', 'F10Col', 'F10Col', 'F10Col', 'F10Col', 'F', 'BAS', 'BAS', 'F10Col', 'F10Col',
             'F10Col', 'F10Col', 'F10Col', 'F10Col', 'F10Col', 'F10Col', 'F10Col', 'BAS', 'BAS', 'F10Col',
             'F10Col', 'F10Col', 'F', 'F', 'BAS', 'BAS', 'F10Col', 'F', 'F', 'F', 'F', 'BAS', 'BAS', 'F', 'F',
             'F', 'F', 'BAS', 'BAS', 'BAS', 'F', 'BAS', 'T', 'BAS', 'BAS', 'F', 'F', 'F', 'F', 'F', 'F', 'F',
             'F', 'F', 'BAS', 'BAS', 'F10Col', 'F10Col', 'F', 'BAS', 'BAS', 'F10Col', 'F10Col', 'F10Col',
             'F10Col', 'F10Col', 'F', 'BAS', 'BAS', 'F10Col', 'F10Col', 'F10Col', 'F10Col', 'F10Col', 'F10Col',
             'F10Col', 'F10Col', 'F10Col', 'BAS', 'BAS', 'F10Col', 'F10Col', 'F10Col', 'F', 'F', 'BAS', 'BAS',
             'F10Col', 'F', 'F', 'F', 'F', 'BAS', 'BAS', 'F', 'F', 'F', 'F', 'BAS', 'BAS', 'BAS', 'BAS', 'F',
             'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'BAS', 'BAS', 'F10Col', 'F10Col', 'F', 'BAS', 'BAS',
             'F10Col', 'F10Col', 'F10Col', 'F10Col', 'F10Col', 'F', 'BAS', 'BAS', 'F10Col', 'F10Col', 'F10Col',
             'F10Col', 'F10Col', 'F10Col', 'F10Col', 'F10Col', 'F10Col', 'BAS', 'BAS', 'F10Col', 'F10Col',
             'F10Col', 'F', 'F', 'BAS', 'BAS', 'F10Col', 'F', 'F', 'F', 'F', 'BAS', 'BAS', 'F', 'F', 'F', 'F',
             'BAS', 'BAS', 'T', 'Psel', 'input', 'Psel'],
            ['output'],
            name='',
            domain='',
            equation='aU,aP,Ta,Ta,Ta,Ta,Ta,Ta,Ta,Ta,Ta,bU,b,b,Tb,bP,cU,cP,c,c,c,c,c,Tc,dU,dP,d,d,d,d,d,d,d,d,d,eU,eP,e,e,e,Te,Te,fU,fP,f,Tf,Tf,Tf,Tf,gU,gP,Tg,Tg,Tg,Tg,hU,hP,RP,rR,nBUV,RU,tB,tB,nBPQ,iV,iQ,Wi,Wi,Wi,Wi,Wi,Wi,Wi,Wi,Wi,jV,jQ,j,j,Wj,kV,kQ,k,k,k,k,k,Wk,lV,lQ,l,l,l,l,l,l,l,l,l,mV,mQ,m,m,m,Wm,Wm,pV,pQ,p,Wp,Wp,Wp,Wp,qV,qQ,Wq,Wq,Wq,Wq,uV,uQ,SQ,rS,SV,rr,vP,vI,Xv,Xv,Xv,Xv,Xv,Xv,Xv,Xv,Xv,wP,wI,w,w,Xw,xP,xI,x,x,x,x,x,Xx,yP,yI,y,y,y,y,y,y,y,y,y,zP,zI,z,z,z,Xz,Xz,CP,CI,C,XC,XC,XC,XC,DP,DI,XD,XD,XD,XD,EP,EI,FQ,FJ,YF,YF,YF,YF,YF,YF,YF,YF,YF,GQ,GJ,G,G,YG,HQ,HJ,H,H,H,H,H,YH,KQ,KJ,K,K,K,K,K,K,K,K,K,LQ,LJ,L,L,L,YL,YL,MQ,MJ,M,YM,YM,YM,YM,NQ,NJ,YN,YN,YN,YN,OQ,OJ,st,sA,nAIJ,so->noIJ',
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
            _tensor('Psel', TensorProto.FLOAT, (2, 10), INIT_PSEL_0),
            _tensor('BAS', TensorProto.FLOAT, (3, 30), INIT_BAS_1),
            _tensor('T', TensorProto.FLOAT, (2, 2), INIT_T_2),
            _tensor('F', TensorProto.FLOAT, (2, 3), INIT_F_3),
            _tensor('F10Col', TensorProto.FLOAT, (3,), INIT_F10COL_4),
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
