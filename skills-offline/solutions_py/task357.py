from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task357'
TASK_NUM = 357
KAGGLE = {'score': 20.545653, 'date': '2026-07-13'}
MEMORY_BYTES = 8
PARAMS = 78
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 8
PRODUCER_NAME = ''
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task357_shared_basis'
OPSETS = [('', 18)]


# B: FLOAT[2, 30], 60 value(s)
INIT_B_0 = [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, -0.18337175250053406, 0.6308336853981018,
 -1.6003177734091878e-05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0, 1.125, -1.2144684791564941, -1.2629079818725586, -8.240007400512695,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# S: FLOAT[10], 10 value(s)
INIT_S_1 = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0]

# T: FLOAT[2, 2, 2], 8 value(s)
INIT_T_2 = [1.4142135381698608, 0.0, 0.5059999823570251, 24.475786209106445, 2.941682815551758, 0.9805608987808228,
 -0.4714045226573944, 0.0]


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
        # 0: Einsum inputs=2 outputs=1
        _node(
            'Einsum',
            ['input', 'B'],
            ['F'],
            name='',
            domain='',
            equation='nchw,kw->nk',
        ),
        # 1: Einsum inputs=191 outputs=1
        _node(
            'Einsum',
            ['T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'F', 'T', 'T', 'T', 'T', 'F', 'T', 'T', 'T', 'T', 'F', 'T',
             'T', 'T', 'T', 'F', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'T', 'T', 'T', 'T', 'B', 'B', 'B',
             'B', 'F', 'T', 'T', 'T', 'T', 'B', 'T', 'T', 'T', 'T', 'B', 'T', 'T', 'T', 'T', 'input', 'F', 'T',
             'T', 'T', 'T', 'B', 'T', 'T', 'T', 'T', 'B', 'T', 'T', 'T', 'T', 'F', 'T', 'T', 'T', 'T', 'B', 'T',
             'T', 'T', 'T', 'B', 'T', 'T', 'T', 'T', 'F', 'T', 'T', 'T', 'T', 'B', 'T', 'T', 'T', 'T', 'B', 'T',
             'T', 'T', 'T', 'F', 'B', 'B', 'T', 'T', 'T', 'T', 'B', 'T', 'T', 'T', 'T', 'F', 'B', 'B', 'T', 'T',
             'T', 'T', 'B', 'T', 'T', 'T', 'T', 'B', 'T', 'T', 'T', 'T', 'B', 'T', 'T', 'T', 'T', 'F', 'T', 'T',
             'T', 'T', 'T', 'T', 'T', 'T', 'B', 'T', 'T', 'T', 'T', 'B', 'T', 'T', 'T', 'T', 'F', 'T', 'T', 'T',
             'T', 'T', 'T', 'T', 'T', 'B', 'T', 'T', 'T', 'T', 'B', 'F', 'T', 'T', 'T', 'T', 'B', 'T', 'T', 'T',
             'T', 'B', 'F', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'S'],
            ['output'],
            name='',
            domain='',
            equation='tZt,tZZ,ttZ,tZZ,wZw,wZZ,wwZ,wZZ,nq,zqt,zqs,zst,sqq,nu,zuw,zuv,zvw,vuu,nQ,zQR,RQQ,zQS,zRS,nT,QY,qY,uY,TY,ZY,Sc,tc,wc,Vc,zTV,zUV,zTU,UTT,Rr,Ur,sr,vr,nb,bbb,bbb,bbb,bbb,dr,zbd,dbb,zbe,zde,ec,eZe,eZZ,eeZ,eZZ,narc,nf,fff,fff,fff,fff,gr,zfg,gff,zfh,zgh,hc,hZh,hZZ,hhZ,hZZ,ni,iZi,iZZ,iiZ,iZZ,jr,zij,jii,zik,zjk,kc,kZk,kZZ,kkZ,kZZ,nl,lZl,lZZ,llZ,lZZ,mr,zlm,mll,zlp,zmp,pc,pZp,pZZ,ppZ,pZZ,nx,xW,yr,zxy,yxx,zxA,zyA,Ac,AZA,AZZ,AAZ,AZZ,nB,BX,Cr,zBC,CBB,zBD,zCD,Dc,DZD,DZZ,DDZ,DZZ,Gc,GZG,GZZ,GGZ,GZZ,Fr,zFG,zEG,zEF,FEE,nE,EZE,EZZ,EEZ,EZZ,EEZ,EEZ,EZZ,ZEE,Jc,JZJ,JZZ,JJZ,JZZ,Ir,zIJ,zHJ,zHI,IHH,nH,HZH,HZZ,HHZ,HZZ,HHZ,HHZ,HZZ,ZHH,Mc,zKM,zLM,zKL,LKK,Lr,nK,KZK,KZZ,KKZ,KZZ,Pc,zNP,zOP,zNO,ONN,Or,nN,NZN,NZZ,NNZ,NZZ,zZz,zZZ,zzZ,zZZ,o->norc',
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
            _tensor('S', TensorProto.FLOAT, (10,), INIT_S_1),
            _tensor('T', TensorProto.FLOAT, (2, 2, 2), INIT_T_2),
        ],
        value_info=[
            _vi('F', TensorProto.FLOAT, [1, 2]),
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
