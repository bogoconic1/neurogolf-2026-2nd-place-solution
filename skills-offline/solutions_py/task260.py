from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task260'
TASK_NUM = 260
KAGGLE = {'score': 20.39483, 'date': '2026-07-15'}
MEMORY_BYTES = 8
PARAMS = 92
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = ''
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'g'
OPSETS = [('', 15)]


# B: FLOAT[2, 30], 60 value(s)
INIT_B_0 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# Klo: FLOAT[2, 2, 2], 8 value(s)
INIT_KLO_1 = [-1.0, 0.0, 0.0, 0.0, -0.5, -1.0, 1.0, 0.0]

# C: FLOAT[2, 10], 20 value(s)
INIT_C_2 = [-0.4714045226573944, 0.4714045226573944, 0.4714045226573944, 0.4714045226573944, 0.4714045226573944,
 -0.4714045226573944, 0.4714045226573944, 0.4714045226573944, 0.4714045226573944, 0.4714045226573944,
 0.4714045226573944, 0.4714045226573944, 0.4714045226573944, 0.4714045226573944, 0.4714045226573944, 0.9428090453147888,
 0.4714045226573944, 0.4714045226573944, 0.4714045226573944, 0.4714045226573944]

# Mgate: FLOAT[2, 2], 4 value(s)
INIT_MGATE_3 = [-0.31622758507728577, 0.6767815947532654, 0.9486833810806274, -2.819923162460327]


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
        # 0: Einsum inputs=21 outputs=1
        _node(
            'Einsum',
            ['B', 'input', 'B', 'Klo', 'C', 'C', 'Klo', 'Klo', 'C', 'C', 'B', 'input', 'B', 'Klo', 'Klo', 'Klo',
             'Klo', 'Klo', 'Klo', 'Klo', 'Klo'],
            ['F1s'],
            name='F1s',
            domain='',
            equation='nj,bkij,ei,uen,ak,ak,aaT,cSS,cl,dl,zy,blxy,gx,vgz,Puv,Rov,AoP,ABC,DBA,DAB,oYY->bo',
        ),
        # 1: Einsum inputs=85 outputs=1
        _node(
            'Einsum',
            ['C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C',
             'C', 'C', 'C', 'C', 'C', 'Klo', 'Klo', 'Klo', 'Klo', 'Klo', 'F1s', 'Klo', 'Klo', 'Klo', 'Klo',
             'Klo', 'F1s', 'Klo', 'Klo', 'Klo', 'B', 'Klo', 'B', 'B', 'B', 'B', 'B', 'Klo', 'Klo', 'Klo', 'Klo',
             'Klo', 'Klo', 'Klo', 'Klo', 'Klo', 'Klo', 'B', 'Klo', 'Klo', 'B', 'Klo', 'Klo', 'Klo', 'Klo',
             'Klo', 'Klo', 'Klo', 'Klo', 'Klo', 'Klo', 'input', 'C', 'Klo', 'Klo', 'input', 'C', 'F1s', 'Mgate',
             'F1s', 'Mgate', 'Klo', 'F1s', 'Mgate', 'Klo'],
            ['output'],
            name='output',
            domain='',
            equation='ZT,ZT,ZT,ZT,ZT,ZT,ZT,ZT,ZT,ZT,ZT,ZT,ZT,ZT,ZT,ZT,ZT,ZT,ZT,ZT,ZT,ZT,ZT,ZT,ZT,QRZ,QRA,QRA,QAs,sAR,...s,ijZ,ijA,ijA,qAj,iAq,...q,sfm,fAB,BAN,er,qen,mc,Nr,nc,zc,gr,hgz,KhL,JhM,KIM,tIJ,UAt,tAV,UVA,UVA,UVZ,Pr,DAP,oAD,wc,low,GlH,FlO,GEO,vEF,vAY,XAv,XYA,XYA,XYZ,...krc,Ak,AAA,aAA,...pxy,ap,...C,CS,...W,Wb,SbZ,...d,du,uuA->...prc',
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
            _tensor('Klo', TensorProto.FLOAT, (2, 2, 2), INIT_KLO_1),
            _tensor('C', TensorProto.FLOAT, (2, 10), INIT_C_2),
            _tensor('Mgate', TensorProto.FLOAT, (2, 2), INIT_MGATE_3),
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
