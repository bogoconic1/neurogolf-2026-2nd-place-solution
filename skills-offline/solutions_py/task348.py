from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task348'
TASK_NUM = 348
KAGGLE = {'score': 20.545653, 'date': '2026-07-12'}
MEMORY_BYTES = 0
PARAMS = 86
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = ''
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'g'
OPSETS = [('', 13)]


# B: FLOAT[2, 30], 60 value(s)
INIT_B_0 = [1.0, 4.0, 16.0, 64.0, 256.0, 1024.0, 4096.0, 16384.0, 65536.0, 262144.0, 1048576.0, 4194304.0, 16777216.0, 67108864.0,
 268435456.0, 1073741824.0, 4294967296.0, 17179869184.0, 68719476736.0, 274877906944.0, 1099511627776.0,
 4398046511104.0, 17592186044416.0, 70368744177664.0, 281474976710656.0, 1125899906842624.0, 4503599627370496.0,
 1.8014398509481984e+16, 7.205759403792794e+16, 2.8823037615171174e+17, 1.0, 0.25, 0.0625, 0.015625, 0.00390625,
 0.0009765625, 0.000244140625, 6.103515625e-05, 1.52587890625e-05, 3.814697265625e-06, 9.5367431640625e-07,
 2.384185791015625e-07, 5.960464477539063e-08, 1.4901161193847656e-08, 3.725290298461914e-09, 9.313225746154785e-10,
 2.3283064365386963e-10, 5.820766091346741e-11, 1.4551915228366852e-11, 3.637978807091713e-12, 9.094947017729282e-13,
 2.2737367544323206e-13, 5.684341886080802e-14, 1.4210854715202004e-14, 3.552713678800501e-15, 8.881784197001252e-16,
 2.220446049250313e-16, 5.551115123125783e-17, 1.3877787807814457e-17, 3.469446951953614e-18]

# C: FLOAT[2, 10], 20 value(s)
INIT_C_1 = [5.2631452973628257e-08, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 8.41600581225066e-07, 1.6358486618628376e-06, 0.0,
 -226.0503692626953, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 225.91543579101562, 27.44498634338379, 0.0]

# J: FLOAT[2, 2], 4 value(s)
INIT_J_2 = [0.0, 1.0, -1.0, 0.0]

# V: FLOAT[2], 2 value(s)
INIT_V_3 = [16.0, 0.0625]


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
        # 0: Einsum inputs=127 outputs=1
        _node(
            'Einsum',
            ['input', 'C', 'V', 'V', 'V', 'V', 'C', 'C', 'C', 'B', 'B', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V',
             'J', 'J', 'B', 'B', 'input', 'B', 'B', 'B', 'B', 'B', 'J', 'B', 'C', 'B', 'J', 'B', 'C', 'V', 'B',
             'J', 'B', 'C', 'V', 'V', 'B', 'J', 'B', 'C', 'V', 'V', 'V', 'B', 'J', 'B', 'C', 'V', 'V', 'V', 'V',
             'B', 'J', 'B', 'C', 'V', 'V', 'V', 'V', 'V', 'B', 'J', 'B', 'C', 'V', 'V', 'V', 'V', 'V', 'V', 'B',
             'J', 'B', 'C', 'B', 'J', 'B', 'C', 'V', 'B', 'J', 'B', 'C', 'V', 'V', 'B', 'J', 'B', 'C', 'V', 'V',
             'V', 'B', 'J', 'B', 'C', 'V', 'V', 'V', 'V', 'B', 'J', 'B', 'C', 'V', 'V', 'V', 'V', 'V', 'B', 'J',
             'B', 'C', 'V', 'V', 'V', 'V', 'V', 'V', 'C'],
            ['output'],
            name='output',
            domain='',
            equation='nlst,xl,x,x,x,x,ah,bh,ch,as,bs,a,a,a,a,b,b,b,b,aA,bB,Ar,Br,ngrw,aw,Bw,At,bt,dw,de,et,eo,fw,fi,it,io,i,jw,jk,kt,ko,k,k,mw,mp,pt,po,p,p,p,qw,qu,ut,uo,u,u,u,u,vw,vy,yt,yo,y,y,y,y,y,zw,zC,Ct,Co,C,C,C,C,C,C,Dw,DE,Et,Eo,Fw,FG,Gt,Go,G,Hw,HI,It,Io,I,I,Jw,JK,Kt,Ko,K,K,K,Lw,LM,Mt,Mo,M,M,M,M,Nw,NO,Ot,Oo,O,O,O,O,O,Pw,PQ,Qt,Qo,Q,Q,Q,Q,Q,Q,Ro->norw',
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
            _tensor('C', TensorProto.FLOAT, (2, 10), INIT_C_1),
            _tensor('J', TensorProto.FLOAT, (2, 2), INIT_J_2),
            _tensor('V', TensorProto.FLOAT, (2,), INIT_V_3),
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
