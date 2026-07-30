from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task059'
TASK_NUM = 59
KAGGLE = {'score': 20.478211, 'date': '2026-07-12'}
MEMORY_BYTES = 8
PARAMS = 84
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 7
PRODUCER_NAME = 'task059_delM_via_R_p84'
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task059_m8_p84_delM_via_R'
OPSETS = [('', 12)]


# P: FLOAT[2, 30], 60 value(s)
INIT_P_0 = [1.0, 1.0, 1.0, 1.7320507764816284, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.4214291572570801, -0.556746780872345,
 -0.993506669998169, 2.6815216541290283, -1.747287631034851, -0.5545783042907715, 0.4338341951370239,
 -0.3070501387119293, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.7320507764816284, 1.0,
 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.2642874717712402, 1.11349356174469, 0.993506669998169, 0.0, -1.747287631034851,
 -1.109156608581543, 1.3015025854110718, -1.2282005548477173, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# U: FLOAT[2, 10], 20 value(s)
INIT_U_1 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.10000000149011612, 0.0, 0.0, 0.0, 0.0, -0.009999999776482582, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0,
 1.0, 1.0, 1.0]

# R: FLOAT[2, 2], 4 value(s)
INIT_R_2 = [1.0, -1.0, 0.0, 1.0]


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
        # 0: Einsum inputs=76 outputs=1
        _node(
            'Einsum',
            ['R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'P', 'P', 'input', 'P', 'P', 'R',
             'R', 'P', 'P', 'input', 'P', 'P', 'R', 'R', 'P', 'P', 'input', 'P', 'P', 'R', 'R', 'P', 'P',
             'input', 'P', 'P', 'R', 'R', 'P', 'P', 'input', 'P', 'P', 'R', 'R', 'P', 'P', 'input', 'P', 'P',
             'R', 'R', 'P', 'P', 'input', 'P', 'P', 'R', 'R', 'P', 'P', 'input', 'P', 'P', 'R', 'U', 'U', 'U',
             'U', 'U', 'U', 'U', 'U'],
            ['I'],
            name='e1',
            domain='',
            equation='NA,NA,Na,Oa,OA,OA,PB,PB,Pb,Qb,QB,QB,Au,ud,ad,...cdl,bl,El,BE,Av,ve,ae,...cem,bm,Fm,BF,Aw,wf,af,...cfn,bn,Gn,BG,Ax,xg,ag,...cgo,bo,Ho,BH,Ay,yh,ah,...chp,bp,Ip,BI,Az,zi,ai,...ciq,bq,Jq,BJ,AC,Cj,aj,...cjr,br,Kr,BK,AD,ak,Dk,...cks,bs,Ls,BL,tc,tc,tc,tc,tc,tc,tc,tc->...t',
        ),
        # 1: Einsum inputs=57 outputs=1
        _node(
            'Einsum',
            ['R', 'R', 'R', 'R', 'R', 'R', 'I', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'U', 'U', 'U', 'U', 'R', 'P',
             'P', 'R', 'P', 'P', 'input', 'R', 'P', 'P', 'R', 'P', 'P', 'input', 'R', 'P', 'P', 'R', 'P', 'P',
             'input', 'R', 'P', 'P', 'R', 'P', 'P', 'input', 'P', 'R', 'P', 'P', 'R', 'P', 'U', 'U', 'input',
             'U', 'input'],
            ['output'],
            name='e2',
            domain='',
            equation='Ea,EA,EA,Fb,FB,FB,...t,tZ,rZ,rZ,rZ,rZ,qZ,qZ,qc,qc,qc,qc,An,ad,nd,By,bj,yj,...cdj,Ap,ae,pe,Bz,bk,zk,...cek,As,af,sf,BC,bl,Cl,...cfl,Ax,ai,xi,BD,bm,Dm,...cim,ah,AK,Kh,bw,BL,Lw,rg,rg,...ghw,ro,...ouv->...ohw',
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
            _tensor('P', TensorProto.FLOAT, (2, 30), INIT_P_0),
            _tensor('U', TensorProto.FLOAT, (2, 10), INIT_U_1),
            _tensor('R', TensorProto.FLOAT, (2, 2), INIT_R_2),
        ],
        value_info=[
            _vi('I', TensorProto.FLOAT, [1, 2]),
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
