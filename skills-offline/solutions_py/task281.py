from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task281'
TASK_NUM = 281
KAGGLE = {'score': 20.317869, 'date': '2026-07-14'}
MEMORY_BYTES = 8
PARAMS = 100
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 8
PRODUCER_NAME = 'task281_updated_v5'
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task281_updated_v5_kf_fold'
OPSETS = [('', 14)]


# B: FLOAT[2, 30], 60 value(s)
INIT_B_0 = [4.315837287549584e-08, 2.4414063659605745e-07, 1.381067932015867e-06, 7.812500371073838e-06, 4.419417382450774e-05,
 0.0002500000118743628, 0.0014142135623842478, 0.00800000037997961, 0.04525483399629593, 0.25600001215934753,
 1.4481546878814697, 8.192000389099121, 46.34095001220703, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 46.34095001220703, 8.192000389099121, 1.4481546878814697, 0.25600001215934753,
 0.04525483399629593, 0.00800000037997961, 0.0014142135623842478, 0.0002500000118743628, 4.419417382450774e-05,
 7.812500371073838e-06, 1.381067932015867e-06, 2.4414063659605745e-07, 4.315837287549584e-08, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# H: FLOAT[2, 2, 2], 8 value(s)
INIT_H_1 = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, -1.0, 0.0]

# E: FLOAT[10, 2], 20 value(s)
INIT_E_2 = [1.0, 0.0, 32.015933990478516, 32.015933990478516, 64.06309509277344, 32.03154754638672, 96.14146423339844,
 32.04715347290039, 128.25100708007812, 32.06275177001953, 160.39170837402344, 32.07834243774414, 192.5635528564453,
 32.09392547607422, 224.76651000976562, 32.109500885009766, 0.0, 32.00031280517578, 289.2656555175781,
 32.140628814697266]

# Uf: FLOAT[3, 2], 6 value(s)
INIT_UF_3 = [40.647483825683594, 43.5723876953125, -3.079493761062622, -304.9317932128906, -6.824760437011719, -387.5617980957031]

# Cpow: FLOAT[3, 2], 6 value(s)
INIT_CPOW_4 = [1.9927959920385896e-15, 186.5601806640625, -0.0019444203935563564, 4.344481468200684, 0.001017831265926361,
 -2.319399356842041]


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
        # 0: Einsum inputs=30 outputs=1
        _node(
            'Einsum',
            ['input', 'input', 'input', 'input', 'input', 'input', 'input', 'input', 'input', 'input', 'input',
             'input', 'input', 'input', 'input', 'input', 'E', 'E', 'E', 'H', 'B', 'B', 'B', 'B', 'B', 'B', 'B',
             'B', 'H', 'H'],
            ['outer_E'],
            name='',
            domain='',
            equation='ncbe,ncfg,nchi,ncjk,nclm,ncop,ncqr,ncst,ncuv,ncwx,ncyz,ncCD,ncEF,ncGH,ncIJ,ncKL,ca,cd,cd,ABd,MO,MO,MO,MO,NO,NO,NO,NO,MPQ,RSN->na',
        ),
        # 1: Einsum inputs=88 outputs=1
        _node(
            'Einsum',
            ['E', 'E', 'input', 'B', 'B', 'input', 'E', 'E', 'B', 'B', 'E', 'E', 'input', 'B', 'B', 'B', 'B',
             'H', 'E', 'E', 'input', 'B', 'B', 'B', 'B', 'H', 'Uf', 'Uf', 'H', 'E', 'E', 'input', 'B', 'B', 'B',
             'B', 'Uf', 'H', 'E', 'E', 'input', 'B', 'B', 'B', 'B', 'H', 'Uf', 'H', 'B', 'B', 'Uf', 'E', 'E',
             'input', 'B', 'B', 'B', 'B', 'H', 'Uf', 'B', 'B', 'Uf', 'H', 'E', 'E', 'input', 'B', 'B', 'B', 'B',
             'H', 'Uf', 'Cpow', 'Cpow', 'Cpow', 'E', 'E', 'outer_E', 'E', 'H', 'E', 'H', 'outer_E', 'E', 'E',
             'input', 'input'],
            ['output'],
            name='',
            domain='',
            equation='na,na,...nKM,qM,aM,...nKQ,na,na,AQ,aQ,Ta,Ta,...TUV,eU,aU,fh,ah,afe,da,da,...djV,gj,bj,ih,bh,agi,re,ri,aab,Wa,Wa,...WXY,kX,aX,lh,ah,rk,alk,va,va,...vFZ,mF,bF,oh,bh,amo,ro,asq,sw,aw,rq,Ea,Ea,...ELN,bN,tN,uw,bw,atu,ru,Bw,aw,rA,aBA,pa,pa,...pOP,bP,CP,Dw,bw,aCD,rD,rx,rx,ry,cx,cx,...G,cH,yGH,cJ,yIJ,...I,cb,cb,...Shw,...cRz->...chw',
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
            _tensor('H', TensorProto.FLOAT, (2, 2, 2), INIT_H_1),
            _tensor('E', TensorProto.FLOAT, (10, 2), INIT_E_2),
            _tensor('Uf', TensorProto.FLOAT, (3, 2), INIT_UF_3),
            _tensor('Cpow', TensorProto.FLOAT, (3, 2), INIT_CPOW_4),
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
