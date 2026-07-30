from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task217'
TASK_NUM = 217
KAGGLE = {'score': 20.695935, 'date': '2026-07-09'}
MEMORY_BYTES = 0
PARAMS = 74
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = 'task217_p78_Bmoment_scratch'
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task217_p78_B_scratch_interp'
OPSETS = [('', 18)]


# sign: FLOAT[10], 10 value(s)
INIT_SIGN_0 = [-1.0, 8.00100040435791, 8.00100040435791, 8.00100040435791, 8.00100040435791, 8.00100040435791, 8.00100040435791,
 8.00100040435791, 8.00100040435791, 8.00100040435791]

# F: FLOAT[30, 2], 60 value(s)
INIT_F_1 = [1.0, 0.0, 0.7660444378852844, 0.6427876353263855, 0.1736481785774231, 0.9848077297210693, -0.5, 0.8660253882408142,
 -0.9396926164627075, 0.3420201539993286, -0.9396926164627075, -0.3420201539993286, -0.5, -0.8660253882408142,
 0.1736481785774231, -0.9848077297210693, 0.7660444378852844, -0.6427876353263855, 1.2686330080032349,
 -1.2482166290283203, -1.0235083103179932, 1.1791982650756836, 1.2359610795974731, -1.202925205230713,
 -1.325059413909912, 1.2633622884750366, 0.1576385200023651, 1.3461213111877441, 1.2748467922210693,
 -0.6484197378158569, 1.274867057800293, -0.6484335064888, -1.4252294301986694, 0.6092520952224731, 1.3193614482879639,
 -0.3399972915649414, -1.3576117753982544, -0.8387080430984497, 1.3996691703796387, 0.9568499326705933,
 -1.313471794128418, -1.0095888376235962, 0.7415190935134888, 1.3389356136322021, -0.9008840918540955,
 -1.4359791278839111, 0.9545102119445801, 1.37654709815979, -0.3255537152290344, -1.4658753871917725,
 0.2637014389038086, 1.3412598371505737, 0.46494048833847046, 1.3446803092956543, 1.2748512029647827,
 -0.648422360420227, 0.4635886549949646, 1.3430840969085693, 1.274795651435852, -0.6484017372131348]

# D: FLOAT[2, 2], 4 value(s)
INIT_D_2 = [1.0, -3.0, 3.0, -1.0]


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
        # 0: Einsum inputs=116 outputs=1
        _node(
            'Einsum',
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F',
             'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F',
             'D', 'D', 'F', 'F', 'F', 'F', 'D', 'F', 'F', 'F', 'F', 'D', 'F', 'F', 'sign', 'input', 'F', 'F',
             'D', 'F', 'F', 'F', 'F', 'D', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F',
             'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'D', 'F', 'F', 'F', 'F', 'F', 'F', 'F',
             'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'D', 'F', 'F', 'input', 'input',
             'sign'],
            ['output'],
            name='output',
            domain='',
            equation='Jh,Jh,Jh,Je,Jh,Je,Jh,Je,Jh,Je,Jh,Jd,Jd,Jh,Jg,Na,Nh,Na,Kt,Kt,Kt,Kt,Kt,Kt,Kt,Kt,Kg,Kp,Kp,Kp,Kp,Km,Km,Ot,Ok,Ot,rd,rm,re,rp,pq,ef,rf,rf,rq,rq,ab,xa,xb,xb,xk,kl,xl,xl,c,ncxy,yE,yE,DE,yD,yv,yv,yu,uv,PD,PI,PI,Qu,Qu,QC,Rg,Rz,Rg,LC,LC,LC,LC,LC,LC,LC,LC,Lg,LA,LA,LA,LA,Lw,Lw,sw,sA,AB,sB,sB,Mg,MI,MI,MI,MI,MI,MI,MI,MI,MF,MF,sF,MG,MG,MG,MG,sG,GH,sH,sH,nTrs,noij,o->nors',
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
            _tensor('sign', TensorProto.FLOAT, (10,), INIT_SIGN_0),
            _tensor('F', TensorProto.FLOAT, (30, 2), INIT_F_1),
            _tensor('D', TensorProto.FLOAT, (2, 2), INIT_D_2),
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
