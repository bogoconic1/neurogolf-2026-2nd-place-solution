from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task356'
TASK_NUM = 356
KAGGLE = {'score': 20.617973, 'date': '2026-07-13'}
MEMORY_BYTES = 0
PARAMS = 80
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 9
PRODUCER_NAME = 'expand_simplify_recompress_Tm3v'
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task356_p80_Tm3v'
OPSETS = [('', 18)]


# R2: FLOAT[30, 2], 60 value(s)
INIT_R2_0 = [1.0, -1.0, 1.0, -0.8571428656578064, 1.0, -0.7142857313156128, 1.0, -0.5714285969734192, 1.0, -0.4285714328289032, 1.0,
 -0.2857142984867096, 1.0, -0.1428571492433548, 1.0, 0.0, 1.0, 0.1428571492433548, 1.0, 0.2857142984867096,
 -0.262118399143219, -0.702653706073761, -0.2564642131328583, -0.7000849843025208, -0.22382721304893494,
 -0.7006261348724365, 0.4469001889228821, 1.2694262266159058, 1.0682721138000488, 1.175777792930603,
 -1.0754905939102173, 1.2200204133987427, 1.2235664129257202, -1.3792067766189575, -1.3070881366729736,
 1.086181402206421, 1.5246835947036743, -0.1810692399740219, -1.4325337409973145, -0.5479530096054077,
 1.2713180780410767, 0.9544094204902649, -1.0719274282455444, -1.3417723178863525, -1.0659996271133423,
 1.223244547843933, -1.1198986768722534, 0.17081928253173828, -1.1198986768722534, 0.17081928253173828,
 -1.1198986768722534, 0.17081928253173828, -1.1198986768722534, 0.17081928253173828, -1.1198986768722534,
 0.17081928253173828, -1.1198986768722534, 0.17081928253173828, -1.1198986768722534, 0.17081928253173828]

# E: FLOAT[2, 10], 20 value(s)
INIT_E_1 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 16.0, 0.0]


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
        # 0: Einsum inputs=72 outputs=1
        _node(
            'Einsum',
            ['R2', 'R2', 'R2', 'R2', 'R2', 'R2', 'R2', 'R2', 'R2', 'R2', 'R2', 'R2', 'R2', 'R2', 'R2', 'R2',
             'R2', 'R2', 'R2', 'R2', 'R2', 'R2', 'R2', 'R2', 'E', 'input', 'R2', 'R2', 'input', 'E', 'R2', 'R2',
             'R2', 'R2', 'R2', 'R2', 'R2', 'R2', 'R2', 'R2', 'R2', 'R2', 'E', 'input', 'R2', 'R2', 'E', 'input',
             'R2', 'R2', 'R2', 'R2', 'R2', 'R2', 'R2', 'R2', 'R2', 'R2', 'E', 'input', 'R2', 'R2', 'R2', 'R2',
             'R2', 'R2', 'R2', 'R2', 'R2', 'R2', 'R2', 'E'],
            ['output'],
            name='output',
            domain='',
            equation='jG,jG,jG,jG,jF,jF,jH,iH,iB,iB,iB,iB,iA,iA,kH,kD,kD,kD,kD,kC,kC,xF,xB,xH,Hb,nbrx,yG,yD,ndry,Hd,rK,lK,lK,lL,lL,lL,lL,lV,OH,OV,Oz,Sz,za,narc,cA,cC,Ve,nepc,pL,pV,pP,mP,mP,mV,mQ,mQ,mQ,mQ,Vf,nfqc,qQ,qN,tV,tN,tN,tN,tN,tM,tM,rM,Uw,wo->norc',
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
            _tensor('R2', TensorProto.FLOAT, (30, 2), INIT_R2_0),
            _tensor('E', TensorProto.FLOAT, (2, 10), INIT_E_1),
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
