from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task084'
TASK_NUM = 84
KAGGLE = {'score': 20.478211, 'date': '2026-07-12'}
MEMORY_BYTES = 0
PARAMS = 92
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 8
PRODUCER_NAME = ''
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task084_m0_p96_try'
OPSETS = [('', 18)]


# feat: FLOAT[2, 30], 60 value(s)
INIT_FEAT_0 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0]

# M: FLOAT[2, 2, 2], 8 value(s)
INIT_M_1 = [1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0]

# H: FLOAT[10, 2], 20 value(s)
INIT_H_2 = [1.0, 8.115549087524414, 1.0, -22.1668643951416, 1.0, 5.304004669189453, 1.0, -18.276016235351562, 1.0,
 9.620107650756836, 1.0, -14.4449462890625, 1.0, -10.637798309326172, 1.0, -6.846980571746826, 1.0, -3.0744576454162598,
 1.0, 0.672335147857666]

# K7: FLOAT[2, 2], 4 value(s)
INIT_K7_3 = [0.38362622261047363, 0.3961816728115082, -1.7907136678695679, 0.699439525604248]


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
        # 0: Einsum inputs=49 outputs=1
        _node(
            'Einsum',
            ['feat', 'input', 'M', 'feat', 'M', 'M', 'feat', 'feat', 'M', 'M', 'M', 'feat', 'input', 'feat',
             'feat', 'M', 'feat', 'input', 'feat', 'M', 'M', 'M', 'K7', 'K7', 'M', 'K7', 'input', 'feat', 'M',
             'feat', 'M', 'M', 'K7', 'K7', 'M', 'K7', 'M', 'H', 'M', 'feat', 'M', 'M', 'H', 'H', 'input',
             'feat', 'feat', 'H', 'H'],
            ['output'],
            name='output',
            domain='',
            equation='Wd,...Mrd,UWY,Yr,QUV,VXZ,Zd,Pr,QQP,QQP,PPP,Qb,...crb,Pb,Dr,ACD,Cx,...frx,Fx,sAB,BEF,Qtp,tz,sp,zue,ke,...jra,Ja,GJK,Kr,sGI,Ogy,sy,iO,ggn,vn,OSi,HS,ILN,Na,uqm,glh,cq,cl,...crw,Ew,Lw,om,oh->...orw',
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
            _tensor('feat', TensorProto.FLOAT, (2, 30), INIT_FEAT_0),
            _tensor('M', TensorProto.FLOAT, (2, 2, 2), INIT_M_1),
            _tensor('H', TensorProto.FLOAT, (10, 2), INIT_H_2),
            _tensor('K7', TensorProto.FLOAT, (2, 2), INIT_K7_3),
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
