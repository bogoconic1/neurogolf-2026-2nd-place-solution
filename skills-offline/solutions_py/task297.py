from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task297'
TASK_NUM = 297
KAGGLE = {'score': 20.841117, 'date': '2026-07-09'}
MEMORY_BYTES = 0
PARAMS = 64
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = 'task297_scratch64A'
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task297_scratch64A_local_exact'
OPSETS = [('', 17)]


# lin: FLOAT[2, 30], 60 value(s)
INIT_LIN_0 = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 3.42356014251709, -4.196007251739502,
 3.4468188285827637, 5.297926425933838, -5.775299072265625, 4.45958948135376, 5.46251106262207, 5.832603931427002,
 -6.550042629241943, 5.3528971672058105, 5.2750420570373535, 5.165621280670166, -5.488970756530762, -4.311265468597412,
 5.0315046310424805, -4.246199607849121, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5,
 -8.558899879455566, 9.091348648071289, -6.319167613983154, -7.946889400482178, 6.73784875869751, -3.7163245677948,
 -2.731255531311035, -0.9721006155014038, -1.091673731803894, 2.6764485836029053, 4.395868301391602, 6.0265583992004395,
 -8.2334566116333, -7.90398645401001, 10.901593208312988, -10.615499496459961]

# A: FLOAT[2, 2], 4 value(s)
INIT_A_1 = [1.0, 1.0, -1.0, 0.0]


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
        # 0: Einsum inputs=67 outputs=1
        _node(
            'Einsum',
            ['lin', 'lin', 'A', 'lin', 'lin', 'lin', 'lin', 'A', 'lin', 'lin', 'lin', 'lin', 'A', 'lin', 'lin',
             'A', 'A', 'lin', 'A', 'lin', 'lin', 'lin', 'A', 'A', 'lin', 'A', 'A', 'A', 'lin', 'lin', 'lin',
             'lin', 'A', 'A', 'A', 'A', 'A', 'lin', 'input', 'lin', 'lin', 'lin', 'lin', 'lin', 'lin', 'A', 'A',
             'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'input', 'A', 'lin', 'A', 'A', 'lin', 'lin', 'input',
             'lin', 'lin', 'lin', 'input'],
            ['output'],
            name='',
            domain='',
            equation='em,tm,Rt,em,Az,Az,Az,SR,Az,Dz,Dz,qz,SJ,ea,Aa,JK,SK,Ba,Bj,Vs,Vs,vs,cv,kc,fa,fM,fL,LM,Ly,Dy,Jy,gy,gh,hl,kE,kX,EX,Ey,noai,Xi,Mi,Ki,Ii,Wi,Gi,Nq,ON,OI,HI,OH,Uq,ZU,ZG,FG,ZF,nPyQ,wW,wy,uw,uW,Cy,Vb,npbx,Hx,Fx,ud,nrdx->noyx',
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
            _tensor('lin', TensorProto.FLOAT, (2, 30), INIT_LIN_0),
            _tensor('A', TensorProto.FLOAT, (2, 2), INIT_A_1),
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
