from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task345'
TASK_NUM = 345
KAGGLE = {'score': 20.617973, 'date': '2026-07-09'}
MEMORY_BYTES = 0
PARAMS = 80
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = ''
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task345_p80_dot28_scratch'
OPSETS = [('', 22)]


# P: FLOAT[30, 2], 60 value(s)
INIT_P_0 = [1.0, 0.0, 0.9594929814338684, 0.28173255920410156, 0.8412535190582275, 0.5406408309936523, 0.6548607349395752,
 0.7557495832443237, 0.4154150187969208, 0.9096319675445557, 0.1423148363828659, 0.9898214340209961,
 -0.1423148363828659, 0.9898214340209961, -0.4154150187969208, 0.9096319675445557, -0.6548607349395752,
 0.7557495832443237, -0.8412535190582275, 0.5406408309936523, 2.0018723011016846, -1.4736402034759521,
 -0.07372615486383438, -1.807160496711731, 1.9391944408416748, -1.8450127840042114, 1.23063063621521,
 -0.08706013858318329, 1.2787598371505737, 1.4020459651947021, -1.78534996509552, 0.9309116005897522,
 -1.195876121520996, -1.7269387245178223, 1.0679428577423096, 2.062016010284424, -0.23603005707263947,
 1.9166723489761353, -0.10159720480442047, 1.4153765439987183, -0.8450471758842468, -1.6818344593048096,
 -0.7739160060882568, -1.5273557901382446, 1.0679428577423096, 2.062016010284424, -1.1521693468093872, -2.2531578540802,
 -1.634519100189209, 1.552514910697937, -1.1200302839279175, -1.722693920135498, -0.03309774398803711,
 -1.861322045326233, -2.054539680480957, 1.8055806159973145, 1.2191829681396484, -0.09594511240720749,
 1.0679423809051514, 2.06201434135437]

# E: FLOAT[10, 2], 20 value(s)
INIT_E_1 = [-1.0, 0.0, 0.0, 0.0, 15.234594345092773, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


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
        # 0: Einsum inputs=99 outputs=1
        _node(
            'Einsum',
            ['P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P',
             'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P',
             'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P',
             'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P',
             'P', 'P', 'P', 'P', 'input', 'E', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'E', 'E', 'E',
             'input'],
            ['output'],
            name='',
            domain='',
            equation='Aq,As,As,At,At,At,At,At,At,xs,zt,xB,zB,xC,zC,xD,zD,xE,zE,xF,zF,xG,zG,xH,zH,xI,zI,xJ,zJ,xK,zK,xL,zL,xM,zM,xN,zN,xO,zO,xP,zP,xQ,zQ,xR,zR,xS,zS,xT,zT,xU,zU,xV,zV,xW,zW,xX,zX,xY,zY,xZ,zZ,xa,za,xb,zb,xd,zd,ge,gf,gf,ze,xf,jh,ji,ji,xh,zi,nk,nm,nm,nm,nm,xk,zm,...crz,cq,rv,wq,wu,wu,wv,wv,wv,wv,yu,op,lp,lp,...lyx->...oyx',
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
            _tensor('P', TensorProto.FLOAT, (30, 2), INIT_P_0),
            _tensor('E', TensorProto.FLOAT, (10, 2), INIT_E_1),
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
