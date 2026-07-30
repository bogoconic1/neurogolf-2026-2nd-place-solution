from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task294'
TASK_NUM = 294
KAGGLE = {'score': 20.522663, 'date': '2026-07-09'}
MEMORY_BYTES = 0
PARAMS = 88
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = 'task294_p88_asym_moment_k'
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task294_p88_asym'
OPSETS = [('', 18)]


# V: FLOAT[30, 2], 60 value(s)
INIT_V_0 = [1.0, 0.0, 0.8412535190582275, 0.5406408309936523, 0.4154150187969208, 0.9096319675445557, -0.1423148363828659,
 0.9898214340209961, -0.6548607349395752, 0.7557495832443237, -0.9594929814338684, 0.28173255920410156,
 -0.9594929814338684, -0.28173255920410156, -0.6548607349395752, -0.7557495832443237, -0.1423148363828659,
 -0.9898214340209961, 0.4154150187969208, -0.9096319675445557, 1.1554460525512695, -1.1554460525512695,
 -1.5047277212142944, 1.074805498123169, 1.7372572422027588, -0.7445388436317444, -1.8238352537155151,
 0.2605479061603546, 1.7341713905334473, 0.247738778591156, -1.4470057487487793, -0.620145320892334, 0.8362226486206055,
 0.5973019003868103, 0.8022340536117554, 0.8022340536117554, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# EC: FLOAT[2, 2, 2], 8 value(s)
INIT_EC_1 = [-0.5406408309936523, -0.8412535190582275, 0.8412535190582275, -0.5406408309936523, -0.28173255920410156,
 0.9594929814338684, -0.9594929814338684, -0.28173255920410156]

# E: FLOAT[2, 10], 20 value(s)
INIT_E_2 = [1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0]


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
        # 0: Einsum inputs=85 outputs=1
        _node(
            'Einsum',
            ['V', 'V', 'V', 'V', 'V', 'V', 'V', 'EC', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V',
             'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'EC', 'V', 'V', 'V', 'V', 'V', 'V',
             'EC', 'V', 'V', 'V', 'V', 'EC', 'V', 'E', 'input', 'V', 'EC', 'EC', 'V', 'V', 'EC', 'V', 'V', 'V',
             'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'EC', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V',
             'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'input', 'E'],
            ['output'],
            name='output',
            domain='',
            equation='nj,nj,nj,nj,ni,ni,nt,tYZ,wY,wT,cT,cZ,wV,cV,cS,wS,wQ,cQ,wJ,cJ,cU,wU,wR,cR,cP,wP,cL,wL,wK,cK,wG,jFG,cF,cN,wN,cO,wO,wH,jHI,cI,cM,wM,cW,tWX,wX,ia,...ahw,hC,tBC,tDE,hD,he,jef,rf,rB,rE,hg,rg,hy,ry,rx,hx,hu,ru,rb,jbd,hd,hm,rm,rs,hs,rk,hk,rz,hz,rq,hq,hl,rl,hp,hv,rp,rv,...Arc,jo->...orc',
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
            _tensor('V', TensorProto.FLOAT, (30, 2), INIT_V_0),
            _tensor('EC', TensorProto.FLOAT, (2, 2, 2), INIT_EC_1),
            _tensor('E', TensorProto.FLOAT, (2, 10), INIT_E_2),
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
