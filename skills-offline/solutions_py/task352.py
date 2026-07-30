from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task352'
TASK_NUM = 352
KAGGLE = {'score': 20.617973, 'date': '2026-07-13'}
MEMORY_BYTES = 0
PARAMS = 80
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = 'task352_m0_p80'
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task352_m0_p80'
OPSETS = [('', 13)]


# F: FLOAT[30, 2], 60 value(s)
INIT_F_0 = [1.0, 0.0, 0.9594929814338684, 0.28173255920410156, 0.8412535190582275, 0.5406408309936523, 0.6548607349395752,
 0.7557495832443237, 0.4154150187969208, 0.9096319675445557, 0.1423148363828659, 0.9898214340209961,
 -0.1423148363828659, 0.9898214340209961, -0.4154150187969208, 0.9096319675445557, -0.6548607349395752,
 0.7557495832443237, -0.8412535190582275, 0.5406408309936523, 0.9725973010063171, -1.9451946020126343,
 -1.3307565450668335, 1.901080846786499, 1.5717722177505493, -1.3472332954406738, -1.5674082040786743,
 0.4478309154510498, -1.3969881534576416, -0.3991394639015198, 1.5267361402511597, 1.3086310625076294,
 -1.30604887008667, -1.8657840490341187, 0.9592348337173462, 1.9184696674346924, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# E: FLOAT[10, 2], 20 value(s)
INIT_E_1 = [-2.117466688156128, -2.1868226528167725, -1.8449349403381348, -4.687727451324463, 5.342056751251221,
 -10.322580337524414, -1.8913984298706055, 0.25145891308784485, 2.9671385288238525, -3.163510799407959,
 4.5901618003845215, -2.0332348346710205, 3.6415390968322754, 0.3082326352596283, 5.493553638458252,
 -1.6116142272949219, 3.147775411605835, 1.3466650247573853, 4.040937900543213, -2.7091996669769287]


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
        # 0: Einsum inputs=98 outputs=1
        _node(
            'Einsum',
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F',
             'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'E',
             'E', 'input', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F',
             'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F',
             'F', 'F', 'F', 'F', 'F', 'F', 'F', 'E', 'input', 'E', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'E', 'E'],
            ['output'],
            name='expr',
            domain='',
            equation='PY,PY,PY,PY,PX,PX,PZ,sH,sF,wF,wH,sO,wO,wJ,sJ,sL,wL,wK,sK,sG,wG,wA,sA,sM,wM,sz,wz,sB,wB,wE,sE,wD,sD,wI,sI,wN,sN,wC,sC,cX,cY,ncrs,rj,rl,hl,hj,hi,ri,rk,hk,rd,hd,hm,rm,hq,rq,he,re,hp,rp,hg,rg,rb,hb,ru,hu,ht,rt,ra,rf,hf,ha,QZ,QZ,QZ,QZ,QW,QW,QT,yW,yV,yV,yV,yV,yU,yU,xV,nxhw,xS,vT,vS,vS,vS,vS,vR,vR,oR,oU->nohw',
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
            _tensor('F', TensorProto.FLOAT, (30, 2), INIT_F_0),
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
