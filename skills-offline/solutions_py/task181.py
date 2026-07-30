from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task181'
TASK_NUM = 181
KAGGLE = {'score': 20.478211, 'date': '2026-07-09'}
MEMORY_BYTES = 0
PARAMS = 92
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 8
PRODUCER_NAME = 'task181_p92_qexpr'
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task181_p92_qexpr'
OPSETS = [('', 22)]


# E: FLOAT[10, 2], 20 value(s)
INIT_E_0 = [2388.652099609375, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -119.43260955810547,
 -39.81071853637695, 0.0, 0.0]

# Fpos: FLOAT[30, 2], 60 value(s)
INIT_FPOS_1 = [10.5, -9.5, 8.0, -7.0, 5.5, -4.5, 3.0, -2.0, 0.5, 0.5, -2.0, 3.0, -4.5, 5.5, -7.0, 8.0, -9.5, 10.5,
 -0.6225267648696899, 0.015665769577026367, -0.6343926787376404, 0.13569924235343933, -0.6560636162757874,
 0.21797868609428406, -8.55467414855957, 2.655623435974121, 2.589170455932617, -8.608322143554688, -0.6213234663009644,
 0.12683986127376556, -0.604119598865509, 0.10722862929105759, 3.603933811187744, 3.8492865562438965,
 30.995574951171875, -3.4433138370513916, -30.995574951171875, 3.4433138370513916, 30.995574951171875,
 -3.4433138370513916, -30.995574951171875, 3.4433138370513916, 30.995574951171875, -3.4433138370513916,
 -30.995574951171875, 3.4433138370513916, 30.995574951171875, -3.4433138370513916, -30.995574951171875,
 3.4433138370513916, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# U: FLOAT[2, 2], 4 value(s)
INIT_U_2 = [-5.576085567474365, 5.659550189971924, -0.004068101290613413, -0.0041321879252791405]

# V: FLOAT[2, 2], 4 value(s)
INIT_V_3 = [-5.577749252319336, 5.363809585571289, -5.576121807098389, 5.365375995635986]

# G: FLOAT[2, 2], 4 value(s)
INIT_G_4 = [-2.2773187160491943, -2.431962728500366, -2.287309408187866, -2.421332597732544]


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
        # 0: Einsum inputs=108 outputs=1
        _node(
            'Einsum',
            ['Fpos', 'Fpos', 'Fpos', 'input', 'E', 'E', 'Fpos', 'E', 'Fpos', 'Fpos', 'Fpos', 'Fpos', 'Fpos',
             'U', 'V', 'V', 'G', 'Fpos', 'Fpos', 'Fpos', 'Fpos', 'Fpos', 'U', 'V', 'V', 'G', 'Fpos', 'Fpos',
             'U', 'V', 'V', 'G', 'Fpos', 'Fpos', 'U', 'V', 'V', 'G', 'Fpos', 'Fpos', 'Fpos', 'Fpos', 'Fpos',
             'Fpos', 'Fpos', 'Fpos', 'Fpos', 'Fpos', 'Fpos', 'Fpos', 'Fpos', 'Fpos', 'V', 'G', 'G', 'G', 'G',
             'G', 'G', 'V', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'V', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'G',
             'V', 'V', 'V', 'V', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'U', 'U', 'U', 'V', 'G', 'G',
             'G', 'G', 'G', 'G', 'G', 'G', 'E', 'input', 'E', 'E', 'input'],
            ['output'],
            name='task181_p92_qexpr',
            domain='',
            equation='WP,Wb,Wh,nfuv,fP,fN,YN,fZ,KZ,vg,OT,Ox,Ox,xp,Ap,Bp,gp,sA,kB,Ry,RU,RU,yq,Dq,Eq,gq,sD,kE,zo,Fo,Go,go,sF,kG,mw,Hw,Iw,gw,Qm,sH,kI,LJ,Lt,Lt,Mg,Mt,Mt,sJ,il,it,it,it,lj,jl,jl,jl,jl,jl,jl,lC,Cl,Cl,Cl,Cl,Cl,Cl,Cl,Cl,lS,Sl,Sl,Sl,Sl,Sl,Sl,Sl,Sl,lX,lX,lX,lX,Xl,Xl,Xl,Xl,Xl,Xl,Xl,Xl,Xl,Xl,Vl,Vl,Vl,Vl,Vl,Vl,Vl,Vl,Vl,Vl,Vl,Vl,el,nerk,ca,da,ndrs->ncrs',
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
            _tensor('E', TensorProto.FLOAT, (10, 2), INIT_E_0),
            _tensor('Fpos', TensorProto.FLOAT, (30, 2), INIT_FPOS_1),
            _tensor('U', TensorProto.FLOAT, (2, 2), INIT_U_2),
            _tensor('V', TensorProto.FLOAT, (2, 2), INIT_V_3),
            _tensor('G', TensorProto.FLOAT, (2, 2), INIT_G_4),
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
