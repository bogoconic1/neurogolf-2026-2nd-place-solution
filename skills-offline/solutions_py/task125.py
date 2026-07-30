from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task125'
TASK_NUM = 125
KAGGLE = {'score': 20.14797, 'date': '2026-07-14'}
MEMORY_BYTES = 0
PARAMS = 128
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = ''
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'g'
OPSETS = [('', 16)]


# F: FLOAT[30, 3], 90 value(s)
INIT_F_0 = [0.3233086168766022, 1.0, 0.0, 0.3233086168766022, 0.9135454297065735, 0.4067366421222687, 0.3233086168766022,
 0.6691305637359619, 0.7431448698043823, 0.3233086168766022, 0.30901697278022766, 0.9510565400123596,
 0.3233086168766022, -0.10452850908041, 0.9945219159126282, 0.3233086168766022, -0.5000000596046448, 0.8660253882408142,
 0.3233086168766022, -0.8090170621871948, 0.5877851843833923, 0.3233086168766022, -0.9781476259231567,
 0.20791161060333252, 0.3233086168766022, -0.978147566318512, -0.20791178941726685, 0.3233086168766022,
 -0.8090169429779053, -0.5877853631973267, 0.3233086168766022, -0.49999991059303284, -0.866025447845459,
 0.3233086168766022, -0.10452833771705627, -0.9945219159126282, 0.3233086168766022, 0.3090171217918396,
 -0.9510564804077148, 0.3233086168766022, 0.6691307425498962, -0.7431447505950928, 0.3233086168766022,
 0.913545548915863, -0.40673649311065674, -0.45335811376571655, 0.15058015286922455, -0.11007920652627945,
 -0.7760764360427856, 0.9515838623046875, -0.14804010093212128, -0.7964035272598267, -0.01403101533651352,
 0.023480908945202827, -0.735304057598114, 0.8751262426376343, -0.13844574987888336, -0.36904025077819824,
 -0.9465450048446655, 0.8124183416366577, -0.8249701857566833, 0.016266293823719025, 0.036982372403144836,
 -4.4267659187316895, -0.6155630946159363, -0.6039463877677917, -1.2734311819076538, -0.9619042277336121,
 -0.6481707096099854, -0.10089278221130371, 0.08979437500238419, -1.2257347106933594, -0.68760085105896,
 0.7060701251029968, 0.8657617568969727, -0.8370746374130249, -0.17786049842834473, 0.22720091044902802,
 -2.014146566390991, -0.24832306802272797, -0.24700944125652313, -0.8833444714546204, 0.2570362389087677,
 -0.01161869429051876, 3.722566843032837, 1.0375882387161255, 1.0111804008483887, -0.7952073812484741,
 -0.11981859803199768, 1.1560204029083252]

# O: FLOAT[3, 10], 30 value(s)
INIT_O_1 = [0.0, 0.0, 0.0, -12.621212005615234, 0.990851879119873, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.4848484992980957,
 0.0, 0.0, 15.960975646972656, 0.0, -2.003094434738159, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
 -0.9937213659286499, 0.0]

# J: FLOAT[2, 3], 6 value(s)
INIT_J_2 = [0.010977823287248611, 0.0, -0.011012011207640171, 0.0, 0.011045890860259533, -0.011012011207640171]

# V: FLOAT[2], 2 value(s)
INIT_V_3 = [7.647071344207834e-12, 5.242568185126473e-11]


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
        # 0: Einsum inputs=91 outputs=1
        _node(
            'Einsum',
            ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'input', 'O', 'J', 'J',
             'V', 'J', 'J', 'J', 'O', 'input', 'J', 'V', 'J', 'J', 'J', 'F', 'O', 'input', 'F', 'F', 'F', 'F',
             'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F',
             'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F',
             'F', 'F', 'F', 'F', 'F', 'F', 'input', 'O', 'J', 'J', 'J', 'J', 'J', 'O'],
            ['output'],
            name='output',
            domain='',
            equation='yE,yE,yB,yC,yD,iB,iE,iD,iC,fE,fC,fD,fB,iA,fA,...lei,ol,po,pP,a,aP,vP,vq,qk,...kef,vW,a,aW,sW,sr,eI,rm,...mjf,jI,jL,jK,jJ,jM,eJ,eK,eL,FL,FK,FJ,FM,FM,eM,HQ,eQ,eV,eS,eT,eU,eR,cQ,cR,cT,cU,cS,GT,GS,GU,GV,GV,cV,fY,NY,fx,fw,fg,fZ,fh,dZ,dY,dg,dh,dw,Oh,Og,Ow,Ox,Ox,dx,...ncd,tn,ut,ut,zt,vX,zX,Xb->...bcd',
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
            _tensor('F', TensorProto.FLOAT, (30, 3), INIT_F_0),
            _tensor('O', TensorProto.FLOAT, (3, 10), INIT_O_1),
            _tensor('J', TensorProto.FLOAT, (2, 3), INIT_J_2),
            _tensor('V', TensorProto.FLOAT, (2,), INIT_V_3),
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
