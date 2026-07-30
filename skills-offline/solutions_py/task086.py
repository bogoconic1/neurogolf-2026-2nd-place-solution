from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task086'
TASK_NUM = 86
KAGGLE = {'score': 19.716796, 'date': '2026-07-15'}
MEMORY_BYTES = 0
PARAMS = 197
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 7
PRODUCER_NAME = ''
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task086_fold_kr_into_rc_cost197'
OPSETS = [('', 12)]


# F: FLOAT[30, 3], 90 value(s)
INIT_F_0 = [1.0, 1.0, 0.0, 1.0, 0.8660253882408142, 0.5, 1.0, 0.5, 0.8660253882408142, 1.0, 6.123234262925839e-17, 1.0, 1.0, -0.5,
 0.8660253882408142, 1.0, -0.8660253882408142, 0.5, 1.0, -1.0, 1.2246468525851679e-16, 1.0, -0.8660253882408142, -0.5,
 1.0, -0.5, -0.8660253882408142, 1.0, -1.8369701465288538e-16, -1.0, 1.0, 0.5, -0.8660253882408142, 1.0,
 0.8660253882408142, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# C4: FLOAT[3], 3 value(s)
INIT_C4_1 = [0.5, 1.0, 1.0]

# C5: FLOAT[3], 3 value(s)
INIT_C5_2 = [0.8660253882408142, 1.0, 1.0]

# UB: FLOAT[2, 3], 6 value(s)
INIT_UB_3 = [-1.3319095040200887e-14, 217.517333984375, 217.517333984375, -63.50251007080078, 63.50251007080078, 63.50251007080078]

# P: FLOAT[2, 10], 20 value(s)
INIT_P_4 = [0.0, 815.6900024414062, 815.6900024414062, 815.6900024414062, 815.6900024414062, 815.6900024414062, 815.6900024414062,
 815.6900024414062, 815.6900024414062, 815.6900024414062, -119.06719970703125, -119.06719970703125, -119.06719970703125,
 -119.06719970703125, -119.06719970703125, -119.06719970703125, -119.06719970703125, -119.06719970703125,
 -119.06719970703125, -119.06719970703125]

# LC: FLOAT[2, 2], 4 value(s)
INIT_LC_5 = [0.0003258724173065275, 0.009152969345450401, 0.0003865892649628222, 0.005946286488324404]

# RC: FLOAT[2, 2], 4 value(s)
INIT_RC_6 = [-0.04310724884271622, -0.14765673875808716, -0.0031238074880093336, -0.07759296149015427]

# UBn: FLOAT[3], 3 value(s)
INIT_UBN_7 = [-1.0, 1.0, 1.0]

# OA: FLOAT[2, 2], 4 value(s)
INIT_OA_8 = [-11874588.0, -4.9723350770136676e-08, 188443152.0, 4.585160880067056e-19]

# E: FLOAT[10, 5], 50 value(s)
INIT_E_9 = [0.4472191333770752, 0.7071045637130737, 0.40824705362319946, 0.2886742353439331, 0.22360610961914062,
 0.4472191333770752, -0.7071045637130737, 0.40824705362319946, 0.2886742353439331, 0.22360610961914062,
 0.4472191333770752, 0.0, -0.8164941072463989, 0.2886742353439331, 0.22360610961914062, 0.4472191333770752, 0.0, 0.0,
 -0.8660227060317993, 0.22360610961914062, 0.4472191333770752, 0.0, 0.0, 0.0, -0.8944244384765625, 0.4472191333770752,
 0.0, -0.8164941072463989, 0.2886742353439331, 0.22360610961914062, 0.4472191333770752, -0.7071045637130737,
 0.40824705362319946, 0.2886742353439331, 0.22360610961914062, 0.4472191333770752, -0.7071045637130737,
 0.40824705362319946, 0.2886742353439331, 0.22360610961914062, 0.4472191333770752, 0.0, 0.0, 0.0, -0.8944244384765625,
 0.4472191333770752, 0.0, -0.8164941072463989, 0.2886742353439331, 0.22360610961914062]

# Pr: FLOAT[2, 5], 10 value(s)
INIT_PR_10 = [407.83489990234375, 815.6950073242188, 815.695068359375, 815.6950073242188, 815.6949462890625, -119.06425476074219,
 -119.06793212890625, -119.06793212890625, -119.06793212890625, -119.06793212890625]


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
            ['C5', 'UBn', 'C4', 'UBn', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'C5', 'F', 'F', 'F', 'C4', 'F', 'F',
             'UB', 'RC', 'C5', 'UBn', 'UBn', 'C4', 'input', 'F', 'F', 'F', 'F', 'F', 'C5', 'F', 'F', 'F', 'F',
             'F', 'F', 'C4', 'F', 'input', 'UB', 'F', 'C4', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'C5', 'F', 'F',
             'F', 'UB', 'F', 'UBn', 'UB', 'LC', 'LC', 'LC', 'LC', 'LC', 'OA', 'LC', 'LC', 'OA', 'LC', 'LC',
             'UB', 'UBn', 'F', 'UB', 'RC', 'UB', 'F', 'F', 'F', 'F', 'F', 'C5', 'F', 'F', 'F', 'F', 'C4', 'F',
             'F', 'UB', 'LC', 'RC', 'Pr', 'E', 'E', 'input', 'P', 'input', 'E', 'E', 'Pr', 'RC', 'RC', 'P',
             'OA', 'E', 'E', 'P', 'input'],
            ['output'],
            name='sgnub_sel_first',
            domain='',
            equation='M,M,N,N,rN,rM,xM,xN,xQ,rQ,rP,P,xP,rO,xO,O,rR,xR,aR,au,S,S,T,T,...cxy,yT,yS,zS,zT,yV,V,zV,zW,yW,yX,yU,zU,U,zX,...crz,uF,rF,B,rB,hF,hB,hD,rD,hC,rC,C,rA,rE,hE,jE,hA,A,nA,nn,nn,nn,nn,nn,nn,nn,nn,nn,nn,qj,nG,G,zG,tX,tf,fL,zL,wG,wL,wI,zI,I,wJ,zJ,zH,zK,H,wH,wK,YK,qY,qg,gk,ck,dk,...dbm,sd,...ehw,ei,ci,li,ll,sl,sc,sp,cZ,oZ,po,...obv->...ohw',
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
            _tensor('C4', TensorProto.FLOAT, (3,), INIT_C4_1),
            _tensor('C5', TensorProto.FLOAT, (3,), INIT_C5_2),
            _tensor('UB', TensorProto.FLOAT, (2, 3), INIT_UB_3),
            _tensor('P', TensorProto.FLOAT, (2, 10), INIT_P_4),
            _tensor('LC', TensorProto.FLOAT, (2, 2), INIT_LC_5),
            _tensor('RC', TensorProto.FLOAT, (2, 2), INIT_RC_6),
            _tensor('UBn', TensorProto.FLOAT, (3,), INIT_UBN_7),
            _tensor('OA', TensorProto.FLOAT, (2, 2), INIT_OA_8),
            _tensor('E', TensorProto.FLOAT, (10, 5), INIT_E_9),
            _tensor('Pr', TensorProto.FLOAT, (2, 5), INIT_PR_10),
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
