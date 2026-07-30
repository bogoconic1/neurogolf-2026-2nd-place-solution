from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task225'
TASK_NUM = 225
KAGGLE = {'score': 20.435652, 'date': '2026-07-12'}
MEMORY_BYTES = 0
PARAMS = 96
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = 'neurogolf-task225-p100-y2'
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task225_m0_p100_y2_nod'
OPSETS = [('', 19)]


# B: FLOAT[30, 3], 90 value(s)
INIT_B_0 = [0.06736771762371063, 0.003417505882680416, 0.028795275837183, -0.16836939752101898, 0.07762846350669861,
 0.008590707555413246, -0.04317976161837578, 0.06469903141260147, -0.0578654520213604, 0.10693181306123734,
 0.0009898318676277995, 0.048307426273822784, -0.16795183718204498, 0.07697832584381104, 0.008117725141346455,
 -0.05734643712639809, -0.007012618239969015, 0.01443200558423996, 11.780016899108887, -15.10678768157959,
 -25.982988357543945, 4.055363655090332, 4.019204616546631, -6.2870917320251465, -6.268202304840088,
 -14.796577453613281, 14.865251541137695, -5.649131774902344, -14.349519729614258, 14.167684555053711,
 -11.825608253479004, 15.293951034545898, 25.626811981201172, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# D1: FLOAT[3], 3 value(s)
INIT_D1_1 = [1.1204885244369507, 8.694847106933594, 8.891297340393066]

# D2: FLOAT[3], 3 value(s)
INIT_D2_2 = [-9.648993492126465, -22.771350860595703, 20.661348342895508]


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
            ['input', 'B', 'D1', 'B', 'D2', 'B', 'B', 'B', 'B', 'input', 'B', 'D1', 'B', 'D2', 'B', 'B', 'B',
             'B', 'B', 'input', 'B', 'B', 'B', 'B', 'input', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B',
             'input', 'input', 'B', 'D1', 'B', 'D2', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B',
             'B', 'B', 'input', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B',
             'input'],
            ['output'],
            name='output',
            domain='',
            equation='...dfg,fk,k,fl,l,fm,nm,nt,nt,...cuv,ux,x,uy,y,uQ,RQ,RS,RS,pA,...dpw,iA,pB,iB,iB,...chw,wC,jC,wD,jD,jD,hE,iE,hF,iF,iF,...ehq,...eTU,TV,V,TW,W,TX,YX,YZ,YZ,qG,jG,qH,jH,jH,aI,iI,aJ,iJ,iJ,...oab,bK,jK,bL,jL,jL,sM,jM,sN,jN,jN,rO,iO,rP,iP,iP,...zrs->...ors',
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
            _tensor('B', TensorProto.FLOAT, (30, 3), INIT_B_0),
            _tensor('D1', TensorProto.FLOAT, (3,), INIT_D1_1),
            _tensor('D2', TensorProto.FLOAT, (3,), INIT_D2_2),
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
