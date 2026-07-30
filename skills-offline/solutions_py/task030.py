from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task030'
TASK_NUM = 30
KAGGLE = {'score': 20.751505, 'date': '2026-07-13'}
MEMORY_BYTES = 0
PARAMS = 70
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 8
PRODUCER_NAME = ''
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task030_fast_v49_70'
OPSETS = [('', 18)]


# E: FLOAT[30, 2], 60 value(s)
INIT_E_0 = [-0.2163507640361786, -0.3306761384010315, -0.2169109582901001, -0.3298190236091614, -0.2174711674451828,
 -0.32896193861961365, -0.2180313766002655, -0.3281048536300659, -0.218591570854187, -0.3272477388381958,
 -0.21915178000926971, -0.32639065384864807, -0.21971197426319122, -0.32553356885910034, -0.22027218341827393,
 -0.3246764540672302, -0.22083239257335663, -0.3238193690776825, -0.22139258682727814, -0.32296228408813477,
 -3.5807266235351562, 3.1784250736236572, 1.5187429189682007, -3.994269847869873, -3.0220422744750977,
 -2.273833751678467, -3.0280470848083496, -2.020606279373169, 3.909526824951172, 2.004128932952881, 3.3491625785827637,
 -2.5497074127197266, 1.0505644083023071, 3.4881882667541504, 2.287769079208374, 2.127855062484741, -2.6214332580566406,
 3.3779921531677246, 2.2659430503845215, 2.1319668292999268, 2.261147975921631, 2.1293179988861084, 2.7566187381744385,
 -2.5102529525756836, 2.130383253097534, -2.4937450885772705, -3.0677709579467773, -2.50407075881958, 2.263944387435913,
 2.123213768005371, 2.1322851181030273, -2.51277756690979, 2.9985625743865967, -2.5379157066345215, -2.3267388343811035,
 3.0526680946350098, -3.0297369956970215, -2.0256083011627197, -3.047497510910034, -0.48030245304107666]

# W: FLOAT[10], 10 value(s)
INIT_W_1 = [-1.999999987845058e-08, 2.0, 0.20000000298023224, 0.0, 0.20000000298023224, 0.0, 0.0, 0.0, 0.0, 0.0]


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
        # 0: Einsum inputs=86 outputs=1
        _node(
            'Einsum',
            ['W', 'W', 'W', 'W', 'W', 'W', 'E', 'E', 'E', 'input', 'E', 'E', 'input', 'E', 'W', 'W', 'W', 'W',
             'W', 'W', 'input', 'E', 'E', 'E', 'E', 'E', 'E', 'W', 'input', 'W', 'E', 'W', 'E', 'input', 'E',
             'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'input', 'W', 'W', 'W', 'input', 'E', 'E', 'E', 'E', 'E',
             'E', 'E', 'E', 'E', 'input', 'input', 'E', 'input', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E',
             'input', 'E', 'input', 'W', 'W', 'input', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'input', 'input',
             'W'],
            ['output'],
            name='',
            domain='',
            equation='H,H,H,H,H,H,da,da,db,...hef,ea,ra,...Hgi,gb,H,H,H,H,H,H,...hmn,mk,rk,lk,lj,lj,pj,H,...Hpu,H,sb,H,sj,...HAB,Aw,sw,xw,xv,xv,xv,xv,rv,yv,...hyz,H,H,H,...hFG,FD,rD,ED,EC,EC,EC,EC,sC,IC,...HIJ,...HWX,WR,...HPQ,PL,sL,ML,MK,MK,MK,MK,rK,NK,...hNO,sR,...hsc,h,h,...hUV,US,rS,TS,TR,TR,TR,TR,...qrc,...otc,o->...orc',
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
            _tensor('E', TensorProto.FLOAT, (30, 2), INIT_E_0),
            _tensor('W', TensorProto.FLOAT, (10,), INIT_W_1),
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
