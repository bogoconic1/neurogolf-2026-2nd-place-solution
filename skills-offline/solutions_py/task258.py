from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task258'
TASK_NUM = 258
KAGGLE = {'score': 20.751505, 'date': '2026-07-09'}
MEMORY_BYTES = 0
PARAMS = 70
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = 'task258_fitted_tail_router'
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task258_fitted_tail_router'
OPSETS = [('', 12)]


# E: FLOAT[2, 30], 60 value(s)
INIT_E_0 = [0.25, 0.2380833923816681, 0.20346961915493011, 0.14945848286151886, 0.08119907230138779, 0.0051987189799547195,
 -0.07129726558923721, -0.1409962773323059, -0.19725368916988373, -0.2347063422203064, -0.3072499930858612,
 -0.3072499930858612, -0.3072499930858612, -0.3072499930858612, -0.3072499930858612, -0.3032500147819519,
 -0.30063191056251526, -0.30063191056251526, -0.30063191056251526, 0.3019999861717224, 0.3617500066757202, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.07626465708017349, 0.1452587991952896, 0.20040498673915863,
 0.23644599318504333, 0.24994593858718872, 0.23961782455444336, 0.20644623041152954, 0.15359356999397278,
 0.08609837293624878, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0]

# C: FLOAT[10], 10 value(s)
INIT_C_1 = [1.0, -3.0, -0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


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
        # 0: Einsum inputs=81 outputs=1
        _node(
            'Einsum',
            ['E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E',
             'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E',
             'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E',
             'E', 'E', 'input', 'C', 'input', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'C', 'E',
             'E', 'E', 'E'],
            ['output'],
            name='fitted_coordinate_tail_router',
            domain='',
            equation='au,aw,bu,bw,cu,cw,du,dw,eu,ew,fu,fw,gu,gw,ku,kw,lu,lw,mu,mw,pu,pw,qu,qw,ru,rw,su,sw,tu,tw,vu,vw,xu,xw,yu,yw,zu,zw,Au,Aw,Bu,Bw,Cu,Cw,Du,Dw,Eu,Ew,Fu,Fw,Gu,Gw,Hu,Hw,Iu,Iw,Ju,Jw,Ku,Kw,Lu,Lw,njhu,j,nihw,QV,QV,QV,QV,QV,QV,QV,QV,QV,QV,QV,o,QV,QV,QV,QV->nohw',
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
            _tensor('E', TensorProto.FLOAT, (2, 30), INIT_E_0),
            _tensor('C', TensorProto.FLOAT, (10,), INIT_C_1),
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
