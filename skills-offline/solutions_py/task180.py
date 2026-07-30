from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task180'
TASK_NUM = 180
KAGGLE = {'score': 20.617973, 'date': '2026-07-09'}
MEMORY_BYTES = 0
PARAMS = 80
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = 'task180_p80_task257_spatial_refit_scale8'
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task180_p80_task257_spatial_refit_scale8'
OPSETS = [('', 18)]


# F: FLOAT[30, 2], 60 value(s)
INIT_F_0 = [0.15199999511241913, 0.0, 0.10748022794723511, 0.10748022794723511, 0.0, 0.15199999511241913, -0.10748022794723511,
 0.10748022794723511, 0.10175380110740662, 0.0, 0.07195080816745758, 0.07195080816745758, 0.0, 0.10175380110740662,
 -0.07195080816745758, 0.07195080816745758, 1.2623094320297241, -0.007034219801425934, -1.0052921772003174,
 0.9973955154418945, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# Ch: FLOAT[2, 10], 20 value(s)
INIT_CH_1 = [-17.363840103149414, 0.0, 0.0, 0.0, -15.361409187316895, 160.0702667236328, -28.21341896057129, 0.0, 0.0,
 90.44722747802734, -3.7492692470550537, 0.0, 0.0, 0.0, 28.1814022064209, -100.52336120605469, -114.95433044433594, 0.0,
 0.0, 84.71407318115234]


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
        # 0: Einsum inputs=79 outputs=1
        _node(
            'Einsum',
            ['input', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F',
             'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F',
             'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F',
             'F', 'F', 'F', 'F', 'F', 'F', 'Ch', 'Ch', 'Ch', 'F', 'F', 'F', 'F', 'F', 'F', 'Ch', 'Ch', 'Ch',
             'input', 'input'],
            ['output'],
            name='task180_p80_task257_spatial_refit_scale8',
            domain='',
            equation='nkhw,ha,hd,he,wi,wm,wo,ca,ca,cb,gf,gf,ge,li,li,lj,tp,tp,to,rb,rd,rf,ru,ru,rv,rv,rx,rx,ry,ry,rz,rz,rA,rA,rB,rB,rC,rC,rD,rD,rE,rE,sj,sm,sp,sF,sF,sG,sG,sH,sH,sI,sI,sJ,sJ,sK,sK,sL,sL,sM,sM,sN,sN,sO,sO,Tk,Vk,Yk,WU,WU,WV,ZY,ZY,ZX,Tq,Uq,Xq,nPrQ,nRSs->nqrs',
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
            _tensor('Ch', TensorProto.FLOAT, (2, 10), INIT_CH_1),
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
