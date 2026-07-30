from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task353'
TASK_NUM = 353
KAGGLE = {'score': 20.415033, 'date': '2026-07-09'}
MEMORY_BYTES = 18
PARAMS = 80
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 8
PRODUCER_NAME = 'task353_symfoldMK'
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'symfoldMK'
OPSETS = [('', 21)]


# Q: FLOAT[2, 30], 60 value(s)
INIT_Q_0 = [0.0625, 0.125, 0.1875, 0.25, 0.3125, 0.375, 0.4375, 0.5, 0.5625, 0.625, 0.6875, 0.75, 0.8125, 0.875, 1.205991268157959,
 -1.1746845245361328, -1.1749616861343384, -1.2741631269454956, -1.1752840280532837, -1.1750540733337402,
 -1.175048589706421, 1.6793339252471924, -0.4610581696033478, 2.577227830886841, -1.172232985496521,
 -1.1747688055038452, -1.106789469718933, 0.7728405594825745, 1.9479107856750488, -2.6817588806152344, 1.0, 1.0, 1.0,
 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -2.772456645965576, -1.42477548122406, -1.424540400505066,
 -5.324024677276611, -1.424267053604126, -1.4244619607925415, -1.424466609954834, 4.2778472900390625,
 -2.245576858520508, -3.206195831298828, -1.4268602132797241, -1.4247039556503296, -1.4856665134429932,
 4.9948506355285645, -2.863065719604492, 3.5986080169677734]

# E: FLOAT[10, 2], 20 value(s)
INIT_E_1 = [-20.062593460083008, 0.0, 0.0, 0.0, 0.0, 0.0, 20.062593460083008, -4.0, 0.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0]


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
        # 0: Einsum inputs=12 outputs=1
        _node(
            'Einsum',
            ['Q', 'Q', 'Q', 'Q', 'Q', 'Q', 'Q', 'E', 'E', 'E', 'Q', 'input'],
            ['state_f'],
            name='',
            domain='',
            equation='tz,az,az,az,bz,aH,bW,Ci,Cj,Oi,iZ,nCHW->nt',
        ),
        # 1: Cast inputs=1 outputs=1
        _node(
            'Cast',
            ['state_f'],
            ['state_i8'],
            name='',
            domain='',
            to=3,
        ),
        # 2: Cast inputs=1 outputs=1
        _node(
            'Cast',
            ['state_i8'],
            ['state'],
            name='',
            domain='',
            to=1,
        ),
        # 3: Einsum inputs=27 outputs=1
        _node(
            'Einsum',
            ['Q', 'Q', 'Q', 'Q', 'Q', 'Q', 'Q', 'Q', 'Q', 'Q', 'state', 'Q', 'Q', 'Q', 'Q', 'Q', 'Q', 'Q', 'Q',
             'Q', 'Q', 'state', 'Q', 'E', 'E', 'E', 'input'],
            ['output'],
            name='',
            domain='',
            equation='pu,qu,qu,qz,az,az,az,bz,aH,bW,np,rv,sv,sv,ry,dy,dy,dy,ey,dH,eW,ns,iZ,Ci,Cj,Oi,nCHW->nOHW',
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
            _tensor('Q', TensorProto.FLOAT, (2, 30), INIT_Q_0),
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
