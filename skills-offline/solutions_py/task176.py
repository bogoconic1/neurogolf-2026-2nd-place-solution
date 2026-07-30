from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task176'
TASK_NUM = 176
KAGGLE = {'score': 21.087977, 'date': '2026-07-12'}
MEMORY_BYTES = 0
PARAMS = 50
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = ''
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'g'
OPSETS = [('', 18)]


# U: FLOAT[2, 10], 20 value(s)
INIT_U_0 = [82774648.0, -76507136.0, -6267556.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -26360.56640625, 76507136.0, 0.0, 0.0,
 102747.578125, 0.0, 0.0, 0.0, 0.0, 0.0]

# Q: FLOAT[30], 30 value(s)
INIT_Q_1 = [6.875201512553144e-34, 3.746299026702692e-33, 8.211262503205742e-31, 6.543611472284569e-32, 9.533076395797199e-32,
 2.7460390625885886e-34, 1.2703012992528407e-34, 1.37111624063642e-34, 1.6047029620049609e-38, 6.420553381751306e-40,
 5.497885893789705e-33, 1.8702278588765755e-33, 6.875201512553144e-34, 3.746299026702692e-33, 8.211262503205742e-31,
 6.543611472284569e-32, 9.533076395797199e-32, 2.7460390625885886e-34, 1.2703012992528407e-34, 1.37111624063642e-34,
 1.6047029620049609e-38, 6.420553381751306e-40, 5.497885893789705e-33, 1.8702278588765755e-33, 6.875201512553144e-34,
 3.746299026702692e-33, 8.211262503205742e-31, 6.543611472284569e-32, 9.533076395797199e-32, 2.7460390625885886e-34]


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
        # 0: Einsum inputs=17 outputs=1
        _node(
            'Einsum',
            ['input', 'U', 'U', 'Q', 'input', 'U', 'U', 'Q', 'U', 'U', 'U', 'U', 'U', 'U', 'U', 'U', 'U'],
            ['output'],
            name='',
            domain='',
            equation='ndhi,sd,sk,i,nchw,sc,sk,w,rk,sk,so,to,uo,sA,sB,sC,sD->nohw',
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
            _tensor('U', TensorProto.FLOAT, (2, 10), INIT_U_0),
            _tensor('Q', TensorProto.FLOAT, (30,), INIT_Q_1),
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
