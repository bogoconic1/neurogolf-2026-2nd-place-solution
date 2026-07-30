from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task287'
TASK_NUM = 287
KAGGLE = {'score': 21.821946, 'date': '2026-07-14'}
MEMORY_BYTES = 0
PARAMS = 24
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 8
PRODUCER_NAME = ''
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task287_mix_k5_local_lt10_reorder'
OPSETS = [('', 20)]


# U: FLOAT[10, 2], 20 value(s)
INIT_U_0 = [0.0, 0.0, 0.8439467549324036, 0.5364269018173218, -0.5881636738777161, 0.8087419271469116, 0.8646422028541565,
 -0.5023881793022156, 0.0, 0.0, 0.17879389226436615, -0.9838865399360657, 0.23636727035045624, 0.971663773059845,
 -0.987993597984314, 0.15449489653110504, -0.6036368012428284, -0.7972594499588013, -0.978262722492218,
 -0.20736929774284363]

# B: FLOAT[2, 2], 4 value(s)
INIT_B_1 = [0.2295754998922348, -0.676461398601532, 0.6486964225769043, 0.262465238571167]


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
        # 0: Einsum inputs=47 outputs=1
        _node(
            'Einsum',
            ['U', 'U', 'input', 'input', 'U', 'U', 'input', 'input', 'input', 'input', 'U', 'U', 'input',
             'input', 'U', 'U', 'input', 'input', 'U', 'U', 'input', 'U', 'U', 'B', 'U', 'U', 'input', 'input',
             'U', 'U', 'input', 'input', 'U', 'U', 'input', 'input', 'U', 'U', 'input', 'input', 'input',
             'input', 'U', 'U', 'B', 'U', 'U'],
            ['output'],
            name='',
            domain='',
            equation='jl,jl,njhk,njrk,fi,fi,nfrg,nfhg,nahb,narb,ae,ae,nmho,nmro,mp,mp,nqht,nqrt,qu,qu,ndrs,dO,dM,LM,IK,IK,nIJs,nIJw,CE,CE,nCDw,nCDs,zB,zB,nzAw,nzAs,vy,vy,nvxw,nvxs,nFGs,nFGw,FH,FH,ON,cN,cL->nchw',
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
            _tensor('U', TensorProto.FLOAT, (10, 2), INIT_U_0),
            _tensor('B', TensorProto.FLOAT, (2, 2), INIT_B_1),
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
