from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task231'
TASK_NUM = 231
KAGGLE = {'score': 20.905655, 'date': '2026-07-09'}
MEMORY_BYTES = 0
PARAMS = 60
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = ''
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task231_updated_v2'
OPSETS = [('', 12)]


# F: FLOAT[30, 2], 60 value(s)
INIT_F_0 = [-0.4803944230079651, -1.0309946537017822, -1.057277798652649, 0.3130705952644348, 1.224094033241272,
 0.10961220413446426, 0.28164732456207275, -1.151593804359436, -0.7534816265106201, 1.087578296661377,
 0.9286054372787476, 0.8738353848457336, -0.4803944230079651, -1.0309946537017822, -1.057277798652649,
 0.3130705952644348, 1.224094033241272, 0.10961220413446426, 0.28164732456207275, -1.151593804359436,
 -0.7534816265106201, 1.087578296661377, 0.9286054372787476, 0.8738353848457336, -4.1456761027477404e-22,
 -9.191796160473684e-22, -1.0171043912004356e-21, 3.9953967119079445e-22, 3.924459310084595e-22, 7.73704741530054e-23,
 4.3377836124638684e-23, -2.786521774504768e-22, -7.639588383625915e-23, 1.0622331419695357e-22, 1.0496875398950077e-22,
 8.474162831671398e-23, -2.2122641676618906e-23, -4.9050339256578485e-23, -5.50509738152104e-23, 3.5517070706939247e-23,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


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
        # 0: Einsum inputs=29 outputs=1
        _node(
            'Einsum',
            ['input', 'input', 'input', 'input', 'input', 'input', 'input', 'input', 'input', 'input', 'input',
             'input', 'input', 'input', 'input', 'input', 'input', 'input', 'input', 'input', 'input', 'input',
             'input', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['output'],
            name='',
            domain='',
            equation='bcrs,bfrg,bhri,bkrl,bmrn,borp,bqrt,burv,bwrx,byrz,bArB,bCrD,bErF,bGrH,bIrJ,bKrL,bMrN,bOrP,bQrR,bSrT,bUrV,bWrX,bYrZ,ja,sa,jd,sd,je,se->bcrj',
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
