from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task376'
TASK_NUM = 376
KAGGLE = {'score': 20.336561, 'date': '2026-07-15'}
MEMORY_BYTES = 0
PARAMS = 106
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = 'task376_p112_shared_plane'
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task376_p112'
OPSETS = [('', 14)]


# P: FLOAT[30, 2], 60 value(s)
INIT_P_0 = [0.0, 1.0, 0.25, 1.0, 0.5, 1.0, 0.75, 1.0, 1.0, 1.0, 1.25, 1.0, 1.5, 1.0, 1.75, 1.0, 2.0, 1.0, 2.25, 1.0, 2.5, 1.0,
 2.75, 1.0, 3.0, 1.0, 3.25, 1.0, 3.5, 1.0, 3.75, 1.0, 4.0, 1.0, 4.25, 1.0, 4.5, 1.0, 4.75, 1.0, 5.0, 1.0, 'nan', 'nan',
 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan']

# B: FLOAT[10, 2], 20 value(s)
INIT_B_1 = [0.06722689419984818, 0.004558824002742767, -0.07563025504350662, -0.07294118404388428, -0.07563025504350662,
 -0.07294118404388428, -0.07563025504350662, -0.07294118404388428, -0.07563025504350662, -0.07294118404388428,
 -0.07563025504350662, -0.07294118404388428, -0.07563025504350662, -0.07294118404388428, -0.07563025504350662,
 -0.07294118404388428, -0.07563025504350662, -0.07294118404388428, -0.07563025504350662, -0.07294118404388428]

# S: FLOAT[3, 2], 6 value(s)
INIT_S_2 = [1.1434848308563232, -0.6271007657051086, -3.134650230407715, -1.9786192178726196, -0.20484521985054016,
 -0.28120553493499756]

# T: FLOAT[3, 3, 2], 18 value(s)
INIT_T_3 = [2.515888214111328, 1.688240885734558, 0.3674013316631317, -2.983819007873535, -893.6145629882812, -949.69482421875,
 -3.4286386966705322, -2.070930004119873, -0.5217958688735962, 4.052624225616455, 1333.97021484375, -1289.1121826171875,
 10139.2900390625, 4649.63427734375, 634.1947631835938, -17561.50390625, 2674693.0, 5870455.5]

# d: FLOAT[2], 2 value(s)
INIT_D_4 = [1.0, 0.49555549025535583]


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
        # 0: Einsum inputs=46 outputs=1
        _node(
            'Einsum',
            ['B', 'input', 'S', 'B', 'input', 'S', 'P', 'B', 'input', 'S', 'P', 'B', 'input', 'P', 'B', 'input',
             'T', 'S', 'd', 'P', 'B', 'input', 'S', 'd', 'P', 'B', 'input', 'S', 'd', 'P', 'B', 'input', 'd',
             'P', 'B', 'input', 'd', 'P', 'B', 'input', 'd', 'P', 'B', 'input', 'input', 'S'],
            ['output'],
            name='p108_reuse_Dy_diagonal',
            domain='',
            equation='gi,...geh,fl,jl,...jek,fp,cp,mp,...mno,ft,ct,qt,...qrs,cx,ux,...uvw,fZY,ZB,B,cB,yB,...yzA,ZF,F,cF,CF,...CDE,ZJ,J,cJ,GJ,...GHI,Y,cY,KY,...KLM,Y,cY,OY,...OPQ,Y,cY,SY,...STU,...bed,fi->...bcd',
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
            _tensor('P', TensorProto.FLOAT, (30, 2), INIT_P_0),
            _tensor('B', TensorProto.FLOAT, (10, 2), INIT_B_1),
            _tensor('S', TensorProto.FLOAT, (3, 2), INIT_S_2),
            _tensor('T', TensorProto.FLOAT, (3, 3, 2), INIT_T_3),
            _tensor('d', TensorProto.FLOAT, (2,), INIT_D_4),
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
