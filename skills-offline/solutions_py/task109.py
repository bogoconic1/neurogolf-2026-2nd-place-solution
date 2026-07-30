from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task109'
TASK_NUM = 109
KAGGLE = {'score': 19.351026, 'date': '2026-07-13'}
MEMORY_BYTES = 0
PARAMS = 284
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = ''
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'g'
OPSETS = [('', 14)]


# O: FLOAT[4, 30, 2], 240 value(s)
INIT_O_0 = [0.5, 0.0, 0.4330126941204071, 0.25, 0.25, 0.4330126941204071, 0.25, 0.4330126941204071, 0.4330126941204071, 0.25, 0.5,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 -3.0326921939849854, -0.6993587613105774, 2.0, 0.0, 1.7320507764816284, 1.0, 1.0, 1.7320507764816284,
 1.2246468525851679e-16, 2.0, 1.2246468525851679e-16, 2.0, 1.0, 1.7320507764816284, 1.7320507764816284, 1.0, 2.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -9.464101791381836,
 -9.464101791381836, 2.0, 0.0, 1.7320507764816284, 1.0, 1.0, 1.7320507764816284, 1.2246468525851679e-16, 2.0, -1.0,
 1.7320507764816284, -1.0, 1.7320507764816284, 1.2246468525851679e-16, 2.0, 1.0, 1.7320507764816284, 1.7320507764816284,
 1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -7.464101791381836,
 -12.928203582763672, 2.0, 0.0, 1.7320507764816284, 1.0, 1.0, 1.7320507764816284, 1.2246468525851679e-16, 2.0, -1.0,
 1.7320507764816284, -1.7320507764816284, 1.0, -1.7320507764816284, 1.0, -1.0, 1.7320507764816284,
 1.2246468525851679e-16, 2.0, 1.0, 1.7320507764816284, 1.7320507764816284, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, -3.740000009536743, -14.408203125]

# Fcol: FLOAT[10, 2], 20 value(s)
INIT_FCOL_1 = [0.0, 1.0, 0.75, 1.0, 0.800000011920929, 1.0, 0.8500000238418579, 1.0, 0.8999999761581421, 1.0, 0.949999988079071, 1.0,
 1.0, 1.0, 1.0499999523162842, 1.0, 1.100000023841858, 1.0, 1.149999976158142, 1.0]

# A: FLOAT[4, 4], 16 value(s)
INIT_A_2 = [2.0, -0.00439453125, 0.0, 0.0, -3.0625, 0.012451171875, -0.003662109375, 0.0, 2.237060546875, -0.0120849609375,
 0.008111000061035156, -0.0024461746215820312, -1.423583984375, 0.0076904296875, -0.006866455078125,
 0.003261566162109375]

# R2: FLOAT[2, 2], 4 value(s)
INIT_R2_3 = [-0.8660253882408142, 0.4999999701976776, -0.5, -0.8660253882408142]

# H: FLOAT[2, 2], 4 value(s)
INIT_H_4 = [-1.0, 50.0, -50.0, 0.5]


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
        # 0: Einsum inputs=57 outputs=1
        _node(
            'Einsum',
            ['O', 'O', 'input', 'O', 'O', 'A', 'H', 'O', 'R2', 'O', 'O', 'O', 'O', 'R2', 'O', 'O', 'O', 'R2',
             'R2', 'R2', 'R2', 'O', 'O', 'O', 'O', 'O', 'input', 'O', 'O', 'H', 'O', 'O', 'O', 'O', 'O', 'R2',
             'R2', 'R2', 'R2', 'R2', 'O', 'O', 'O', 'O', 'O', 'Fcol', 'Fcol', 'Fcol', 'H', 'H', 'H', 'Fcol',
             'Fcol', 'Fcol', 'R2', 'input', 'input'],
            ['output'],
            name='',
            domain='',
            equation='qya,qya,nmxy,qxb,qxb,sq,RR,WOR,ED,WwA,PwE,PjZ,PwI,HI,PwC,PwF,PwK,QK,JQ,CL,LB,stB,stH,stJ,stF,stD,nchw,WhA,WST,TT,Phi,Phf,Php,Phk,Phv,fM,Me,Xv,uX,lp,srk,sru,sre,srl,srg,cV,cG,cz,YV,GU,zz,oY,oU,oz,ig,noNN,ndrt->nort',
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
            _tensor('O', TensorProto.FLOAT, (4, 30, 2), INIT_O_0),
            _tensor('Fcol', TensorProto.FLOAT, (10, 2), INIT_FCOL_1),
            _tensor('A', TensorProto.FLOAT, (4, 4), INIT_A_2),
            _tensor('R2', TensorProto.FLOAT, (2, 2), INIT_R2_3),
            _tensor('H', TensorProto.FLOAT, (2, 2), INIT_H_4),
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
