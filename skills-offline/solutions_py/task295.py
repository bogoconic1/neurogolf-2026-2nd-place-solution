from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task295'
TASK_NUM = 295
KAGGLE = {'score': 20.841117, 'date': '2026-07-09'}
MEMORY_BYTES = 0
PARAMS = 64
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = ''
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'p64'
OPSETS = [('', 18)]


# F: FLOAT[2, 30], 60 value(s)
INIT_F_0 = [1.0, 3.0, 9.0, 27.0, 81.0, 243.0, 729.0, 2187.0, 6561.0, 19683.0, 59049.0, 177147.0, 531441.0, 1594323.0, 4782969.0,
 14348907.0, 43046720.0, 129140160.0, 387420480.0, 1162261504.0, 3486784512.0, 10460353536.0, 31381059584.0,
 94143176704.0, 282429521920.0, 847288598528.0, 2541865926656.0, 7625597517824.0, 22876793077760.0, 68630377136128.0,
 1.0, 0.3333333432674408, 0.1111111119389534, 0.03703703731298447, 0.012345679104328156, 0.004115226212888956,
 0.0013717421097680926, 0.00045724736992269754, 0.0001524157851235941, 5.08052617078647e-05, 1.6935087842284702e-05,
 5.6450294323440176e-06, 1.8816764395523933e-06, 6.272254609029915e-07, 2.090751536343305e-07, 6.969172261506174e-08,
 2.3230573020782685e-08, 7.743524044201422e-09, 2.5811748294302106e-09, 8.603915913063531e-10, 2.867971971021177e-10,
 9.559906338774127e-11, 3.186635677554506e-11, 1.0622118636061106e-11, 3.540706139740224e-12, 1.180235379913408e-12,
 3.9341178426945123e-13, 1.311372614231504e-13, 4.3712422733137996e-14, 1.4570806448335402e-14]

# J: FLOAT[2, 2], 4 value(s)
INIT_J_1 = [0.0, 9.999999717180685e-10, -3.333333331578814e-10, 3.333333331578814e-10]


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
        # 0: Einsum inputs=27 outputs=1
        _node(
            'Einsum',
            ['J', 'J', 'J', 'F', 'input', 'J', 'J', 'F', 'F', 'F', 'F', 'input', 'F', 'F', 'J', 'F', 'F', 'F',
             'input', 'F', 'input', 'input', 'F', 'F', 'F', 'J', 'F'],
            ['output'],
            name='',
            domain='',
            equation='QB,QD,pu,pa,nAha,qb,qd,dr,Dr,XZ,YW,nGhv,kv,kv,kl,lr,le,le,nohe,ks,nFhs,nohc,ic,ic,ir,ij,js->nors',
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
            _tensor('F', TensorProto.FLOAT, (2, 30), INIT_F_0),
            _tensor('J', TensorProto.FLOAT, (2, 2), INIT_J_1),
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
