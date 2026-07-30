from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task306'
TASK_NUM = 306
KAGGLE = {'score': 20.569183, 'date': '2026-07-10'}
MEMORY_BYTES = 0
PARAMS = 84
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 8
PRODUCER_NAME = ''
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task306_p84_power_81_23'
OPSETS = [('', 20)]


# code: FLOAT[30, 2], 60 value(s)
INIT_CODE_0 = [1.0, 0.0, 0.9510565400123596, 0.30901700258255005, 0.80901700258255, 0.5877852439880371, 0.5877852439880371,
 0.80901700258255, 0.30901697278022766, 0.9510565400123596, -4.371138828673793e-08, 1.0, -0.30901703238487244,
 0.9510564804077148, -0.5877854228019714, 0.8090168833732605, -0.80901700258255, 0.5877852439880371,
 -0.9510564804077148, 0.30901703238487244, 1.0, 0.0, 0.9510565400123596, 0.30901700258255005, 0.80901700258255,
 0.5877852439880371, 0.5877852439880371, 0.80901700258255, 0.30901697278022766, 0.9510565400123596,
 -4.371138828673793e-08, 1.0, -0.30901703238487244, 0.9510564804077148, -0.5877854228019714, 0.8090168833732605,
 -0.80901700258255, 0.5877852439880371, -0.9510564804077148, 0.30901703238487244, 1.0, 0.0, 0.9510565400123596,
 0.30901700258255005, 0.80901700258255, 0.5877852439880371, 0.5877852439880371, 0.80901700258255, 0.30901697278022766,
 0.9510565400123596, -4.371138828673793e-08, 1.0, -0.30901703238487244, 0.9510564804077148, -0.5877854228019714,
 0.8090168833732605, -0.80901700258255, 0.5877852439880371, 0.0, 0.0]

# Ccode: FLOAT[1, 10, 2], 20 value(s)
INIT_CCODE_1 = [0.48390495777130127, 0.5794053673744202, 3.0471432209014893, 1.7065043449401855, -0.17642472684383392, -4.758544921875,
 4.6558003425598145, -0.5247663855552673, 1.7005294561386108, 4.530010223388672, -4.7465715408325195, 2.081671714782715,
 -5.351457118988037, -1.0498065948486328, -1.203844666481018, 4.334376335144043, -2.794283866882324, 4.361884117126465,
 4.137637615203857, -3.5471229553222656]

# B: FLOAT[2, 2], 4 value(s)
INIT_B_2 = [1.0, 1.0139755010604858, -1.0139755010604858, 1.0]


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
        # 0: Einsum inputs=1684 outputs=1
        _node(
            'Einsum',
            ['B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B',
             'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B',
             'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B',
             'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B',
             'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B',
             'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B',
             'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B',
             'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B',
             'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B',
             'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B',
             'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B',
             'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B',
             'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B',
             'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B',
             'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B',
             'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B',
             'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B',
             'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B',
             'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B',
             'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B',
             'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B',
             'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B',
             'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B',
             'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B',
             'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B',
             'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B',
             'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B',
             'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B',
             'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B',
             'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B',
             'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'code', 'code', 'code', 'code',
             'code', 'code', 'code', 'code', 'code', 'code', 'code', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B',
             'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B',
             'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B',
             'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B',
             'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'code', 'code', 'B', 'B', 'B',
             'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B',
             'code', 'code', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B',
             'B', 'B', 'B', 'B', 'B', 'B', 'B', 'code', 'code', 'code', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B',
             'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B',
             'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B',
             'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B',
             'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B',
             'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B',
             'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B',
             'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B',
             'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B',
             'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B',
             'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B',
             'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B',
             'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B',
             'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'code',
             'code', 'input', 'Ccode', 'Ccode', 'code', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B',
             'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B',
             'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B',
             'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B',
             'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'code', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B',
             'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'code', 'code', 'code',
             'code', 'input', 'code', 'code', 'code', 'code', 'code', 'code', 'B', 'B', 'B', 'B', 'B', 'B', 'B',
             'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'code', 'B', 'B',
             'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B',
             'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B',
             'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B',
             'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B',
             'code', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B',
             'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B',
             'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B',
             'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B',
             'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B',
             'B', 'B', 'B', 'B', 'B', 'B', 'code', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B',
             'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B',
             'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B',
             'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B',
             'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B',
             'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B',
             'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B',
             'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B',
             'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'code', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B',
             'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B',
             'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B',
             'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B',
             'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B',
             'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B',
             'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B',
             'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B',
             'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'Ccode', 'Ccode'],
            ['output'],
            name='output',
            domain='',
            equation='ST,ST,ST,ST,ST,ST,ST,ST,ST,ST,ST,ST,ST,ST,ST,ST,ST,ST,ST,ST,ST,ST,ST,ST,ST,ST,ST,ST,ST,ST,ST,ST,ST,ST,ST,ST,ST,ST,ST,ST,ST,ST,ST,ST,ST,ST,ST,ST,ST,ST,ST,ST,ST,ST,ST,ST,ST,ST,ST,ST,ST,ST,ST,ST,ST,ST,ST,ST,ST,ST,ST,ST,ST,ST,ST,ST,ST,ST,ST,ST,ST,TU,TU,TU,TU,TU,TU,TU,TU,TU,TU,TU,TU,TU,TU,TU,TU,TU,TU,TU,TU,TU,TU,TU,TU,TU,TU,TU,TU,TU,TU,TU,TU,TU,TU,TU,TU,TU,TU,TU,TU,TU,TU,TU,TU,TU,TU,TU,TU,TU,TU,TU,TU,TU,TU,TU,TU,TU,TU,TU,TU,TU,TU,TU,TU,TU,TU,TU,TU,TU,TU,TU,TU,TU,TU,TU,TU,TU,TU,TU,TU,TU,Kk,Kk,Kk,Kk,Kk,Kk,Kk,Kk,Kk,Kk,Kk,Kk,Kk,Kk,Kk,Kk,Kk,Kk,Kk,Kk,Kk,Kk,Kk,Kk,Kk,Kk,Kk,Kk,Kk,Kk,Kk,Kk,Kk,Kk,Kk,Kk,Kk,Kk,Kk,Kk,Kk,Kk,Kk,Kk,Kk,Kk,Kk,Kk,Kk,Kk,Kk,Kk,Kk,Kk,Kk,Kk,Kk,Kk,Kk,Kk,Kk,Kk,Kk,Kk,Kk,Kk,Kk,Kk,Kk,Kk,Kk,Kk,Kk,Kk,Kk,Kk,Kk,Kk,Kk,Kk,Kk,kI,kI,kI,kI,kI,kI,kI,kI,kI,kI,kI,kI,kI,kI,kI,kI,kI,kI,kI,kI,kI,kI,kI,kI,kI,kI,kI,kI,kI,kI,kI,kI,kI,kI,kI,kI,kI,kI,kI,kI,kI,kI,kI,kI,kI,kI,kI,kI,kI,kI,kI,kI,kI,kI,kI,kI,kI,kI,kI,kI,kI,kI,kI,kI,kI,kI,kI,kI,kI,kI,kI,kI,kI,kI,kI,kI,kI,kI,kI,kI,kI,LM,LM,LM,LM,LM,LM,LM,LM,LM,LM,LM,LM,LM,LM,LM,LM,LM,LM,LM,LM,LM,LM,LM,LM,LM,LM,LM,LM,LM,LM,LM,LM,LM,LM,LM,LM,LM,LM,LM,LM,LM,LM,LM,LM,LM,LM,LM,LM,LM,LM,LM,LM,LM,LM,LM,LM,LM,LM,LM,LM,LM,LM,LM,LM,LM,LM,LM,LM,LM,LM,LM,LM,LM,LM,LM,LM,LM,LM,LM,LM,LM,NM,NM,NM,NM,NM,NM,NM,NM,NM,NM,NM,NM,NM,NM,NM,NM,NM,NM,NM,NM,NM,NM,NM,RQ,RQ,RQ,RQ,RQ,RQ,RQ,RQ,RQ,RQ,RQ,RQ,RQ,RQ,RQ,RQ,RQ,RQ,RQ,RQ,RQ,RQ,RQ,RQ,RQ,RQ,RQ,RQ,RQ,RQ,RQ,RQ,RQ,RQ,RQ,RQ,RQ,RQ,RQ,RQ,RQ,RQ,RQ,RQ,RQ,RQ,RQ,RQ,RQ,RQ,RQ,RQ,RQ,RQ,RQ,RQ,RQ,RQ,RQ,RQ,RQ,RQ,RQ,RQ,RQ,RQ,RQ,RQ,RQ,RQ,RQ,RQ,RQ,RQ,RQ,RQ,RQ,RQ,RQ,RQ,RQ,PQ,PQ,PQ,PQ,PQ,PQ,PQ,PQ,PQ,PQ,PQ,PQ,PQ,PQ,PQ,PQ,PQ,PQ,PQ,PQ,PQ,PQ,PQ,YX,YX,YX,YX,YX,YX,YX,YX,YX,YX,YX,YX,YX,YX,YX,YX,YX,YX,YX,YX,YX,YX,YX,YX,YX,YX,YX,YX,YX,YX,YX,YX,YX,YX,YX,YX,YX,YX,YX,YX,YX,YX,YX,YX,YX,YX,YX,YX,YX,YX,YX,YX,YX,YX,YX,YX,YX,YX,YX,YX,YX,YX,YX,YX,YX,YX,YX,YX,YX,YX,YX,YX,YX,YX,YX,YX,YX,YX,YX,YX,YX,wY,wR,wU,wN,wK,sI,sL,sP,sS,sX,wF,EF,EF,EF,EF,EF,EF,EF,EF,EF,EF,EF,EF,EF,EF,EF,EF,EF,EF,EF,EF,EF,EF,EF,EF,EF,EF,EF,EF,EF,EF,EF,EF,EF,EF,EF,EF,EF,EF,EF,EF,EF,EF,EF,EF,EF,EF,EF,EF,EF,EF,EF,EF,EF,EF,EF,EF,EF,EF,EF,EF,EF,EF,EF,EF,EF,EF,EF,EF,EF,EF,EF,EF,EF,EF,EF,EF,EF,EF,EF,EF,EF,sE,sG,GH,GH,GH,GH,GH,GH,GH,GH,GH,GH,GH,GH,GH,GH,GH,GH,GH,GH,GH,GH,GH,GH,GH,wH,wW,WV,WV,WV,WV,WV,WV,WV,WV,WV,WV,WV,WV,WV,WV,WV,WV,WV,WV,WV,WV,WV,WV,WV,sV,sO,wO,yz,yz,yz,yz,yz,yz,yz,yz,yz,yz,yz,yz,yz,yz,yz,yz,yz,yz,yz,yz,yz,yz,yz,yz,yz,yz,yz,yz,yz,yz,yz,yz,yz,yz,yz,yz,yz,yz,yz,yz,yz,yz,yz,yz,yz,yz,yz,yz,yz,yz,yz,yz,yz,yz,yz,yz,yz,yz,yz,yz,yz,yz,yz,yz,yz,yz,yz,yz,yz,yz,yz,yz,yz,yz,yz,yz,yz,yz,yz,yz,yz,xy,xy,xy,xy,xy,xy,xy,xy,xy,xy,xy,xy,xy,xy,xy,xy,xy,xy,xy,xy,xy,xy,xy,xy,xy,xy,xy,xy,xy,xy,xy,xy,xy,xy,xy,xy,xy,xy,xy,xy,xy,xy,xy,xy,xy,xy,xy,xy,xy,xy,xy,xy,xy,xy,xy,xy,xy,xy,xy,xy,xy,xy,xy,xy,xy,xy,xy,xy,xy,xy,xy,xy,xy,xy,xy,xy,xy,xy,xy,xy,xy,vu,vu,vu,vu,vu,vu,vu,vu,vu,vu,vu,vu,vu,vu,vu,vu,vu,vu,vu,vu,vu,vu,vu,vu,vu,vu,vu,vu,vu,vu,vu,vu,vu,vu,vu,vu,vu,vu,vu,vu,vu,vu,vu,vu,vu,vu,vu,vu,vu,vu,vu,vu,vu,vu,vu,vu,vu,vu,vu,vu,vu,vu,vu,vu,vu,vu,vu,vu,vu,vu,vu,vu,vu,vu,vu,vu,vu,vu,vu,vu,vu,tu,tu,tu,tu,tu,tu,tu,tu,tu,tu,tu,tu,tu,tu,tu,tu,tu,tu,tu,tu,tu,tu,tu,rt,rx,acrs,aci,acq,ra,ab,ab,ab,ab,ab,ab,ab,ab,ab,ab,ab,ab,ab,ab,ab,ab,ab,ab,ab,ab,ab,ab,ab,ab,ab,ab,ab,ab,ab,ab,ab,ab,ab,ab,ab,ab,ab,ab,ab,ab,ab,ab,ab,ab,ab,ab,ab,ab,ab,ab,ab,ab,ab,ab,ab,ab,ab,ab,ab,ab,ab,ab,ab,ab,ab,ab,ab,ab,ab,ab,ab,ab,ab,ab,ab,ab,ab,ab,ab,ab,ab,re,ef,ef,ef,ef,ef,ef,ef,ef,ef,ef,ef,ef,ef,ef,ef,ef,ef,ef,ef,ef,ef,ef,ef,hb,hf,hz,hv,adhw,hp,rp,rA,rC,rl,rg,BA,BA,BA,BA,BA,BA,BA,BA,BA,BA,BA,BA,BA,BA,BA,BA,BA,BA,BA,BA,BA,BA,BA,hB,DC,DC,DC,DC,DC,DC,DC,DC,DC,DC,DC,DC,DC,DC,DC,DC,DC,DC,DC,DC,DC,DC,DC,DC,DC,DC,DC,DC,DC,DC,DC,DC,DC,DC,DC,DC,DC,DC,DC,DC,DC,DC,DC,DC,DC,DC,DC,DC,DC,DC,DC,DC,DC,DC,DC,DC,DC,DC,DC,DC,DC,DC,DC,DC,DC,DC,DC,DC,DC,DC,DC,DC,DC,DC,DC,DC,DC,DC,DC,DC,DC,hD,lm,lm,lm,lm,lm,lm,lm,lm,lm,lm,lm,lm,lm,lm,lm,lm,lm,lm,lm,lm,lm,lm,lm,lm,lm,lm,lm,lm,lm,lm,lm,lm,lm,lm,lm,lm,lm,lm,lm,lm,lm,lm,lm,lm,lm,lm,lm,lm,lm,lm,lm,lm,lm,lm,lm,lm,lm,lm,lm,lm,lm,lm,lm,lm,lm,lm,lm,lm,lm,lm,lm,lm,lm,lm,lm,lm,lm,lm,lm,lm,lm,om,om,om,om,om,om,om,om,om,om,om,om,om,om,om,om,om,om,om,om,om,om,om,ho,ng,ng,ng,ng,ng,ng,ng,ng,ng,ng,ng,ng,ng,ng,ng,ng,ng,ng,ng,ng,ng,ng,ng,ng,ng,ng,ng,ng,ng,ng,ng,ng,ng,ng,ng,ng,ng,ng,ng,ng,ng,ng,ng,ng,ng,ng,ng,ng,ng,ng,ng,ng,ng,ng,ng,ng,ng,ng,ng,ng,ng,ng,ng,ng,ng,ng,ng,ng,ng,ng,ng,ng,ng,ng,ng,ng,ng,ng,ng,ng,ng,jn,jn,jn,jn,jn,jn,jn,jn,jn,jn,jn,jn,jn,jn,jn,jn,jn,jn,jn,jn,jn,jn,jn,jn,jn,jn,jn,jn,jn,jn,jn,jn,jn,jn,jn,jn,jn,jn,jn,jn,jn,jn,jn,jn,jn,jn,jn,jn,jn,jn,jn,jn,jn,jn,jn,jn,jn,jn,jn,jn,jn,jn,jn,jn,jn,jn,jn,jn,jn,jn,jn,jn,jn,jn,jn,jn,jn,jn,jn,jn,jn,hj,qZ,qZ,qZ,qZ,qZ,qZ,qZ,qZ,qZ,qZ,qZ,qZ,qZ,qZ,qZ,qZ,qZ,qZ,qZ,qZ,qZ,qZ,qZ,qZ,qZ,qZ,qZ,qZ,qZ,qZ,qZ,qZ,qZ,qZ,qZ,qZ,qZ,qZ,qZ,qZ,qZ,qZ,qZ,qZ,qZ,qZ,qZ,qZ,qZ,qZ,qZ,qZ,qZ,qZ,qZ,qZ,qZ,qZ,qZ,qZ,qZ,qZ,qZ,qZ,qZ,qZ,qZ,qZ,qZ,qZ,qZ,qZ,qZ,qZ,qZ,qZ,qZ,qZ,qZ,qZ,qZ,Ji,Ji,Ji,Ji,Ji,Ji,Ji,Ji,Ji,Ji,Ji,Ji,Ji,Ji,Ji,Ji,Ji,Ji,Ji,Ji,Ji,Ji,Ji,Ji,Ji,Ji,Ji,Ji,Ji,Ji,Ji,Ji,Ji,Ji,Ji,Ji,Ji,Ji,Ji,Ji,Ji,Ji,Ji,Ji,Ji,Ji,Ji,Ji,Ji,Ji,Ji,Ji,Ji,Ji,Ji,Ji,Ji,Ji,Ji,Ji,Ji,Ji,Ji,Ji,Ji,Ji,Ji,Ji,Ji,Ji,Ji,Ji,Ji,Ji,Ji,Ji,Ji,Ji,Ji,Ji,Ji,...Z,...J->...hw',
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
            _tensor('code', TensorProto.FLOAT, (30, 2), INIT_CODE_0),
            _tensor('Ccode', TensorProto.FLOAT, (1, 10, 2), INIT_CCODE_1),
            _tensor('B', TensorProto.FLOAT, (2, 2), INIT_B_2),
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
