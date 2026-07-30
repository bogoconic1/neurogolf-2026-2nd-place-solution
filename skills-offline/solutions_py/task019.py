from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task019'
TASK_NUM = 19
KAGGLE = {'score': 18.836685, 'date': '2026-07-14'}
MEMORY_BYTES = 369
PARAMS = 106
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = ''
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'g'
OPSETS = [('', 17)]


# p4: FLOAT[1, 1, 1, 1], 1 value(s)
INIT_P4_0 = [0.6850000023841858]

# z4: INT8[1, 1, 1, 1], 1 value(s)
INIT_Z4_1 = [0]

# o4: INT8[1, 1, 1, 1], 1 value(s)
INIT_O4_2 = [1]

# zi: INT8[1], 1 value(s)
INIT_ZI_3 = [0]

# onef: FLOAT[1], 1 value(s)
INIT_ONEF_4 = [1.0]

# dm5: INT8[1], 1 value(s)
INIT_DM5_5 = [-5]

# W: INT8[10, 1, 3, 3], 90 value(s)
INIT_W_6 = [34, 0, 34, 0, 101, 0, 34, 0, 34, 0, 0, 0, 0, -7, 0, 0, 0, 0, 0, 0, 0, 0, -11, 0, 0, 0, 0, 0, 0, 0, 0, -9, 0, 0, 0, 0,
 0, 0, 0, 0, -13, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, -15, -8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -9,
 -2, -9, 1, 114, 1, -9, 0, -9, 0, 0, 0, 0, -17, 0, 0, 0, 0]

# bias: INT32[10], 10 value(s)
INIT_BIAS_7 = [-68, -28, -52, -39, -67, 0, -76, 0, -110, -103]


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
        # 0: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p4'],
            ['P00'],
            name='',
            domain='',
            dtype=3,
            seed=1043315.0,
        ),
        # 1: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p4'],
            ['P01'],
            name='',
            domain='',
            dtype=3,
            seed=5299622.0,
        ),
        # 2: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p4'],
            ['P02'],
            name='',
            domain='',
            dtype=3,
            seed=546168.0,
        ),
        # 3: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p4'],
            ['P03'],
            name='',
            domain='',
            dtype=3,
            seed=641773.0,
        ),
        # 4: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p4'],
            ['P10'],
            name='',
            domain='',
            dtype=3,
            seed=5429621.0,
        ),
        # 5: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p4'],
            ['P11'],
            name='',
            domain='',
            dtype=3,
            seed=11617742.0,
        ),
        # 6: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p4'],
            ['P12'],
            name='',
            domain='',
            dtype=3,
            seed=165338.0,
        ),
        # 7: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p4'],
            ['P13'],
            name='',
            domain='',
            dtype=3,
            seed=2011.0,
        ),
        # 8: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p4'],
            ['P14'],
            name='',
            domain='',
            dtype=3,
            seed=81719.0,
        ),
        # 9: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p4'],
            ['P20'],
            name='',
            domain='',
            dtype=3,
            seed=6024455.0,
        ),
        # 10: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p4'],
            ['P21'],
            name='',
            domain='',
            dtype=3,
            seed=947278.0,
        ),
        # 11: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p4'],
            ['P22'],
            name='',
            domain='',
            dtype=3,
            seed=458638.0,
        ),
        # 12: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p4'],
            ['P23'],
            name='',
            domain='',
            dtype=3,
            seed=8890.0,
        ),
        # 13: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p4'],
            ['P30'],
            name='',
            domain='',
            dtype=3,
            seed=313593.0,
        ),
        # 14: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p4'],
            ['P40'],
            name='',
            domain='',
            dtype=3,
            seed=314831.0,
        ),
        # 15: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p4'],
            ['P41'],
            name='',
            domain='',
            dtype=3,
            seed=297770.0,
        ),
        # 16: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p4'],
            ['P42'],
            name='',
            domain='',
            dtype=3,
            seed=181222.0,
        ),
        # 17: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p4'],
            ['P43'],
            name='',
            domain='',
            dtype=3,
            seed=460958.0,
        ),
        # 18: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p4'],
            ['P51'],
            name='',
            domain='',
            dtype=3,
            seed=203723.0,
        ),
        # 19: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p4'],
            ['H2'],
            name='',
            domain='',
            dtype=3,
            seed=234169.0,
        ),
        # 20: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p4'],
            ['H4'],
            name='',
            domain='',
            dtype=3,
            seed=8369695.0,
        ),
        # 21: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p4'],
            ['H5'],
            name='',
            domain='',
            dtype=3,
            seed=1218996.0,
        ),
        # 22: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p4'],
            ['H6'],
            name='',
            domain='',
            dtype=3,
            seed=16027466.0,
        ),
        # 23: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p4'],
            ['W2'],
            name='',
            domain='',
            dtype=3,
            seed=19746878.0,
        ),
        # 24: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p4'],
            ['W3'],
            name='',
            domain='',
            dtype=3,
            seed=6713216.0,
        ),
        # 25: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p4'],
            ['W4'],
            name='',
            domain='',
            dtype=3,
            seed=6762747.0,
        ),
        # 26: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p4'],
            ['W5'],
            name='',
            domain='',
            dtype=3,
            seed=1037325.0,
        ),
        # 27: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p4'],
            ['H3'],
            name='',
            domain='',
            dtype=3,
            seed=1087657.0,
        ),
        # 28: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p4'],
            ['W6'],
            name='',
            domain='',
            dtype=3,
            seed=171921696.0,
        ),
        # 29: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p4'],
            ['B0'],
            name='',
            domain='',
            dtype=3,
            seed=60964936.0,
        ),
        # 30: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p4'],
            ['B1'],
            name='',
            domain='',
            dtype=3,
            seed=26591892.0,
        ),
        # 31: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p4'],
            ['B2'],
            name='',
            domain='',
            dtype=3,
            seed=514173088.0,
        ),
        # 32: Add inputs=2 outputs=1
        _node(
            'Add',
            ['B2', 'B2'],
            ['B2x2'],
            name='',
            domain='',
        ),
        # 33: Add inputs=2 outputs=1
        _node(
            'Add',
            ['B1', 'B2x2'],
            ['B12'],
            name='',
            domain='',
        ),
        # 34: Add inputs=2 outputs=1
        _node(
            'Add',
            ['B12', 'B12'],
            ['B12x2'],
            name='',
            domain='',
        ),
        # 35: Add inputs=2 outputs=1
        _node(
            'Add',
            ['B0', 'B12x2'],
            ['code'],
            name='',
            domain='',
        ),
        # 36: Sub inputs=2 outputs=1
        _node(
            'Sub',
            ['dm5', 'code'],
            ['d'],
            name='',
            domain='',
        ),
        # 37: Add inputs=2 outputs=1
        _node(
            'Add',
            ['H5', 'H6'],
            ['r4'],
            name='',
            domain='',
        ),
        # 38: Add inputs=2 outputs=1
        _node(
            'Add',
            ['H4', 'r4'],
            ['r3'],
            name='',
            domain='',
        ),
        # 39: Add inputs=2 outputs=1
        _node(
            'Add',
            ['H3', 'r3'],
            ['r2'],
            name='',
            domain='',
        ),
        # 40: Add inputs=2 outputs=1
        _node(
            'Add',
            ['W5', 'W6'],
            ['c4'],
            name='',
            domain='',
        ),
        # 41: Add inputs=2 outputs=1
        _node(
            'Add',
            ['W4', 'c4'],
            ['c3'],
            name='',
            domain='',
        ),
        # 42: Add inputs=2 outputs=1
        _node(
            'Add',
            ['W3', 'c3'],
            ['c2'],
            name='',
            domain='',
        ),
        # 43: Mul inputs=2 outputs=1
        _node(
            'Mul',
            ['r2', 'c4'],
            ['b24'],
            name='',
            domain='',
        ),
        # 44: Mul inputs=2 outputs=1
        _node(
            'Mul',
            ['r2', 'W6'],
            ['b25'],
            name='',
            domain='',
        ),
        # 45: Mul inputs=2 outputs=1
        _node(
            'Mul',
            ['r3', 'c4'],
            ['b34'],
            name='',
            domain='',
        ),
        # 46: Mul inputs=2 outputs=1
        _node(
            'Mul',
            ['r3', 'W6'],
            ['b35'],
            name='',
            domain='',
        ),
        # 47: Mul inputs=2 outputs=1
        _node(
            'Mul',
            ['r4', 'c4'],
            ['b44'],
            name='',
            domain='',
        ),
        # 48: Mul inputs=2 outputs=1
        _node(
            'Mul',
            ['r4', 'W6'],
            ['b45'],
            name='',
            domain='',
        ),
        # 49: Mul inputs=2 outputs=1
        _node(
            'Mul',
            ['H6', 'c3'],
            ['b53'],
            name='',
            domain='',
        ),
        # 50: Mul inputs=2 outputs=1
        _node(
            'Mul',
            ['H6', 'c4'],
            ['b54'],
            name='',
            domain='',
        ),
        # 51: Mul inputs=2 outputs=1
        _node(
            'Mul',
            ['H6', 'W6'],
            ['b55'],
            name='',
            domain='',
        ),
        # 52: ReduceMin inputs=1 outputs=1
        _node(
            'ReduceMin',
            ['r4'],
            ['r4s'],
            name='',
            domain='',
            axes=[0, 1, 2, 3],
            keepdims=0,
        ),
        # 53: ReduceMin inputs=1 outputs=1
        _node(
            'ReduceMin',
            ['o4'],
            ['o4s'],
            name='',
            domain='',
            axes=[0, 1, 2, 3],
            keepdims=0,
        ),
        # 54: ReduceMin inputs=1 outputs=1
        _node(
            'ReduceMin',
            ['c2'],
            ['c2s'],
            name='',
            domain='',
            axes=[0, 1, 2, 3],
            keepdims=0,
        ),
        # 55: ReduceMin inputs=1 outputs=1
        _node(
            'ReduceMin',
            ['c3'],
            ['c3s'],
            name='',
            domain='',
            axes=[0, 1, 2, 3],
            keepdims=0,
        ),
        # 56: ReduceMin inputs=1 outputs=1
        _node(
            'ReduceMin',
            ['r2'],
            ['r2s'],
            name='',
            domain='',
            axes=[0, 1, 2, 3],
            keepdims=0,
        ),
        # 57: ReduceMin inputs=1 outputs=1
        _node(
            'ReduceMin',
            ['r3'],
            ['r3s'],
            name='',
            domain='',
            axes=[0, 1, 2, 3],
            keepdims=0,
        ),
        # 58: Mul inputs=2 outputs=1
        _node(
            'Mul',
            ['r2s', 'c2s'],
            ['b22s'],
            name='',
            domain='',
        ),
        # 59: Mul inputs=2 outputs=1
        _node(
            'Mul',
            ['r2s', 'c3s'],
            ['b23s'],
            name='',
            domain='',
        ),
        # 60: Mul inputs=2 outputs=1
        _node(
            'Mul',
            ['r3s', 'c2s'],
            ['b32s'],
            name='',
            domain='',
        ),
        # 61: Mul inputs=2 outputs=1
        _node(
            'Mul',
            ['r4s', 'c2s'],
            ['b42s'],
            name='',
            domain='',
        ),
        # 62: Mul inputs=2 outputs=1
        _node(
            'Mul',
            ['r4s', 'c3s'],
            ['b43s'],
            name='',
            domain='',
        ),
        # 63: QLinearConv inputs=8 outputs=1
        _node(
            'QLinearConv',
            ['P00', 'onef', 'zi', 'd', 'onef', 'zi', 'onef', 'o4s'],
            ['pq00'],
            name='',
            domain='',
            kernel_shape=[1, 1],
        ),
        # 64: QLinearConv inputs=8 outputs=1
        _node(
            'QLinearConv',
            ['P01', 'onef', 'zi', 'd', 'onef', 'zi', 'onef', 'o4s'],
            ['pq01'],
            name='',
            domain='',
            kernel_shape=[1, 1],
        ),
        # 65: QLinearConv inputs=8 outputs=1
        _node(
            'QLinearConv',
            ['P02', 'onef', 'zi', 'd', 'onef', 'zi', 'onef', 'c2s'],
            ['pq02'],
            name='',
            domain='',
            kernel_shape=[1, 1],
        ),
        # 66: QLinearConv inputs=8 outputs=1
        _node(
            'QLinearConv',
            ['P03', 'onef', 'zi', 'd', 'onef', 'zi', 'onef', 'c3s'],
            ['pq03'],
            name='',
            domain='',
            kernel_shape=[1, 1],
        ),
        # 67: Concat inputs=6 outputs=1
        _node(
            'Concat',
            ['pq00', 'pq01', 'pq02', 'pq03', 'c4', 'W6'],
            ['row0'],
            name='',
            domain='',
            axis=3,
        ),
        # 68: QLinearConv inputs=8 outputs=1
        _node(
            'QLinearConv',
            ['P10', 'onef', 'zi', 'd', 'onef', 'zi', 'onef', 'o4s'],
            ['pq10'],
            name='',
            domain='',
            kernel_shape=[1, 1],
        ),
        # 69: QLinearConv inputs=8 outputs=1
        _node(
            'QLinearConv',
            ['P11', 'onef', 'zi', 'd', 'onef', 'zi', 'onef', 'o4s'],
            ['pq11'],
            name='',
            domain='',
            kernel_shape=[1, 1],
        ),
        # 70: QLinearConv inputs=8 outputs=1
        _node(
            'QLinearConv',
            ['P12', 'onef', 'zi', 'd', 'onef', 'zi', 'onef', 'c2s'],
            ['pq12'],
            name='',
            domain='',
            kernel_shape=[1, 1],
        ),
        # 71: QLinearConv inputs=8 outputs=1
        _node(
            'QLinearConv',
            ['P13', 'onef', 'zi', 'd', 'onef', 'zi', 'onef', 'c3s'],
            ['pq13'],
            name='',
            domain='',
            kernel_shape=[1, 1],
        ),
        # 72: Mul inputs=2 outputs=1
        _node(
            'Mul',
            ['P14', 'd'],
            ['ps14'],
            name='',
            domain='',
        ),
        # 73: Add inputs=2 outputs=1
        _node(
            'Add',
            ['c4', 'ps14'],
            ['pq14'],
            name='',
            domain='',
        ),
        # 74: Concat inputs=6 outputs=1
        _node(
            'Concat',
            ['pq10', 'pq11', 'pq12', 'pq13', 'pq14', 'W6'],
            ['row1'],
            name='',
            domain='',
            axis=3,
        ),
        # 75: QLinearConv inputs=8 outputs=1
        _node(
            'QLinearConv',
            ['P20', 'onef', 'zi', 'd', 'onef', 'zi', 'onef', 'r2s'],
            ['pq20'],
            name='',
            domain='',
            kernel_shape=[1, 1],
        ),
        # 76: QLinearConv inputs=8 outputs=1
        _node(
            'QLinearConv',
            ['P21', 'onef', 'zi', 'd', 'onef', 'zi', 'onef', 'r2s'],
            ['pq21'],
            name='',
            domain='',
            kernel_shape=[1, 1],
        ),
        # 77: QLinearConv inputs=8 outputs=1
        _node(
            'QLinearConv',
            ['P22', 'onef', 'zi', 'd', 'onef', 'zi', 'onef', 'b22s'],
            ['pq22'],
            name='',
            domain='',
            kernel_shape=[1, 1],
        ),
        # 78: QLinearConv inputs=8 outputs=1
        _node(
            'QLinearConv',
            ['P23', 'onef', 'zi', 'd', 'onef', 'zi', 'onef', 'b23s'],
            ['pq23'],
            name='',
            domain='',
            kernel_shape=[1, 1],
        ),
        # 79: Concat inputs=6 outputs=1
        _node(
            'Concat',
            ['pq20', 'pq21', 'pq22', 'pq23', 'b24', 'b25'],
            ['row2'],
            name='',
            domain='',
            axis=3,
        ),
        # 80: QLinearConv inputs=8 outputs=1
        _node(
            'QLinearConv',
            ['P30', 'onef', 'zi', 'd', 'onef', 'zi', 'onef', 'r3s'],
            ['pq30'],
            name='',
            domain='',
            kernel_shape=[1, 1],
        ),
        # 81: QLinearConv inputs=8 outputs=1
        _node(
            'QLinearConv',
            ['b55', 'onef', 'zi', 'd', 'onef', 'zi', 'onef', 'b32s'],
            ['pq32'],
            name='',
            domain='',
            kernel_shape=[1, 1],
        ),
        # 82: QLinearConv inputs=8 outputs=1
        _node(
            'QLinearConv',
            ['P51', 'onef', 'zi', 'd', 'onef', 'zi', 'onef', 'r3s'],
            ['p33_inner'],
            name='',
            domain='',
            kernel_shape=[1, 1],
        ),
        # 83: Mul inputs=2 outputs=1
        _node(
            'Mul',
            ['c3', 'p33_inner'],
            ['pq33'],
            name='',
            domain='',
        ),
        # 84: Concat inputs=6 outputs=1
        _node(
            'Concat',
            ['pq30', 'r3', 'pq32', 'pq33', 'b34', 'b35'],
            ['row3'],
            name='',
            domain='',
            axis=3,
        ),
        # 85: QLinearConv inputs=8 outputs=1
        _node(
            'QLinearConv',
            ['P40', 'onef', 'zi', 'd', 'onef', 'zi', 'onef', 'r4s'],
            ['pq40'],
            name='',
            domain='',
            kernel_shape=[1, 1],
        ),
        # 86: QLinearConv inputs=8 outputs=1
        _node(
            'QLinearConv',
            ['P41', 'onef', 'zi', 'd', 'onef', 'zi', 'onef', 'r4s'],
            ['pq41'],
            name='',
            domain='',
            kernel_shape=[1, 1],
        ),
        # 87: QLinearConv inputs=8 outputs=1
        _node(
            'QLinearConv',
            ['P42', 'onef', 'zi', 'd', 'onef', 'zi', 'onef', 'b42s'],
            ['pq42'],
            name='',
            domain='',
            kernel_shape=[1, 1],
        ),
        # 88: QLinearConv inputs=8 outputs=1
        _node(
            'QLinearConv',
            ['P43', 'onef', 'zi', 'd', 'onef', 'zi', 'onef', 'b43s'],
            ['pq43'],
            name='',
            domain='',
            kernel_shape=[1, 1],
        ),
        # 89: Concat inputs=6 outputs=1
        _node(
            'Concat',
            ['pq40', 'pq41', 'pq42', 'pq43', 'b44', 'b45'],
            ['row4'],
            name='',
            domain='',
            axis=3,
        ),
        # 90: Mul inputs=2 outputs=1
        _node(
            'Mul',
            ['H6', 'p33_inner'],
            ['pq51'],
            name='pq51',
            domain='',
        ),
        # 91: QLinearConv inputs=8 outputs=1
        _node(
            'QLinearConv',
            ['W4', 'onef', 'zi', 'd', 'onef', 'zi', 'onef', 'c2s'],
            ['p52_inner'],
            name='',
            domain='',
            kernel_shape=[1, 1],
        ),
        # 92: Mul inputs=2 outputs=1
        _node(
            'Mul',
            ['H6', 'p52_inner'],
            ['pq52'],
            name='',
            domain='',
        ),
        # 93: Concat inputs=6 outputs=1
        _node(
            'Concat',
            ['H6', 'pq51', 'pq52', 'b53', 'b54', 'b55'],
            ['row5'],
            name='',
            domain='',
            axis=3,
        ),
        # 94: Concat inputs=6 outputs=1
        _node(
            'Concat',
            ['row0', 'row1', 'row2', 'row3', 'row4', 'row5'],
            ['q6'],
            name='',
            domain='',
            axis=2,
        ),
        # 95: Concat inputs=7 outputs=1
        _node(
            'Concat',
            ['H6', 'H5', 'H4', 'H3', 'H2', 'z4', 'o4'],
            ['sr'],
            name='',
            domain='',
            axis=2,
        ),
        # 96: Concat inputs=7 outputs=1
        _node(
            'Concat',
            ['W6', 'W5', 'W4', 'W3', 'W2', 'z4', 'o4'],
            ['sc'],
            name='',
            domain='',
            axis=3,
        ),
        # 97: Mul inputs=2 outputs=1
        _node(
            'Mul',
            ['sr', 'sc'],
            ['k7'],
            name='',
            domain='',
        ),
        # 98: QLinearConv inputs=8 outputs=1
        _node(
            'QLinearConv',
            ['q6', 'onef', 'zi', 'k7', 'onef', 'zi', 'onef', 'o4s'],
            ['q12'],
            name='',
            domain='',
            kernel_shape=[7, 7],
            pads=[6, 6, 6, 6],
        ),
        # 99: QLinearMatMul inputs=8 outputs=1
        _node(
            'QLinearMatMul',
            ['d', 'onef', 'zi', 'code', 'onef', 'zi', 'onef', 'zi'],
            ['yzp4'],
            name='',
            domain='',
        ),
        # 100: ReduceMin inputs=1 outputs=1
        _node(
            'ReduceMin',
            ['yzp4'],
            ['yzp'],
            name='',
            domain='',
            axes=[0, 1, 2, 3],
            keepdims=0,
        ),
        # 101: QLinearConv inputs=9 outputs=1
        _node(
            'QLinearConv',
            ['q12', 'onef', 'o4s', 'W', 'onef', 'zi', 'onef', 'yzp', 'bias'],
            ['output'],
            name='',
            domain='',
            kernel_shape=[3, 3],
            pads=[1, 1, 19, 19],
        ),
    ]

    graph = helper.make_graph(
        nodes,
        GRAPH_NAME,
        [
            _vi('input', TensorProto.FLOAT, [1, 10, 30, 30]),
        ],
        [
            _vi('output', TensorProto.INT8, [1, 10, 30, 30]),
        ],
        initializer=[
            _tensor('p4', TensorProto.FLOAT, (1, 1, 1, 1), INIT_P4_0),
            _tensor('z4', TensorProto.INT8, (1, 1, 1, 1), INIT_Z4_1),
            _tensor('o4', TensorProto.INT8, (1, 1, 1, 1), INIT_O4_2),
            _tensor('zi', TensorProto.INT8, (1,), INIT_ZI_3),
            _tensor('onef', TensorProto.FLOAT, (1,), INIT_ONEF_4),
            _tensor('dm5', TensorProto.INT8, (1,), INIT_DM5_5),
            _tensor('W', TensorProto.INT8, (10, 1, 3, 3), INIT_W_6),
            _tensor('bias', TensorProto.INT32, (10,), INIT_BIAS_7),
        ],
        value_info=[
            _vi('P00', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('P01', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('P02', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('P03', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('P10', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('P11', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('P12', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('P13', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('P14', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('P20', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('P21', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('P22', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('P23', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('P30', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('P40', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('P41', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('P42', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('P43', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('P51', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('H2', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('H4', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('H5', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('H6', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('W2', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('W3', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('W4', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('W5', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('H3', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('W6', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('B0', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('B1', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('B2', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('B2x2', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('B12', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('B12x2', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('code', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('d', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('r4', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('r3', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('r2', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('c4', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('c3', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('c2', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('b24', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('b25', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('b34', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('b35', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('b44', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('b45', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('b53', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('b54', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('b55', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('r4s', TensorProto.INT8, []),
            _vi('o4s', TensorProto.INT8, []),
            _vi('c2s', TensorProto.INT8, []),
            _vi('c3s', TensorProto.INT8, []),
            _vi('r2s', TensorProto.INT8, []),
            _vi('r3s', TensorProto.INT8, []),
            _vi('b22s', TensorProto.INT8, []),
            _vi('b23s', TensorProto.INT8, []),
            _vi('b32s', TensorProto.INT8, []),
            _vi('b42s', TensorProto.INT8, []),
            _vi('b43s', TensorProto.INT8, []),
            _vi('pq00', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('pq01', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('pq02', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('pq03', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('row0', TensorProto.INT8, [1, 1, 1, 6]),
            _vi('pq10', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('pq11', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('pq12', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('pq13', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('ps14', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('pq14', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('row1', TensorProto.INT8, [1, 1, 1, 6]),
            _vi('pq20', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('pq21', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('pq22', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('pq23', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('row2', TensorProto.INT8, [1, 1, 1, 6]),
            _vi('pq30', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('pq32', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('p33_inner', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('pq33', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('row3', TensorProto.INT8, [1, 1, 1, 6]),
            _vi('pq40', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('pq41', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('pq42', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('pq43', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('row4', TensorProto.INT8, [1, 1, 1, 6]),
            _vi('pq51', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('p52_inner', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('pq52', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('row5', TensorProto.INT8, [1, 1, 1, 6]),
            _vi('q6', TensorProto.INT8, [1, 1, 6, 6]),
            _vi('sr', TensorProto.INT8, [1, 1, 7, 1]),
            _vi('sc', TensorProto.INT8, [1, 1, 1, 7]),
            _vi('k7', TensorProto.INT8, [1, 1, 7, 7]),
            _vi('q12', TensorProto.INT8, [1, 1, 12, 12]),
            _vi('yzp4', TensorProto.INT8, [1, 1, 1, 1]),
            _vi('yzp', TensorProto.INT8, []),
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
