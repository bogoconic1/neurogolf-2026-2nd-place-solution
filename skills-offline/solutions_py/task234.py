from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task234'
TASK_NUM = 234
KAGGLE = {'score': 19.918596, 'date': '2026-07-15'}
MEMORY_BYTES = 52
PARAMS = 109
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = ''
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task234_integer_tfidf'
OPSETS = [('', 17)]


# logits: FLOAT[1, 5], 5 value(s)
INIT_LOGITS_0 = [0.0, 0.0, 0.0, 0.0, 0.0]

# X: FLOAT[2, 30], 60 value(s)
INIT_X_1 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0]

# C: FLOAT[2, 10], 20 value(s)
INIT_C_2 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0]

# D: FLOAT[2, 2, 2], 8 value(s)
INIT_D_3 = [1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0]

# M: FLOAT[2, 2, 3], 12 value(s)
INIT_M_4 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, -2.0, 5.0, 0.0, 0.0, 2.0]

# U: FLOAT[2, 2], 4 value(s)
INIT_U_5 = [-7.503363609313965, -0.008874359540641308, 7.510756969451904, -0.008869989775121212]


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
        # 0: Multinomial inputs=1 outputs=1
        _node(
            'Multinomial',
            ['logits'],
            ['tok'],
            name='',
            domain='',
            sample_size=5,
            seed=2064.0,
        ),
        # 1: TfIdfVectorizer inputs=1 outputs=1
        _node(
            'TfIdfVectorizer',
            ['tok'],
            ['BR'],
            name='BR',
            domain='',
            max_gram_length=5,
            max_skip_count=4,
            min_gram_length=1,
            mode='TFIDF',
            ngram_counts=[0, 3, 37, 232, 364],
            ngram_indexes=[0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 2, 0, 2, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1,
 0, 2, 0, 0, 0, 1, 1, 0, 1, 2, 0, 2, 0, 0, 0, 2, 1, 2, 0, 1, 0, 0, 2, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0,
 0, 0, 1, 2, 0, 1, 1, 0, 0, 0, 2, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 2, 0, 1, 1, 1, 0, 1,
 1, 1, 1, 1, 1, 1, 0, 0, 1, 2, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 2, 1, 2, 2, 1, 1],
            pool_int64s=[2, 3, 4, 0, 0, 0, 2, 0, 3, 1, 0, 1, 1, 1, 2, 1, 3, 2, 0, 2, 4, 3, 0, 3, 1, 3, 2, 3, 3, 3, 4, 4, 1,
 4, 2, 4, 4, 0, 0, 0, 0, 0, 1, 0, 0, 2, 0, 0, 3, 0, 0, 4, 0, 1, 1, 0, 1, 4, 0, 2, 0, 0, 2, 1, 0, 2,
 3, 0, 3, 0, 0, 3, 1, 0, 3, 4, 0, 4, 0, 0, 4, 2, 0, 4, 3, 1, 0, 1, 1, 0, 3, 1, 0, 4, 1, 1, 0, 1, 1,
 2, 1, 1, 4, 1, 2, 1, 1, 2, 2, 1, 4, 0, 1, 4, 1, 1, 4, 3, 1, 4, 4, 2, 0, 0, 2, 0, 3, 2, 0, 4, 2, 1,
 1, 2, 1, 4, 2, 2, 0, 2, 2, 1, 2, 2, 2, 2, 2, 3, 2, 2, 4, 2, 3, 1, 2, 3, 2, 2, 3, 3, 2, 4, 2, 3, 0,
 0, 3, 0, 2, 3, 0, 4, 3, 1, 0, 3, 1, 1, 3, 1, 3, 3, 1, 4, 3, 3, 0, 3, 3, 1, 3, 3, 4, 3, 4, 1, 3, 4,
 2, 3, 4, 4, 4, 0, 0, 4, 0, 1, 4, 0, 2, 4, 1, 0, 4, 1, 1, 4, 1, 4, 4, 3, 0, 4, 3, 1, 4, 3, 3, 4, 4,
 0, 0, 0, 0, 2, 0, 1, 1, 4, 0, 3, 1, 3, 0, 4, 1, 4, 1, 0, 2, 1, 1, 0, 3, 1, 1, 0, 4, 1, 1, 1, 0, 4,
 1, 1, 4, 3, 1, 1, 4, 4, 1, 2, 0, 0, 1, 4, 0, 2, 1, 4, 3, 0, 2, 0, 0, 4, 2, 0, 3, 4, 2, 2, 0, 3, 2,
 2, 2, 3, 2, 2, 3, 1, 2, 4, 1, 0, 3, 0, 0, 0, 3, 0, 4, 1, 3, 0, 4, 3, 3, 1, 4, 0, 3, 1, 4, 1, 3, 4,
 1, 1, 4, 0, 0, 3, 4, 0, 3, 1, 4, 1, 0, 2, 4, 1, 0, 3, 4, 1, 0, 4, 4, 1, 1, 0, 4, 3, 0, 4, 4, 4, 0,
 0, 0, 1, 1, 4, 4, 1, 1, 4, 3, 0, 1, 2, 0, 0, 4, 2, 2, 0, 3, 4, 2, 2, 2, 3, 1, 3, 0, 0, 0, 2, 3, 1,
 4, 0, 2, 3, 4, 1, 1, 0, 4, 0, 3, 1, 3, 4, 1, 0, 2, 1, 4, 4, 0, 0, 3],
            weights=[1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ),
        # 2: TfIdfVectorizer inputs=1 outputs=1
        _node(
            'TfIdfVectorizer',
            ['tok'],
            ['BS'],
            name='BS',
            domain='',
            max_gram_length=5,
            max_skip_count=4,
            min_gram_length=1,
            mode='TFIDF',
            ngram_counts=[0, 3, 31, 223, 351],
            ngram_indexes=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 2, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0,
 1, 0, 0, 0, 2, 2, 2, 1, 0, 0, 0, 1, 2, 0, 1, 1, 1, 1, 2, 0, 1, 0, 1, 1, 2, 0, 2, 0, 1, 0, 0, 0, 1,
 1, 2, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1,
 1, 1, 1, 1, 1, 2, 0, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2],
            pool_int64s=[0, 1, 3, 0, 0, 0, 1, 0, 3, 1, 1, 1, 2, 1, 3, 2, 2, 2, 3, 3, 0, 3, 3, 4, 0, 4, 1, 4, 2, 4, 4, 0, 0,
 0, 0, 0, 1, 0, 0, 3, 0, 0, 4, 0, 1, 1, 0, 1, 2, 0, 1, 4, 0, 2, 0, 0, 2, 1, 0, 2, 3, 0, 3, 0, 0, 3,
 1, 0, 3, 4, 0, 4, 0, 0, 4, 2, 0, 4, 3, 1, 0, 0, 1, 0, 2, 1, 0, 3, 1, 0, 4, 1, 1, 0, 1, 1, 4, 1, 2,
 0, 1, 2, 2, 1, 4, 0, 1, 4, 1, 1, 4, 3, 2, 0, 0, 2, 0, 3, 2, 0, 4, 2, 1, 0, 2, 1, 1, 2, 1, 4, 2, 2,
 1, 2, 2, 2, 2, 2, 3, 2, 2, 4, 2, 3, 1, 2, 3, 3, 2, 4, 0, 2, 4, 1, 2, 4, 2, 3, 0, 4, 3, 1, 0, 3, 1,
 1, 3, 1, 3, 3, 1, 4, 3, 2, 3, 3, 3, 0, 3, 3, 1, 3, 3, 4, 3, 4, 1, 3, 4, 2, 3, 4, 4, 4, 0, 0, 4, 0,
 2, 4, 0, 3, 4, 1, 0, 4, 1, 4, 4, 2, 0, 4, 2, 1, 4, 3, 0, 4, 3, 3, 4, 4, 0, 0, 0, 1, 2, 0, 2, 3, 3,
 0, 3, 1, 3, 0, 4, 1, 4, 0, 4, 2, 1, 1, 0, 0, 1, 1, 0, 2, 1, 1, 0, 3, 1, 1, 1, 4, 3, 1, 2, 0, 0, 1,
 2, 0, 4, 1, 2, 2, 1, 1, 4, 0, 2, 2, 0, 4, 0, 2, 2, 1, 1, 2, 2, 2, 3, 2, 2, 3, 1, 2, 3, 2, 3, 2, 4,
 1, 0, 2, 4, 2, 0, 3, 0, 4, 1, 3, 0, 4, 3, 3, 1, 4, 1, 3, 3, 1, 4, 3, 4, 1, 1, 4, 0, 0, 3, 4, 1, 0,
 2, 4, 1, 0, 4, 4, 1, 1, 0, 4, 2, 0, 0, 4, 3, 0, 4, 4, 4, 0, 0, 1, 2, 2, 1, 1, 2, 4, 1, 0, 4, 3, 0,
 4, 1, 4, 4, 4, 0, 0, 3],
            weights=[1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ),
        # 3: TfIdfVectorizer inputs=1 outputs=1
        _node(
            'TfIdfVectorizer',
            ['tok'],
            ['BQ'],
            name='BQ',
            domain='',
            max_gram_length=5,
            max_skip_count=4,
            min_gram_length=1,
            mode='TFIDF',
            ngram_counts=[0, 0, 14, 176, 256],
            ngram_indexes=[1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0,
 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1,
 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            pool_int64s=[0, 4, 2, 3, 2, 4, 3, 0, 3, 2, 4, 2, 4, 4, 0, 0, 0, 0, 0, 1, 0, 0, 2, 0, 0, 3, 0, 0, 4, 0, 1, 1, 0,
 1, 2, 0, 1, 4, 0, 2, 0, 0, 2, 1, 0, 2, 3, 0, 3, 1, 0, 3, 2, 0, 3, 4, 0, 4, 1, 0, 4, 2, 1, 0, 0, 1,
 0, 1, 1, 0, 3, 1, 0, 4, 1, 1, 0, 1, 1, 2, 1, 2, 0, 1, 2, 1, 1, 4, 0, 1, 4, 1, 2, 0, 0, 2, 0, 3, 2,
 0, 4, 2, 1, 0, 2, 1, 4, 2, 2, 0, 2, 2, 1, 2, 2, 4, 2, 3, 2, 2, 3, 4, 2, 4, 2, 3, 0, 0, 3, 0, 2, 3,
 0, 4, 3, 1, 1, 3, 1, 3, 3, 1, 4, 3, 2, 3, 3, 3, 1, 3, 3, 4, 3, 4, 0, 3, 4, 4, 4, 1, 0, 4, 1, 4, 4,
 2, 0, 4, 2, 1, 4, 3, 1, 4, 3, 3, 0, 0, 0, 2, 0, 4, 1, 4, 0, 4, 2, 1, 1, 0, 2, 1, 1, 0, 4, 1, 1, 1,
 0, 4, 1, 1, 2, 1, 1, 4, 0, 3, 2, 0, 3, 4, 2, 3, 2, 3, 2, 3, 4, 0, 2, 4, 1, 0, 3, 0, 4, 1, 3, 1, 4,
 1, 3, 3, 1, 4, 3, 3, 4, 1, 4, 0, 3, 2, 4, 1, 0, 4, 4, 1, 1, 2, 4, 2, 1, 0, 1, 1, 0, 4, 1, 1, 4, 0,
 3, 2, 2, 4, 1, 0, 4, 4, 1, 1, 2, 1],
            weights=[1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ),
        # 4: Einsum inputs=62 outputs=1
        _node(
            'Einsum',
            ['input', 'U', 'U', 'U', 'D', 'C', 'C', 'D', 'C', 'C', 'U', 'U', 'U', 'D', 'C', 'BQ', 'D', 'C',
             'BQ', 'D', 'C', 'BQ', 'C', 'D', 'C', 'BQ', 'C', 'U', 'U', 'U', 'M', 'X', 'BR', 'M', 'X', 'BR', 'M',
             'X', 'BR', 'M', 'X', 'BR', 'M', 'X', 'BS', 'M', 'X', 'BS', 'M', 'X', 'BS', 'M', 'X', 'BS', 'D',
             'C', 'BQ', 'C', 'D', 'C', 'BQ', 'C'],
            ['output'],
            name='integer_move_swap',
            domain='',
            equation='narw,hb,hq,hq,bij,ia,jc,bkl,ka,lc,md,me,mq,dop,oa,np,dtu,ta,nu,evx,vc,nx,ec,eyz,yc,nz,ec,Af,Ag,As,fBC,Br,nC,fDE,Dr,nE,fFG,Fr,nG,fHI,Hr,nI,gJK,Jw,nK,gLM,Lw,nM,gNO,Nw,nO,gPQ,Pw,nQ,sRS,Rc,nS,sc,sTU,Tc,nU,sc->ncrw',
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
            _tensor('logits', TensorProto.FLOAT, (1, 5), INIT_LOGITS_0),
            _tensor('X', TensorProto.FLOAT, (2, 30), INIT_X_1),
            _tensor('C', TensorProto.FLOAT, (2, 10), INIT_C_2),
            _tensor('D', TensorProto.FLOAT, (2, 2, 2), INIT_D_3),
            _tensor('M', TensorProto.FLOAT, (2, 2, 3), INIT_M_4),
            _tensor('U', TensorProto.FLOAT, (2, 2), INIT_U_5),
        ],
        value_info=[
            _vi('tok', TensorProto.INT32, [1, 5]),
            _vi('BR', TensorProto.FLOAT, [1, 3]),
            _vi('BS', TensorProto.FLOAT, [1, 3]),
            _vi('BQ', TensorProto.FLOAT, [1, 2]),
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
