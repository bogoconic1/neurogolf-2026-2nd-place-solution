from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task383'
TASK_NUM = 383
KAGGLE = {'score': 20.336561, 'date': '2026-07-15'}
MEMORY_BYTES = 64
PARAMS = 42
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 8
PRODUCER_NAME = 'task383_q2_removed_tfidf_lut_L6C4S6'
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task383_m64_p70_candidate'
OPSETS = [('', 13)]


# H: FLOAT[10, 2], 20 value(s)
INIT_H_0 = [1.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 5.0, 1.0, 6.0, 1.0, 7.0, 1.0, 8.0, 1.0, 9.0]

# K: FLOAT[2, 2, 2], 8 value(s)
INIT_K_1 = [1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0]

# logits: FLOAT[1, 4], 4 value(s)
INIT_LOGITS_2 = [0.0, 0.0, 0.0, 0.0]

# U: FLOAT[2], 2 value(s)
INIT_U_3 = [1.0, 3.0]

# T3: FLOAT[2, 2, 2], 8 value(s)
INIT_T3_4 = [0.0, 0.0, 0.0, 1.0, 0.10000000149011612, 1.0, 1.0, 0.0]


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
            ['tokens'],
            name='rng_tokens',
            domain='',
            dtype=6,
            sample_size=6,
            seed=6.0,
        ),
        # 1: TfIdfVectorizer inputs=1 outputs=1
        _node(
            'TfIdfVectorizer',
            ['tokens'],
            ['A'],
            name='A',
            domain='',
            max_gram_length=6,
            max_skip_count=3,
            min_gram_length=1,
            mode='TF',
            ngram_counts=[0, 0, 0, 84, 224, 314],
            ngram_indexes=[1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1,
 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1,
 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1],
            pool_int64s=[3, 3, 1, 3, 1, 0, 3, 0, 0, 3, 2, 3, 2, 1, 1, 3, 1, 1, 2, 2, 1, 0, 3, 3, 1, 2, 3, 1, 3, 0, 3, 0, 1,
 1, 0, 1, 1, 0, 3, 1, 3, 3, 0, 2, 3, 3, 3, 0, 3, 0, 3, 3, 3, 3, 2, 0, 1, 0, 1, 1, 1, 3, 2, 1, 2, 0,
 1, 2, 2, 2, 2, 0, 2, 2, 3, 2, 3, 3, 0, 2, 2, 2, 0, 2, 3, 3, 1, 0, 3, 1, 0, 2, 1, 0, 2, 0, 0, 1, 0,
 2, 3, 0, 2, 1, 0, 2, 1, 3, 2, 3, 2, 3, 3, 2, 3, 1, 2, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 0, 1, 3, 0, 0,
 1, 3, 0, 1, 3, 1, 1, 0, 1, 0, 1, 3, 1, 3, 0, 2, 3, 0, 1, 3, 1, 1, 0, 0, 3, 3, 0, 2, 1, 3, 3, 2, 3,
 3, 2, 0, 2, 3, 1, 3, 3, 1, 3, 2, 1, 2, 0, 0, 2, 0, 0, 3, 3, 1, 1, 2, 1, 1, 2, 2, 2, 2, 0, 3, 0, 3,
 3, 1, 2, 0, 3, 0, 0, 2, 2, 0, 2, 2, 0, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 2, 3, 3, 1, 0, 2, 3, 1,
 0, 2, 0, 0, 0, 1, 0, 2, 2, 3, 2, 3, 1, 3, 2, 3, 1, 1, 1, 1, 3, 0, 0, 3, 1, 1, 0, 1, 0, 0, 0, 1, 3,
 0, 0, 1, 3, 0, 3, 1, 1, 2, 2, 2, 2, 0, 3, 3, 2, 0, 3, 3, 1, 3, 2, 2, 3, 3, 2, 0, 3, 0, 1, 0, 2, 2,
 0, 2, 2, 2, 0, 2, 1, 1, 2, 0, 1, 2, 2, 0, 1, 2, 2, 3, 3, 1, 0, 2, 0, 2, 0, 3, 2, 1, 1, 2, 3, 2, 3,
 1, 1, 1, 3, 1, 2, 3, 0, 3, 1, 1, 3, 0, 0, 3, 1, 1, 0, 1, 3, 0, 1, 3, 0, 2, 3, 1, 1, 3, 3, 0, 3, 1,
 1, 3, 0, 3, 2, 3, 3, 3, 3, 0, 2, 0, 0, 0, 1, 3, 0, 1, 2, 0, 0, 3, 1, 0, 3, 2, 2, 3, 3, 2, 0, 3, 0,
 1, 3, 0, 2, 2, 0, 2, 1, 1, 2, 0, 1, 2, 2],
        ),
        # 2: TfIdfVectorizer inputs=1 outputs=1
        _node(
            'TfIdfVectorizer',
            ['tokens'],
            ['B'],
            name='B',
            domain='',
            max_gram_length=6,
            max_skip_count=3,
            min_gram_length=1,
            mode='TF',
            ngram_counts=[0, 0, 0, 93, 277, 407],
            ngram_indexes=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0,
 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1,
 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1,
 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
            pool_int64s=[3, 3, 1, 3, 1, 0, 0, 0, 0, 0, 0, 1, 0, 2, 1, 2, 1, 3, 2, 0, 3, 3, 2, 1, 2, 1, 1, 2, 3, 2, 0, 3, 3,
 1, 3, 1, 1, 2, 3, 2, 3, 0, 1, 1, 3, 0, 1, 3, 1, 0, 3, 1, 1, 1, 0, 2, 3, 3, 3, 0, 2, 1, 0, 0, 1, 1,
 0, 0, 3, 3, 1, 3, 1, 2, 0, 2, 0, 0, 1, 1, 2, 2, 2, 2, 3, 2, 2, 2, 2, 3, 2, 3, 3, 0, 0, 0, 1, 0, 3,
 0, 2, 2, 0, 3, 2, 0, 3, 2, 1, 3, 2, 1, 1, 3, 2, 3, 1, 0, 3, 3, 2, 3, 3, 2, 1, 1, 3, 1, 2, 3, 1, 2,
 3, 1, 2, 3, 0, 3, 1, 1, 3, 1, 3, 0, 0, 0, 1, 3, 0, 2, 1, 1, 1, 1, 1, 1, 3, 1, 1, 3, 3, 1, 3, 0, 2,
 3, 0, 2, 3, 1, 3, 3, 0, 3, 3, 0, 3, 3, 3, 0, 1, 3, 0, 1, 3, 1, 3, 0, 3, 2, 2, 1, 1, 2, 1, 1, 0, 1,
 1, 0, 0, 2, 0, 1, 1, 0, 0, 1, 3, 1, 3, 3, 2, 3, 3, 2, 0, 1, 2, 0, 0, 2, 0, 0, 3, 0, 0, 3, 1, 1, 1,
 2, 2, 0, 3, 2, 2, 3, 2, 2, 3, 2, 2, 3, 3, 3, 2, 3, 2, 2, 3, 2, 0, 2, 0, 3, 0, 0, 3, 0, 1, 2, 0, 2,
 1, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 2, 0, 0, 0, 1, 0, 0, 3, 2, 1, 1, 0, 3, 3, 2, 1, 3, 3, 2, 1, 1,
 1, 3, 1, 2, 3, 3, 1, 1, 3, 0, 1, 1, 3, 0, 0, 1, 1, 1, 3, 3, 0, 1, 3, 0, 2, 1, 3, 0, 2, 3, 1, 1, 3,
 3, 0, 1, 3, 3, 0, 3, 3, 3, 3, 0, 1, 3, 3, 0, 1, 3, 0, 0, 0, 1, 3, 0, 0, 1, 3, 0, 1, 1, 3, 3, 2, 1,
 3, 3, 2, 0, 1, 2, 0, 0, 3, 0, 3, 2, 2, 3, 3, 2, 2, 3, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 0, 2, 0, 3, 0,
 1, 0, 3, 0, 1, 3, 2, 0, 1, 2, 2, 3, 3, 1, 0, 2, 0, 2, 0, 3, 2, 1, 1, 0, 3, 3, 2, 1, 1, 1, 3, 1, 2,
 3, 0, 3, 1, 1, 3, 0, 0, 0, 1, 3, 0, 2, 3, 1, 1, 3, 3, 0, 3, 3, 3, 3, 0, 1, 3, 3, 3, 3, 3, 0, 2, 0,
 0, 0, 1, 3, 0, 1, 1, 3, 3, 2, 0, 1, 2, 0, 0, 3, 1, 0, 3, 2, 2, 3, 3, 2, 3, 2, 3, 2, 0, 2, 0, 3, 0,
 1, 3],
        ),
        # 3: TfIdfVectorizer inputs=1 outputs=1
        _node(
            'TfIdfVectorizer',
            ['tokens'],
            ['DH'],
            name='DH',
            domain='',
            max_gram_length=6,
            max_skip_count=3,
            min_gram_length=1,
            mode='TFIDF',
            ngram_counts=[0, 4, 20, 104, 228, 343],
            ngram_indexes=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1,
 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0,
 1, 1, 1, 0, 1, 1, 1, 1],
            pool_int64s=[3, 1, 0, 2, 1, 0, 0, 2, 2, 0, 1, 2, 3, 2, 0, 1, 2, 1, 1, 3, 3, 3, 1, 1, 0, 2, 3, 0, 0, 0, 1, 0, 0,
 3, 0, 0, 2, 1, 3, 2, 3, 2, 0, 3, 0, 3, 2, 2, 3, 1, 2, 3, 2, 3, 1, 1, 2, 2, 1, 3, 3, 2, 0, 3, 1, 0,
 1, 3, 3, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 2, 3, 3, 0, 3, 3, 3, 3, 2, 0, 1, 2, 2, 2, 2, 2, 0, 2,
 2, 3, 0, 2, 2, 3, 3, 1, 0, 0, 3, 0, 2, 2, 3, 2, 3, 3, 2, 3, 1, 2, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 0,
 1, 3, 0, 0, 1, 3, 0, 1, 3, 0, 1, 0, 3, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 3, 1, 3, 0, 2, 3, 0, 2, 3, 3,
 3, 3, 0, 3, 0, 1, 3, 1, 3, 0, 3, 3, 0, 3, 2, 3, 3, 3, 3, 3, 3, 0, 2, 0, 0, 1, 3, 1, 3, 3, 2, 2, 2,
 0, 3, 2, 0, 3, 3, 0, 3, 3, 1, 3, 2, 3, 2, 2, 3, 2, 0, 2, 0, 3, 0, 0, 3, 0, 1, 0, 2, 2, 0, 3, 3, 1,
 0, 2, 3, 1, 0, 2, 0, 2, 0, 3, 2, 1, 2, 3, 2, 3, 1, 3, 2, 3, 1, 1, 3, 1, 1, 3, 0, 0, 1, 3, 0, 1, 1,
 3, 0, 1, 0, 1, 1, 0, 1, 3, 3, 3, 3, 0, 1, 1, 1, 3, 0, 3, 1, 3, 0, 3, 2, 2, 1, 1, 0, 0, 3, 3, 3, 0,
 2, 0, 0, 0, 1, 3, 0, 0, 1, 3, 0, 1, 1, 3, 3, 2, 2, 2, 0, 3, 3, 0, 3, 2, 2, 3, 2, 3, 2, 3, 2, 3, 2,
 3, 2, 0, 2, 0, 3, 0, 1, 0, 3, 0, 1, 3, 3, 3, 1, 0, 2, 0, 2, 0, 3, 2, 1, 1, 2, 3, 2, 3, 1, 1, 0, 3,
 3, 2, 1, 1, 1, 3, 1, 2, 3, 0, 0, 1, 3, 0, 1, 0, 0, 1, 3, 0, 2, 3, 1, 1, 3, 0, 3, 2, 2, 2, 1, 1, 0,
 0, 2, 2, 0, 3, 3, 1, 0, 3, 2, 2, 3, 3, 2, 3, 2, 3, 2, 0, 2, 0, 3, 0, 1, 3],
            weights=[1.0, 0.8999999761581421, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ),
        # 4: TfIdfVectorizer inputs=1 outputs=1
        _node(
            'TfIdfVectorizer',
            ['tokens'],
            ['DW'],
            name='DW',
            domain='',
            max_gram_length=6,
            max_skip_count=3,
            min_gram_length=1,
            mode='TFIDF',
            ngram_counts=[0, 3, 21, 114, 274, 424],
            ngram_indexes=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1,
 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1,
 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1],
            pool_int64s=[3, 1, 2, 3, 3, 1, 0, 2, 0, 3, 0, 0, 0, 0, 1, 0, 3, 2, 3, 2, 2, 3, 3, 1, 0, 0, 1, 0, 1, 0, 3, 0, 2,
 0, 2, 1, 2, 1, 3, 3, 2, 3, 3, 2, 1, 2, 1, 1, 3, 1, 1, 0, 3, 3, 3, 3, 2, 2, 3, 0, 1, 1, 3, 3, 2, 0,
 1, 3, 0, 0, 1, 3, 1, 0, 0, 1, 3, 3, 3, 0, 3, 2, 0, 1, 0, 1, 1, 1, 3, 2, 1, 2, 0, 1, 1, 2, 2, 2, 2,
 2, 2, 0, 3, 2, 2, 2, 2, 3, 0, 2, 2, 2, 0, 2, 3, 3, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 2, 0, 3,
 0, 2, 2, 0, 3, 2, 3, 2, 1, 1, 2, 3, 2, 3, 3, 2, 3, 1, 2, 3, 1, 1, 0, 3, 3, 2, 3, 3, 2, 1, 1, 3, 0,
 0, 0, 1, 3, 0, 1, 3, 0, 1, 3, 0, 1, 0, 1, 0, 1, 3, 1, 3, 3, 0, 3, 3, 0, 3, 3, 3, 3, 0, 3, 3, 0, 1,
 3, 0, 3, 2, 3, 3, 3, 3, 2, 0, 1, 1, 1, 1, 1, 1, 1, 3, 3, 2, 3, 3, 2, 0, 2, 3, 1, 3, 3, 1, 3, 2, 3,
 1, 1, 2, 1, 1, 2, 2, 1, 2, 2, 2, 2, 2, 0, 3, 2, 0, 3, 3, 0, 3, 3, 1, 0, 3, 2, 2, 2, 3, 2, 0, 0, 2,
 2, 0, 2, 2, 0, 2, 2, 0, 2, 1, 0, 0, 0, 1, 0, 3, 0, 2, 1, 3, 2, 0, 3, 2, 1, 0, 3, 2, 1, 1, 2, 3, 2,
 3, 1, 3, 2, 3, 1, 1, 0, 3, 3, 2, 1, 3, 3, 2, 1, 1, 0, 1, 3, 0, 1, 1, 3, 0, 1, 0, 3, 1, 1, 0, 1, 1,
 1, 3, 3, 0, 1, 3, 3, 0, 3, 3, 3, 3, 0, 1, 3, 3, 0, 1, 3, 1, 3, 0, 3, 2, 0, 0, 1, 3, 0, 1, 1, 3, 3,
 2, 1, 3, 3, 2, 0, 2, 3, 1, 3, 2, 3, 1, 1, 2, 2, 1, 1, 2, 2, 2, 2, 2, 0, 3, 3, 2, 0, 3, 3, 1, 3, 2,
 2, 3, 3, 2, 3, 2, 3, 2, 0, 2, 2, 0, 2, 2, 2, 0, 2, 1, 1, 2, 0, 1, 2, 2, 0, 1, 2, 2, 2, 0, 3, 2, 1,
 1, 2, 3, 2, 3, 1, 1, 0, 3, 3, 2, 1, 1, 1, 3, 1, 2, 3, 0, 3, 1, 1, 3, 0, 0, 0, 1, 3, 0, 1, 0, 3, 1,
 1, 0, 1, 3, 1, 1, 3, 3, 0, 3, 3, 3, 3, 0, 1, 3, 1, 1, 3, 0, 3, 2, 0, 0, 0, 1, 3, 0, 1, 1, 3, 3, 2,
 0, 3, 1, 1, 2, 2, 2, 2, 2, 0, 3, 3, 1, 0, 2, 2, 0, 2, 1, 1, 2, 0, 1, 2, 2],
            weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ),
        # 5: Einsum inputs=3 outputs=1
        _node(
            'Einsum',
            ['A', 'B', 'K'],
            ['D'],
            name='diff_state',
            domain='',
            equation='...a,...b,cab->...c',
        ),
        # 6: Einsum inputs=67 outputs=1
        _node(
            'Einsum',
            ['B', 'K', 'K', 'K', 'D', 'DW', 'A', 'input', 'H', 'H', 'K', 'A', 'H', 'K', 'D', 'H', 'DW', 'K',
             'U', 'B', 'input', 'K', 'K', 'K', 'D', 'DH', 'A', 'input', 'H', 'H', 'K', 'B', 'DH', 'A', 'D', 'U',
             'K', 'K', 'H', 'B', 'H', 'input', 'K', 'H', 'A', 'H', 'B', 'K', 'K', 'U', 'D', 'H', 'K', 'H', 'A',
             'H', 'B', 'K', 'H', 'input', 'D', 'input', 'T3', 'T3', 'T3', 'K', 'K'],
            ['output'],
            name='fused',
            domain='',
            equation='...i,dgj,jOF,jOF,...j,...g,...j,...ebf,eg,eh,ghi,...q,ln,nop,...q,lo,...n,dnq,q,...p,...lbm,dux,xYG,xYG,...x,...u,...x,...stc,su,sv,uvw,...w,...B,...E,...E,E,dBE,BCD,zC,...D,zB,...zAc,HIJ,aI,...J,aL,...M,KLM,ydk,k,...N,aN,RST,PS,...T,PV,...W,UVW,PX,...Pbc,...X,...aZZ,NHK,HKX,XyU,yyr,yRr->...abc',
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
            _tensor('H', TensorProto.FLOAT, (10, 2), INIT_H_0),
            _tensor('K', TensorProto.FLOAT, (2, 2, 2), INIT_K_1),
            _tensor('logits', TensorProto.FLOAT, (1, 4), INIT_LOGITS_2),
            _tensor('U', TensorProto.FLOAT, (2,), INIT_U_3),
            _tensor('T3', TensorProto.FLOAT, (2, 2, 2), INIT_T3_4),
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
