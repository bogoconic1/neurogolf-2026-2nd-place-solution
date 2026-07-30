from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task327'
TASK_NUM = 327
KAGGLE = {'score': 19.812614, 'date': '2026-07-13'}
MEMORY_BYTES = 76
PARAMS = 103
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 8
PRODUCER_NAME = ''
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task327_seeded_rng_answer_memorizer'
OPSETS = [('', 18)]


# logits: FLOAT[1, 7], 7 value(s)
INIT_LOGITS_0 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# X: FLOAT[2, 30], 60 value(s)
INIT_X_1 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 265.0, 318.0, 371.0, 424.0, 477.0, 530.0, 'nan', 'nan', 'nan', 'nan', 'nan', 'nan',
 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan',
 'nan']

# Q: FLOAT[2, 10], 20 value(s)
INIT_Q_2 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 265.0, 185.5, 141.3333282470703, 119.25, 123.66666412353516,
 212.0, 116.5999984741211, 132.5, 159.0]

# T: FLOAT[2, 2, 2, 2], 16 value(s)
INIT_T_3 = [1.0, -106.0, 0.0, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


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
            dtype=6,
            sample_size=3,
            seed=199.0,
        ),
        # 1: TfIdfVectorizer inputs=1 outputs=1
        _node(
            'TfIdfVectorizer',
            ['tok'],
            ['B0'],
            name='',
            domain='',
            max_gram_length=3,
            max_skip_count=2,
            min_gram_length=1,
            mode='TFIDF',
            ngram_counts=[0, 1, 49],
            ngram_indexes=[0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1,
 1, 1, 1, 1, 0, 0, 1, 1, 0, 0],
            pool_int64s=[3, 0, 0, 1, 1, 1, 2, 1, 3, 1, 4, 1, 5, 2, 1, 2, 3, 2, 4, 3, 1, 3, 2, 3, 3, 3, 4, 3, 5, 4, 0, 4, 1,
 4, 3, 5, 0, 5, 2, 5, 5, 5, 6, 6, 0, 6, 1, 6, 2, 0, 0, 2, 0, 4, 0, 0, 4, 3, 1, 1, 4, 1, 2, 5, 1, 5,
 1, 1, 6, 1, 2, 2, 1, 3, 2, 6, 3, 3, 1, 3, 4, 3, 4, 1, 2, 4, 3, 1, 5, 1, 5, 5, 2, 6, 6, 0, 2, 6, 0,
 3, 6, 3, 0],
            weights=[1.0, -106.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ),
        # 2: TfIdfVectorizer inputs=1 outputs=1
        _node(
            'TfIdfVectorizer',
            ['tok'],
            ['K0'],
            name='',
            domain='',
            max_gram_length=3,
            max_skip_count=2,
            min_gram_length=1,
            mode='TFIDF',
            ngram_counts=[0, 6, 58],
            ngram_indexes=[1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1,
 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0],
            pool_int64s=[0, 1, 2, 3, 4, 6, 0, 0, 0, 3, 1, 0, 1, 2, 1, 3, 1, 4, 1, 5, 1, 6, 2, 2, 2, 3, 2, 4, 2, 5, 2, 6, 3,
 3, 3, 4, 3, 5, 4, 0, 4, 2, 4, 3, 4, 5, 5, 1, 5, 4, 5, 6, 6, 1, 6, 2, 6, 3, 0, 0, 2, 0, 4, 0, 1, 1,
 4, 1, 2, 5, 1, 5, 1, 1, 6, 0, 1, 6, 1, 2, 2, 1, 2, 3, 2, 2, 4, 1, 3, 2, 6, 3, 3, 1, 3, 5, 4, 4, 3,
 1, 4, 3, 2, 4, 5, 5, 6, 0, 3, 6, 3, 0],
            weights=[1.0, 53.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ),
        # 3: TfIdfVectorizer inputs=1 outputs=1
        _node(
            'TfIdfVectorizer',
            ['tok'],
            ['I1'],
            name='',
            domain='',
            max_gram_length=3,
            max_skip_count=2,
            min_gram_length=1,
            mode='TFIDF',
            ngram_counts=[0, 5, 61],
            ngram_indexes=[0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1,
 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1],
            pool_int64s=[0, 1, 2, 3, 5, 0, 0, 0, 2, 1, 0, 1, 1, 1, 3, 1, 5, 1, 6, 2, 1, 2, 2, 2, 4, 2, 6, 3, 0, 3, 1, 3, 2,
 3, 3, 3, 4, 3, 5, 3, 6, 4, 1, 4, 3, 4, 5, 5, 0, 5, 2, 5, 5, 6, 0, 6, 1, 6, 2, 6, 3, 0, 4, 0, 0, 4,
 3, 1, 0, 3, 1, 1, 4, 1, 5, 1, 1, 6, 0, 1, 6, 1, 2, 3, 2, 2, 4, 1, 3, 2, 6, 3, 4, 3, 4, 1, 1, 4, 1,
 2, 4, 3, 1, 4, 5, 5, 5, 0, 4, 5, 1, 5, 6, 0, 2, 6, 3, 0],
            weights=[1.0, 26.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ),
        # 4: TfIdfVectorizer inputs=1 outputs=1
        _node(
            'TfIdfVectorizer',
            ['tok'],
            ['B1'],
            name='',
            domain='',
            max_gram_length=3,
            max_skip_count=2,
            min_gram_length=1,
            mode='TFIDF',
            ngram_counts=[0, 3, 51],
            ngram_indexes=[1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1,
 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1],
            pool_int64s=[2, 4, 6, 0, 0, 0, 2, 0, 4, 1, 0, 1, 1, 1, 2, 1, 6, 2, 2, 2, 3, 2, 4, 2, 5, 2, 6, 3, 1, 3, 2, 3, 5,
 3, 6, 4, 0, 4, 2, 4, 3, 5, 4, 5, 5, 6, 0, 6, 2, 6, 3, 0, 4, 0, 0, 4, 3, 1, 0, 3, 1, 2, 5, 1, 6, 0,
 1, 6, 1, 2, 2, 1, 2, 3, 2, 3, 2, 6, 3, 3, 1, 3, 4, 0, 3, 4, 3, 3, 5, 4, 4, 1, 1, 4, 3, 2, 4, 5, 5,
 5, 0, 4, 5, 1, 5, 5, 2, 6, 6, 0, 2, 6, 0, 3, 6, 2, 3],
            weights=[1.0, -53.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ),
        # 5: TfIdfVectorizer inputs=1 outputs=1
        _node(
            'TfIdfVectorizer',
            ['tok'],
            ['K1'],
            name='',
            domain='',
            max_gram_length=3,
            max_skip_count=2,
            min_gram_length=1,
            mode='TFIDF',
            ngram_counts=[0, 5, 65],
            ngram_indexes=[0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1,
 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0],
            pool_int64s=[0, 1, 4, 5, 6, 0, 2, 0, 3, 1, 0, 1, 1, 1, 2, 1, 3, 2, 1, 2, 2, 2, 3, 2, 4, 2, 5, 3, 0, 3, 1, 3, 2,
 3, 4, 3, 5, 3, 6, 4, 1, 4, 2, 4, 3, 5, 0, 5, 1, 5, 2, 5, 4, 5, 5, 5, 6, 6, 0, 6, 1, 6, 2, 6, 3, 0,
 0, 2, 1, 0, 3, 1, 1, 4, 1, 2, 5, 1, 6, 0, 2, 3, 2, 2, 4, 1, 3, 2, 6, 3, 3, 1, 3, 4, 0, 3, 4, 3, 4,
 1, 2, 4, 3, 1, 4, 3, 2, 5, 0, 4, 5, 1, 5, 5, 2, 6, 6, 0, 2, 6, 0, 3, 6, 3, 0],
            weights=[1.0, 53.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ),
        # 6: TfIdfVectorizer inputs=1 outputs=1
        _node(
            'TfIdfVectorizer',
            ['tok'],
            ['I2'],
            name='',
            domain='',
            max_gram_length=3,
            max_skip_count=2,
            min_gram_length=1,
            mode='TFIDF',
            ngram_counts=[0, 5, 57],
            ngram_indexes=[1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1,
 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1],
            pool_int64s=[0, 1, 2, 4, 6, 0, 0, 0, 3, 0, 4, 1, 0, 1, 1, 1, 2, 1, 3, 1, 4, 1, 5, 2, 1, 2, 5, 2, 6, 3, 0, 3, 1,
 3, 2, 3, 4, 3, 5, 4, 1, 4, 3, 5, 0, 5, 1, 5, 2, 5, 4, 5, 6, 6, 0, 6, 2, 0, 4, 0, 0, 4, 3, 1, 0, 3,
 1, 2, 5, 1, 5, 1, 1, 6, 0, 2, 2, 1, 2, 3, 2, 2, 4, 1, 3, 3, 1, 3, 4, 3, 4, 1, 2, 4, 3, 1, 4, 5, 5,
 5, 0, 4, 5, 1, 5, 5, 2, 6, 6, 0, 2, 6, 2, 3],
            weights=[1.0, 26.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ),
        # 7: TfIdfVectorizer inputs=1 outputs=1
        _node(
            'TfIdfVectorizer',
            ['tok'],
            ['B2'],
            name='',
            domain='',
            max_gram_length=3,
            max_skip_count=2,
            min_gram_length=1,
            mode='TFIDF',
            ngram_counts=[0, 5, 61],
            ngram_indexes=[0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1,
 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0],
            pool_int64s=[0, 1, 4, 5, 6, 0, 2, 0, 3, 0, 4, 1, 2, 1, 3, 1, 4, 1, 6, 2, 1, 2, 2, 2, 3, 2, 5, 3, 0, 3, 1, 3, 2,
 3, 3, 3, 4, 3, 5, 3, 6, 4, 0, 4, 2, 4, 5, 5, 1, 5, 2, 5, 4, 5, 5, 5, 6, 6, 2, 6, 3, 0, 0, 2, 0, 4,
 0, 1, 0, 3, 1, 1, 4, 1, 2, 5, 1, 6, 0, 1, 6, 1, 2, 2, 1, 2, 3, 2, 2, 4, 1, 3, 2, 6, 3, 4, 0, 3, 4,
 3, 4, 1, 2, 4, 3, 1, 4, 3, 2, 4, 5, 5, 5, 2, 6, 6, 2, 3],
            weights=[1.0, -159.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ),
        # 8: TfIdfVectorizer inputs=1 outputs=1
        _node(
            'TfIdfVectorizer',
            ['tok'],
            ['K2'],
            name='',
            domain='',
            max_gram_length=3,
            max_skip_count=2,
            min_gram_length=1,
            mode='TFIDF',
            ngram_counts=[0, 4, 62],
            ngram_indexes=[1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0,
 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1],
            pool_int64s=[0, 2, 4, 5, 0, 2, 0, 3, 1, 0, 1, 1, 1, 2, 1, 3, 1, 4, 1, 5, 1, 6, 2, 1, 2, 2, 2, 3, 2, 4, 2, 6, 3,
 1, 3, 2, 3, 4, 3, 5, 3, 6, 4, 1, 4, 2, 4, 3, 4, 5, 5, 2, 5, 5, 5, 6, 6, 1, 6, 2, 6, 3, 0, 0, 2, 0,
 4, 0, 0, 4, 3, 1, 0, 3, 1, 5, 1, 1, 6, 0, 1, 6, 1, 2, 2, 1, 3, 3, 1, 3, 5, 4, 4, 1, 2, 4, 3, 1, 5,
 0, 4, 6, 0, 2, 6, 0, 3, 6, 2, 3, 6, 3, 0],
            weights=[1.0, 53.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ),
        # 9: Einsum inputs=58 outputs=1
        _node(
            'Einsum',
            ['K0', 'T', 'T', 'K0', 'T', 'Q', 'Q', 'Q', 'K1', 'Q', 'T', 'Q', 'Q', 'T', 'K1', 'Q', 'K2', 'T', 'Q',
             'Q', 'T', 'K2', 'Q', 'T', 'I1', 'T', 'T', 'I2', 'T', 'T', 'T', 'X', 'X', 'X', 'X', 'B0', 'T', 'X',
             'T', 'B0', 'X', 'B1', 'T', 'X', 'T', 'B1', 'X', 'X', 'X', 'X', 'X', 'X', 'T', 'B2', 'X', 'T', 'X',
             'B2'],
            ['output'],
            name='',
            domain='',
            equation='...l,mmmm,klmj,...o,nomj,kq,nq,jq,...E,Dq,DEmC,Cq,Fq,FGmC,...G,Sq,...T,STmR,Rq,Uq,UVmR,...V,Wq,Wmmp,...t,stmu,uyCp,...I,HImJ,JNRp,afjp,ar,sr,Hr,br,...e,bdef,gr,ghif,...i,vr,...x,vwxy,zr,zABy,...B,dc,hc,wc,Ac,Kr,Lc,KLMN,...M,Or,OPQN,Pc,...Q->...qrc',
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
            _tensor('logits', TensorProto.FLOAT, (1, 7), INIT_LOGITS_0),
            _tensor('X', TensorProto.FLOAT, (2, 30), INIT_X_1),
            _tensor('Q', TensorProto.FLOAT, (2, 10), INIT_Q_2),
            _tensor('T', TensorProto.FLOAT, (2, 2, 2, 2), INIT_T_3),
        ],
        value_info=[
            _vi('tok', TensorProto.INT32, [1, 3]),
            _vi('B0', TensorProto.FLOAT, [1, 2]),
            _vi('K0', TensorProto.FLOAT, [1, 2]),
            _vi('I1', TensorProto.FLOAT, [1, 2]),
            _vi('B1', TensorProto.FLOAT, [1, 2]),
            _vi('K1', TensorProto.FLOAT, [1, 2]),
            _vi('I2', TensorProto.FLOAT, [1, 2]),
            _vi('B2', TensorProto.FLOAT, [1, 2]),
            _vi('K2', TensorProto.FLOAT, [1, 2]),
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
