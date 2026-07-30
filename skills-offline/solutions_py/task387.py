from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task387'
TASK_NUM = 387
KAGGLE = {'score': 18.581635, 'date': '2026-07-14'}
MEMORY_BYTES = 490
PARAMS = 123
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 8
PRODUCER_NAME = 'task387_seeded_rng_tfidf_s4_weighted'
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task387_multinomial_tfidf_s4_seed20431'
OPSETS = [('', 21)]


# four_u8: INT8[], 1 value(s)
INIT_FOUR_U8_0 = [4]

# ch_base: FLOAT16[1, 10], 10 value(s)
INIT_CH_BASE_1 = [3.814697265625e-06, 0.0, 0.0, 0.0, 0.0, -3.814697265625e-06, 0.0, 0.0, 0.0, 0.0]

# pm_updates: FLOAT16[1, 2], 2 value(s)
INIT_PM_UPDATES_2 = [-0.01439666748046875, 0.01439666748046875]

# q10h: FLOAT16[1, 14], 14 value(s)
INIT_Q10H_3 = [2.0, 3.0, 2.0, -2.0, -3.0, -2.0, 4.0, 4.0, 4.0, 4.0, 0.0, 0.0, 0.0, 0.0]

# F0: FLOAT16[2, 2, 2], 8 value(s)
INIT_F0_4 = [-1.59765625, 1.9033203125, 0.14599609375, -0.2288818359375, 6.4140625, -11.9609375, -0.72119140625, 1.5302734375]

# F1: FLOAT16[2, 2], 4 value(s)
INIT_F1_5 = [-0.0034465789794921875, 0.0034465789794921875, -0.0016155242919921875, 0.00434112548828125]

# F2: FLOAT16[2, 2], 4 value(s)
INIT_F2_6 = [2642.0, -10568.0, 1081.0, 3220.0]

# feat_base: FLOAT16[2, 30], 60 value(s)
INIT_FEAT_BASE_7 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

# ch_mask: FLOAT16[2, 10], 20 value(s)
INIT_CH_MASK_8 = [0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]


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
            ['ch_base'],
            ['key4'],
            name='',
            domain='',
            dtype=6,
            sample_size=4,
            seed=20431.0,
        ),
        # 1: TfIdfVectorizer inputs=1 outputs=1
        _node(
            'TfIdfVectorizer',
            ['key4'],
            ['h3_f'],
            name='',
            domain='',
            max_gram_length=4,
            max_skip_count=4,
            min_gram_length=1,
            mode='TFIDF',
            ngram_counts=[0, 8, 140, 272],
            ngram_indexes=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            pool_int64s=[0, 1, 3, 5, 6, 7, 8, 9, 0, 3, 0, 4, 0, 6, 0, 9, 1, 1, 1, 4, 1, 8, 2, 0, 2, 1, 2, 2, 2, 5, 2, 6, 2,
 7, 2, 8, 2, 9, 3, 0, 3, 1, 3, 2, 3, 3, 3, 4, 3, 5, 3, 8, 3, 9, 4, 0, 4, 1, 4, 2, 4, 3, 4, 4, 4, 5,
 4, 6, 4, 7, 4, 8, 4, 9, 5, 1, 5, 3, 5, 4, 5, 5, 5, 6, 5, 7, 5, 8, 5, 9, 6, 0, 6, 2, 6, 5, 6, 6, 6,
 7, 6, 8, 7, 0, 7, 1, 7, 2, 7, 3, 7, 4, 7, 5, 7, 6, 7, 9, 8, 2, 8, 4, 8, 6, 8, 8, 9, 0, 9, 1, 9, 4,
 9, 6, 9, 7, 9, 8, 9, 9, 0, 0, 9, 0, 1, 4, 0, 2, 0, 0, 3, 9, 1, 2, 1, 1, 2, 2, 1, 3, 3, 1, 4, 2, 1,
 4, 3, 1, 6, 8, 2, 0, 0, 2, 0, 2, 2, 0, 4, 2, 5, 6, 2, 6, 4, 2, 9, 9, 3, 4, 3, 3, 8, 2, 3, 9, 0, 3,
 9, 1, 4, 3, 4, 4, 3, 6, 4, 3, 9, 4, 4, 6, 4, 6, 5, 4, 7, 7, 4, 8, 0, 5, 1, 4, 5, 9, 7, 6, 0, 4, 6,
 0, 6, 6, 2, 0, 6, 2, 5, 6, 4, 7, 6, 5, 8, 6, 7, 7, 6, 8, 2, 7, 1, 2, 7, 7, 4, 8, 2, 8, 9, 0, 4, 9,
 6, 0, 9, 9, 0, 9, 9, 8, 0, 1, 4, 2, 1, 6, 8, 2, 2, 0, 2, 0, 2, 6, 4, 7, 4, 3, 4, 3, 4, 6, 5, 8, 4,
 7, 7, 4, 5, 1, 4, 3, 6, 2, 0, 4, 7, 1, 2, 2, 9, 9, 0, 4],
            weights=[-1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125],
        ),
        # 2: Cast inputs=1 outputs=1
        _node(
            'Cast',
            ['h3_f'],
            ['h3'],
            name='',
            domain='',
            to=3,
        ),
        # 3: TfIdfVectorizer inputs=1 outputs=1
        _node(
            'TfIdfVectorizer',
            ['key4'],
            ['w3_f'],
            name='',
            domain='',
            max_gram_length=4,
            max_skip_count=4,
            min_gram_length=1,
            mode='TFIDF',
            ngram_counts=[0, 9, 141, 252],
            ngram_indexes=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            pool_int64s=[1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 0, 0, 1, 0, 2, 0, 3, 0, 4, 0, 6, 0, 9, 1, 1, 1, 2, 1, 3, 1, 4, 1, 6,
 1, 8, 2, 2, 2, 4, 2, 5, 2, 6, 2, 8, 3, 0, 3, 1, 3, 2, 3, 4, 3, 5, 3, 6, 3, 8, 3, 9, 4, 0, 4, 2, 4,
 3, 4, 5, 4, 6, 4, 7, 4, 8, 4, 9, 5, 1, 5, 3, 5, 4, 5, 5, 5, 6, 5, 8, 5, 9, 6, 0, 6, 2, 6, 4, 6, 5,
 6, 6, 6, 7, 6, 8, 6, 9, 7, 1, 7, 3, 7, 4, 7, 5, 7, 6, 7, 7, 7, 9, 8, 0, 8, 2, 8, 4, 8, 6, 8, 7, 8,
 8, 9, 0, 9, 1, 9, 6, 9, 7, 0, 0, 9, 0, 1, 4, 0, 2, 0, 0, 3, 9, 1, 3, 3, 1, 4, 2, 1, 4, 3, 1, 6, 8,
 2, 0, 0, 2, 0, 2, 2, 0, 4, 2, 6, 4, 2, 9, 9, 3, 8, 2, 3, 9, 0, 3, 9, 1, 4, 3, 9, 4, 4, 6, 4, 6, 2,
 4, 6, 5, 4, 7, 7, 4, 8, 0, 5, 9, 7, 6, 0, 6, 6, 2, 0, 6, 5, 8, 6, 7, 0, 6, 7, 7, 6, 8, 2, 7, 1, 3,
 8, 2, 8, 8, 6, 7, 9, 0, 4, 9, 6, 0, 9, 7, 9, 9, 9, 0, 9, 9, 8, 0, 1, 4, 2, 0, 3, 9, 1, 2, 0, 0, 9,
 2, 0, 2, 0, 2, 9, 9, 8, 4, 3, 9, 0, 5, 9, 7, 9, 7, 1, 3, 3, 9, 6, 0, 6],
            weights=[-1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125, -1.125],
        ),
        # 4: Cast inputs=1 outputs=1
        _node(
            'Cast',
            ['w3_f'],
            ['w3'],
            name='',
            domain='',
            to=3,
        ),
        # 5: TfIdfVectorizer inputs=1 outputs=1
        _node(
            'TfIdfVectorizer',
            ['key4'],
            ['v54_f'],
            name='',
            domain='',
            max_gram_length=4,
            max_skip_count=4,
            min_gram_length=1,
            mode='TFIDF',
            ngram_counts=[0, 3, 51, 111],
            ngram_indexes=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            pool_int64s=[1, 2, 7, 0, 4, 0, 6, 1, 6, 1, 8, 2, 9, 3, 0, 3, 4, 3, 5, 4, 1, 4, 8, 4, 9, 5, 1, 5, 3, 5, 4, 5, 8,
 6, 7, 6, 9, 7, 0, 7, 6, 7, 9, 8, 1, 8, 4, 9, 7, 9, 8, 0, 0, 9, 0, 1, 4, 0, 2, 0, 1, 3, 3, 1, 4, 2,
 1, 6, 8, 2, 0, 2, 2, 0, 4, 2, 5, 6, 2, 9, 9, 3, 4, 3, 4, 8, 0, 4, 8, 1, 5, 4, 8, 5, 9, 7, 6, 0, 4,
 6, 7, 7, 6, 8, 2, 7, 6, 0, 7, 7, 7, 1, 6, 8, 2, 5, 4, 8, 1],
            weights=[1.125, 1.125, 1.125, 1.125, 1.125, 1.125, 1.125, 1.125, 1.125, 1.125, 1.125, 1.125, 1.125, 1.125, 1.125, 1.125, 1.125, 1.125, 1.125, 1.125, 1.125, 1.125, 1.125, 1.125, 1.125, 1.125, 1.125, 1.125, 1.125, 1.125, 1.125, 1.125, 1.125, 1.125, 1.125, 1.125, 1.125, 1.125, 1.125, 1.125, 1.125, 1.125, 1.125, 1.125, 1.125, 1.125, 1.125, 1.125, 1.125],
        ),
        # 6: Cast inputs=1 outputs=1
        _node(
            'Cast',
            ['v54_f'],
            ['v54'],
            name='',
            domain='',
            to=3,
        ),
        # 7: TfIdfVectorizer inputs=1 outputs=1
        _node(
            'TfIdfVectorizer',
            ['key4'],
            ['v19_f'],
            name='',
            domain='',
            max_gram_length=4,
            max_skip_count=4,
            min_gram_length=1,
            mode='TFIDF',
            ngram_counts=[0, 8, 96, 228],
            ngram_indexes=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
 0, 0, 0, 0, 0, 0, 0],
            pool_int64s=[0, 1, 2, 3, 6, 7, 8, 9, 0, 0, 0, 2, 0, 3, 0, 6, 1, 2, 1, 3, 2, 0, 2, 2, 2, 4, 2, 5, 2, 6, 2, 7, 3,
 0, 3, 1, 3, 2, 3, 3, 3, 4, 3, 5, 3, 8, 3, 9, 4, 1, 4, 5, 4, 6, 4, 7, 4, 9, 5, 3, 5, 5, 6, 9, 7, 1,
 7, 2, 7, 3, 7, 5, 7, 6, 7, 9, 8, 1, 8, 4, 8, 6, 8, 7, 8, 8, 9, 0, 9, 1, 9, 4, 9, 6, 9, 7, 0, 0, 9,
 0, 1, 4, 0, 2, 0, 0, 3, 9, 1, 3, 3, 1, 4, 2, 1, 4, 3, 2, 0, 0, 2, 5, 6, 2, 5, 7, 2, 6, 4, 2, 9, 9,
 3, 4, 3, 3, 5, 6, 3, 8, 2, 3, 9, 0, 3, 9, 1, 4, 3, 4, 4, 3, 6, 4, 3, 9, 4, 4, 6, 4, 6, 2, 4, 6, 5,
 4, 7, 7, 4, 8, 0, 4, 8, 1, 5, 6, 9, 5, 7, 5, 5, 9, 7, 6, 0, 4, 6, 0, 6, 6, 2, 5, 6, 4, 7, 6, 5, 8,
 6, 7, 0, 6, 7, 7, 7, 1, 2, 7, 6, 0, 7, 7, 4, 7, 7, 7, 8, 0, 4, 8, 6, 7, 9, 0, 4, 9, 9, 0, 0, 1, 4,
 2, 0, 3, 9, 1, 1, 4, 3, 6, 2, 5, 7, 5, 2, 6, 4, 7, 4, 3, 4, 3, 4, 8, 0, 4, 7, 6, 0, 4, 8, 6, 7, 0,
 9, 9, 0, 4],
            weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ),
        # 8: Cast inputs=1 outputs=1
        _node(
            'Cast',
            ['v19_f'],
            ['v19'],
            name='',
            domain='',
            to=3,
        ),
        # 9: TfIdfVectorizer inputs=1 outputs=1
        _node(
            'TfIdfVectorizer',
            ['key4'],
            ['v31_f'],
            name='',
            domain='',
            max_gram_length=4,
            max_skip_count=4,
            min_gram_length=1,
            mode='TFIDF',
            ngram_counts=[0, 1, 51, 129],
            ngram_indexes=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            pool_int64s=[5, 0, 0, 1, 1, 1, 8, 2, 1, 2, 2, 2, 5, 2, 8, 2, 9, 3, 2, 3, 4, 3, 5, 4, 1, 4, 4, 5, 3, 5, 5, 5, 6,
 6, 0, 6, 4, 6, 9, 7, 0, 7, 4, 7, 5, 7, 6, 8, 1, 9, 8, 0, 2, 0, 1, 2, 1, 1, 2, 2, 1, 3, 3, 1, 6, 8,
 2, 0, 2, 2, 0, 4, 2, 5, 7, 2, 6, 4, 2, 9, 9, 3, 4, 3, 3, 5, 6, 4, 3, 4, 4, 3, 6, 4, 8, 1, 5, 1, 4,
 5, 4, 8, 5, 6, 9, 5, 7, 5, 6, 0, 4, 6, 2, 0, 7, 1, 3, 7, 6, 0, 9, 0, 4, 9, 9, 0, 9, 9, 8, 1, 4, 3,
 6, 2, 0, 2, 0, 2, 5, 7, 5, 2, 9, 9, 8, 3, 5, 6, 9, 4, 3, 4, 3, 5, 1, 4, 3, 5, 4, 8, 1, 7, 6, 0, 4,
 9, 9, 0, 4, 0, 3, 9, 1, 6, 7, 7, 7, 4, 3, 9, 0, 0, 1, 4, 2],
            weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ),
        # 10: Cast inputs=1 outputs=1
        _node(
            'Cast',
            ['v31_f'],
            ['v67'],
            name='',
            domain='',
            to=3,
        ),
        # 11: TfIdfVectorizer inputs=1 outputs=1
        _node(
            'TfIdfVectorizer',
            ['key4'],
            ['v25_f'],
            name='',
            domain='',
            max_gram_length=4,
            max_skip_count=4,
            min_gram_length=1,
            mode='TFIDF',
            ngram_counts=[0, 8, 124, 223],
            ngram_indexes=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
 0, 0, 0, 0, 0, 0, 0, 0],
            pool_int64s=[0, 1, 2, 3, 6, 7, 8, 9, 0, 0, 0, 1, 0, 2, 0, 3, 0, 4, 0, 6, 1, 1, 1, 2, 1, 3, 1, 4, 1, 6, 1, 8, 2,
 0, 2, 1, 2, 2, 2, 5, 2, 6, 2, 7, 3, 0, 3, 1, 3, 2, 3, 3, 3, 4, 3, 6, 4, 0, 4, 1, 4, 2, 4, 3, 4, 4,
 4, 5, 4, 7, 4, 8, 4, 9, 5, 1, 5, 3, 5, 6, 5, 7, 5, 9, 6, 4, 6, 6, 6, 7, 7, 1, 7, 2, 7, 4, 7, 6, 7,
 7, 7, 9, 8, 0, 8, 1, 8, 4, 8, 6, 8, 7, 8, 8, 9, 0, 9, 1, 9, 6, 9, 7, 9, 9, 0, 1, 4, 0, 3, 9, 1, 2,
 1, 1, 4, 3, 1, 6, 8, 2, 0, 4, 2, 5, 6, 2, 6, 4, 3, 4, 3, 3, 8, 2, 3, 9, 0, 3, 9, 1, 4, 3, 4, 4, 3,
 6, 4, 3, 9, 4, 4, 6, 4, 6, 2, 4, 6, 5, 4, 7, 7, 4, 8, 0, 4, 8, 1, 5, 1, 4, 5, 4, 8, 5, 9, 7, 6, 0,
 4, 6, 0, 6, 6, 2, 0, 6, 2, 5, 6, 4, 7, 6, 7, 7, 7, 7, 4, 8, 0, 4, 9, 7, 9, 1, 4, 3, 6, 2, 6, 4, 7,
 4, 3, 4, 3, 4, 3, 9, 0, 4, 4, 6, 2, 4, 7, 7, 4, 4, 8, 0, 4, 6, 2, 5, 6],
            weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ),
        # 12: Cast inputs=1 outputs=1
        _node(
            'Cast',
            ['v25_f'],
            ['v25'],
            name='',
            domain='',
            to=3,
        ),
        # 13: TfIdfVectorizer inputs=1 outputs=1
        _node(
            'TfIdfVectorizer',
            ['key4'],
            ['v48_f'],
            name='',
            domain='',
            max_gram_length=4,
            max_skip_count=4,
            min_gram_length=1,
            mode='TFIDF',
            ngram_counts=[0, 7, 125, 233],
            ngram_indexes=[1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0,
 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1,
 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1,
 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0],
            pool_int64s=[0, 1, 2, 5, 6, 8, 9, 0, 0, 0, 1, 0, 2, 0, 3, 0, 4, 0, 6, 0, 9, 1, 1, 1, 8, 2, 0, 2, 4, 2, 5, 2, 6,
 2, 7, 2, 8, 2, 9, 3, 0, 3, 1, 3, 2, 3, 3, 3, 4, 3, 5, 3, 6, 3, 8, 3, 9, 4, 0, 4, 1, 4, 4, 4, 7, 4,
 8, 4, 9, 5, 1, 5, 5, 5, 6, 5, 7, 5, 8, 5, 9, 6, 0, 6, 2, 6, 6, 7, 0, 7, 1, 7, 3, 7, 6, 7, 7, 7, 9,
 8, 1, 8, 2, 8, 4, 8, 6, 8, 7, 8, 8, 9, 0, 9, 1, 9, 4, 9, 6, 9, 7, 9, 8, 9, 9, 0, 0, 9, 0, 1, 4, 0,
 2, 0, 1, 3, 3, 1, 4, 2, 1, 6, 8, 2, 0, 0, 2, 0, 2, 2, 0, 4, 2, 5, 6, 2, 9, 9, 3, 4, 3, 3, 8, 2, 4,
 3, 4, 4, 3, 6, 4, 4, 6, 4, 6, 2, 4, 7, 7, 4, 8, 1, 5, 4, 8, 5, 9, 7, 6, 0, 4, 6, 0, 6, 6, 2, 0, 6,
 2, 5, 6, 7, 7, 6, 8, 2, 7, 1, 2, 7, 1, 3, 7, 6, 0, 7, 7, 7, 9, 0, 4, 9, 6, 0, 9, 7, 9, 9, 9, 0, 9,
 9, 8, 0, 1, 4, 2, 1, 6, 8, 2, 2, 0, 0, 9, 2, 0, 2, 0, 2, 9, 9, 8, 5, 4, 8, 1, 6, 2, 5, 6, 7, 1, 3,
 3, 7, 6, 0, 4, 9, 6, 0, 6, 9, 9, 0, 4],
            weights=[1.225000023841858, 1.225000023841858, 1.225000023841858, 1.225000023841858, 1.225000023841858, 1.225000023841858, 1.225000023841858, 1.225000023841858, 1.225000023841858, 1.225000023841858, 1.225000023841858, 1.225000023841858, 1.225000023841858, 1.225000023841858, 1.225000023841858, 1.225000023841858, 1.225000023841858, 1.225000023841858, 1.225000023841858, 1.225000023841858, 1.225000023841858, 1.225000023841858, 1.225000023841858, 1.225000023841858, 1.225000023841858, 1.225000023841858, 1.225000023841858, 1.225000023841858, 1.225000023841858, 1.225000023841858, 1.225000023841858, 1.225000023841858, 1.225000023841858, 1.225000023841858, 1.225000023841858, 1.225000023841858, 1.225000023841858, 1.225000023841858, 1.225000023841858, 1.225000023841858, 1.225000023841858, 1.225000023841858, 1.225000023841858, 1.225000023841858, 1.225000023841858, 1.225000023841858, 1.225000023841858, 1.225000023841858, 1.225000023841858, 1.225000023841858, 1.225000023841858, 1.225000023841858, 1.225000023841858, 1.225000023841858, 1.225000023841858, 1.225000023841858, 1.225000023841858, 1.225000023841858, 1.225000023841858, 1.225000023841858, 1.225000023841858, 1.225000023841858, 1.225000023841858, 1.225000023841858, 1.225000023841858, 1.225000023841858, 1.225000023841858, 1.225000023841858, 1.225000023841858, 1.225000023841858, 1.225000023841858, 1.225000023841858, 1.225000023841858, 1.225000023841858, 1.225000023841858, 1.225000023841858, 1.225000023841858, 1.225000023841858, 1.225000023841858, 1.225000023841858, 1.225000023841858, 1.225000023841858, 1.225000023841858, 1.225000023841858, 1.225000023841858, 1.225000023841858, 1.225000023841858, 1.225000023841858, 1.225000023841858, 1.225000023841858, 1.225000023841858, 1.225000023841858, 1.225000023841858, 1.225000023841858, 1.225000023841858, 1.225000023841858, 1.225000023841858, 1.225000023841858, 1.225000023841858, 1.225000023841858, 1.225000023841858, 1.225000023841858, 1.225000023841858, 1.225000023841858, 1.225000023841858, 1.225000023841858, 1.225000023841858, 1.225000023841858, 1.225000023841858, 1.225000023841858, 1.225000023841858, 1.225000023841858, 1.225000023841858],
        ),
        # 14: Cast inputs=1 outputs=1
        _node(
            'Cast',
            ['v48_f'],
            ['v49'],
            name='',
            domain='',
            to=6,
        ),
        # 15: Shrink inputs=1 outputs=1
        _node(
            'Shrink',
            ['h3'],
            ['h2'],
            name='',
            domain='',
            bias=-1.0,
            lambd=0.0,
        ),
        # 16: Shrink inputs=1 outputs=1
        _node(
            'Shrink',
            ['h2'],
            ['h1'],
            name='',
            domain='',
            bias=-1.0,
            lambd=0.0,
        ),
        # 17: Shrink inputs=1 outputs=1
        _node(
            'Shrink',
            ['h1'],
            ['hs'],
            name='',
            domain='',
            bias=-1.0,
            lambd=0.0,
        ),
        # 18: Shrink inputs=1 outputs=1
        _node(
            'Shrink',
            ['w3'],
            ['w2'],
            name='',
            domain='',
            bias=-1.0,
            lambd=0.0,
        ),
        # 19: Shrink inputs=1 outputs=1
        _node(
            'Shrink',
            ['w2'],
            ['w1'],
            name='',
            domain='',
            bias=-1.0,
            lambd=0.0,
        ),
        # 20: Shrink inputs=1 outputs=1
        _node(
            'Shrink',
            ['w1'],
            ['ws'],
            name='',
            domain='',
            bias=-1.0,
            lambd=0.0,
        ),
        # 21: Shrink inputs=1 outputs=1
        _node(
            'Shrink',
            ['v54'],
            ['v30'],
            name='',
            domain='',
            bias=1.0,
            lambd=0.0,
        ),
        # 22: Shrink inputs=1 outputs=1
        _node(
            'Shrink',
            ['v54'],
            ['v57'],
            name='',
            domain='',
            bias=-1.0,
            lambd=0.0,
        ),
        # 23: Shrink inputs=1 outputs=1
        _node(
            'Shrink',
            ['v57'],
            ['v58'],
            name='',
            domain='',
            bias=-1.0,
            lambd=0.0,
        ),
        # 24: Shrink inputs=1 outputs=1
        _node(
            'Shrink',
            ['v67'],
            ['v31'],
            name='',
            domain='',
            bias=1.0,
            lambd=0.0,
        ),
        # 25: Shrink inputs=1 outputs=1
        _node(
            'Shrink',
            ['v67'],
            ['v70'],
            name='',
            domain='',
            bias=-1.0,
            lambd=0.0,
        ),
        # 26: Shrink inputs=1 outputs=1
        _node(
            'Shrink',
            ['v70'],
            ['v71'],
            name='',
            domain='',
            bias=-1.0,
            lambd=0.0,
        ),
        # 27: Add inputs=2 outputs=1
        _node(
            'Add',
            ['v30', 'v19'],
            ['v27'],
            name='',
            domain='',
        ),
        # 28: Div inputs=2 outputs=1
        _node(
            'Div',
            ['v19', 'four_u8'],
            ['v51'],
            name='',
            domain='',
        ),
        # 29: Add inputs=2 outputs=1
        _node(
            'Add',
            ['v51', 'v51'],
            ['v52'],
            name='',
            domain='',
        ),
        # 30: Shrink inputs=1 outputs=1
        _node(
            'Shrink',
            ['v27'],
            ['v55'],
            name='',
            domain='',
            bias=1.0,
            lambd=0.0,
        ),
        # 31: Shrink inputs=1 outputs=1
        _node(
            'Shrink',
            ['v27'],
            ['v56'],
            name='',
            domain='',
            bias=-1.0,
            lambd=0.0,
        ),
        # 32: Shrink inputs=1 outputs=1
        _node(
            'Shrink',
            ['v56'],
            ['v53'],
            name='',
            domain='',
            bias=-1.0,
            lambd=0.0,
        ),
        # 33: Add inputs=2 outputs=1
        _node(
            'Add',
            ['v54', 'v52'],
            ['v59'],
            name='',
            domain='',
        ),
        # 34: Sub inputs=2 outputs=1
        _node(
            'Sub',
            ['v56', 'v52'],
            ['v60'],
            name='',
            domain='',
        ),
        # 35: Concat inputs=14 outputs=1
        _node(
            'Concat',
            ['v30', 'v54', 'v57', 'v27', 'v56', 'v53', 'v58', 'v55', 'v59', 'v60', 'hs', 'h1', 'h2', 'h3'],
            ['v61'],
            name='',
            domain='',
            axis=1,
        ),
        # 36: Cast inputs=1 outputs=1
        _node(
            'Cast',
            ['v61'],
            ['v62'],
            name='',
            domain='',
            to=6,
        ),
        # 37: ScatterElements inputs=3 outputs=1
        _node(
            'ScatterElements',
            ['feat_base', 'v62', 'q10h'],
            ['v77'],
            name='',
            domain='',
            axis=1,
        ),
        # 38: Add inputs=2 outputs=1
        _node(
            'Add',
            ['v31', 'v25'],
            ['v29'],
            name='',
            domain='',
        ),
        # 39: Div inputs=2 outputs=1
        _node(
            'Div',
            ['v25', 'four_u8'],
            ['v64'],
            name='',
            domain='',
        ),
        # 40: Add inputs=2 outputs=1
        _node(
            'Add',
            ['v64', 'v64'],
            ['v65'],
            name='',
            domain='',
        ),
        # 41: Shrink inputs=1 outputs=1
        _node(
            'Shrink',
            ['v29'],
            ['v68'],
            name='',
            domain='',
            bias=1.0,
            lambd=0.0,
        ),
        # 42: Shrink inputs=1 outputs=1
        _node(
            'Shrink',
            ['v29'],
            ['v69'],
            name='',
            domain='',
            bias=-1.0,
            lambd=0.0,
        ),
        # 43: Shrink inputs=1 outputs=1
        _node(
            'Shrink',
            ['v69'],
            ['v66'],
            name='',
            domain='',
            bias=-1.0,
            lambd=0.0,
        ),
        # 44: Add inputs=2 outputs=1
        _node(
            'Add',
            ['v67', 'v65'],
            ['v72'],
            name='',
            domain='',
        ),
        # 45: Sub inputs=2 outputs=1
        _node(
            'Sub',
            ['v69', 'v65'],
            ['v73'],
            name='',
            domain='',
        ),
        # 46: Concat inputs=14 outputs=1
        _node(
            'Concat',
            ['v31', 'v67', 'v70', 'v29', 'v69', 'v66', 'v71', 'v68', 'v72', 'v73', 'ws', 'w1', 'w2', 'w3'],
            ['v74'],
            name='',
            domain='',
            axis=1,
        ),
        # 47: Cast inputs=1 outputs=1
        _node(
            'Cast',
            ['v74'],
            ['v75'],
            name='',
            domain='',
            to=6,
        ),
        # 48: ScatterElements inputs=3 outputs=1
        _node(
            'ScatterElements',
            ['feat_base', 'v75', 'q10h'],
            ['v78'],
            name='',
            domain='',
            axis=1,
        ),
        # 49: ScatterElements inputs=3 outputs=1
        _node(
            'ScatterElements',
            ['ch_base', 'v49', 'pm_updates'],
            ['v50'],
            name='',
            domain='',
            axis=1,
        ),
        # 50: Einsum inputs=251 outputs=1
        _node(
            'Einsum',
            ['F0', 'F0', 'F0', 'F0', 'F0', 'F0', 'F0', 'F0', 'F0', 'F0', 'F0', 'F0', 'F0', 'F0', 'F0', 'F0',
             'F1', 'F1', 'F1', 'F1', 'v77', 'v77', 'v77', 'feat_base', 'v77', 'F0', 'F1', 'v77', 'F0', 'F0',
             'F0', 'F1', 'F1', 'F1', 'F1', 'F2', 'pm_updates', 'pm_updates', 'F2', 'pm_updates', 'pm_updates',
             'F2', 'pm_updates', 'pm_updates', 'F2', 'pm_updates', 'pm_updates', 'F2', 'pm_updates',
             'pm_updates', 'F2', 'pm_updates', 'pm_updates', 'F2', 'pm_updates', 'pm_updates', 'F2',
             'pm_updates', 'pm_updates', 'F2', 'pm_updates', 'pm_updates', 'F2', 'pm_updates', 'pm_updates',
             'F2', 'pm_updates', 'F2', 'pm_updates', 'F2', 'pm_updates', 'F2', 'pm_updates', 'F2', 'pm_updates',
             'F2', 'pm_updates', 'v77', 'F1', 'F1', 'F1', 'F1', 'v78', 'v78', 'v78', 'feat_base', 'v78', 'F0',
             'F1', 'v78', 'F0', 'F0', 'F0', 'F1', 'F1', 'F1', 'F1', 'F2', 'pm_updates', 'pm_updates', 'F2',
             'pm_updates', 'pm_updates', 'F2', 'pm_updates', 'pm_updates', 'F2', 'pm_updates', 'pm_updates',
             'F2', 'pm_updates', 'pm_updates', 'F2', 'pm_updates', 'pm_updates', 'F2', 'pm_updates',
             'pm_updates', 'F2', 'pm_updates', 'pm_updates', 'F2', 'pm_updates', 'pm_updates', 'F2',
             'pm_updates', 'pm_updates', 'F2', 'pm_updates', 'F2', 'pm_updates', 'F2', 'pm_updates', 'F2',
             'pm_updates', 'F2', 'pm_updates', 'F2', 'pm_updates', 'v78', 'v78', 'F0', 'F0', 'F1', 'F1', 'F1',
             'F1', 'F2', 'pm_updates', 'pm_updates', 'F2', 'pm_updates', 'pm_updates', 'F2', 'pm_updates',
             'pm_updates', 'F2', 'pm_updates', 'pm_updates', 'F2', 'pm_updates', 'pm_updates', 'F2',
             'pm_updates', 'pm_updates', 'F2', 'pm_updates', 'pm_updates', 'F2', 'pm_updates', 'pm_updates',
             'F2', 'pm_updates', 'pm_updates', 'F2', 'pm_updates', 'F2', 'pm_updates', 'F2', 'pm_updates', 'F2',
             'pm_updates', 'F2', 'pm_updates', 'F2', 'pm_updates', 'F2', 'pm_updates', 'v77', 'F0', 'F0', 'F1',
             'F1', 'F1', 'F1', 'F2', 'pm_updates', 'pm_updates', 'F2', 'pm_updates', 'pm_updates', 'F2',
             'pm_updates', 'pm_updates', 'F2', 'pm_updates', 'pm_updates', 'F2', 'pm_updates', 'pm_updates',
             'F2', 'pm_updates', 'pm_updates', 'F2', 'pm_updates', 'pm_updates', 'F2', 'pm_updates',
             'pm_updates', 'F2', 'pm_updates', 'pm_updates', 'F2', 'pm_updates', 'F2', 'pm_updates', 'F2',
             'pm_updates', 'F2', 'pm_updates', 'F2', 'pm_updates', 'F2', 'pm_updates', 'F2', 'pm_updates',
             'v78', 'F0', 'F0', 'F0', 'v77', 'v78', 'F2', 'v78', 'F1', 'v77', 'F2', 'v77', 'F1', 'ch_mask',
             'v50'],
            ['output'],
            name='',
            domain='',
            equation='ZbZ,ZbZ,ZbZ,ZbZ,Zbb,ZbZ,ZbZ,ZdZ,ZdZ,ZdZ,ZdZ,ZdZ,ZdZ,ZYd,bbY,Ydd,QR,Qo,UV,UA,br,or,Ar,mC,mr,KLg,Mg,gr,EFe,eGe,eeG,Ge,Ge,Ge,Ge,Ge,yz,yz,Ge,yz,yz,Ge,yz,yz,Ge,yz,yz,Ge,yz,yz,Ge,yz,yz,eG,yz,yz,eG,yz,yz,eG,yz,yz,eG,yz,yz,eG,yz,eG,yz,eG,yz,eG,yz,eG,yz,eG,yz,er,ST,Sx,WX,WB,dw,xw,Bw,nD,nw,NOh,Ph,hw,HIf,fJf,ffJ,Jf,Jf,Jf,Jf,Jf,yz,yz,Jf,yz,yz,Jf,yz,yz,Jf,yz,yz,Jf,yz,yz,Jf,yz,yz,fJ,yz,yz,fJ,yz,yz,fJ,yz,yz,fJ,yz,yz,fJ,yz,fJ,yz,fJ,yz,fJ,yz,fJ,yz,fJ,yz,fw,vw,vav,vva,av,av,av,av,av,yz,yz,av,yz,yz,av,yz,yz,av,yz,yz,av,yz,yz,av,yz,yz,va,yz,yz,va,yz,yz,va,yz,yz,va,yz,va,yz,va,yz,va,yz,va,yz,va,yz,va,yz,lr,lal,lla,al,al,al,al,al,yz,yz,al,yz,yz,al,yz,yz,al,yz,yz,al,yz,yz,al,yz,yz,la,yz,yz,la,yz,yz,la,yz,yz,la,yz,la,yz,la,yz,la,yz,la,yz,la,yz,la,yz,sw,pas,paq,pai,ir,uw,pu,tw,pt,kr,pk,jr,pj,pc,yc->ycrw',
        ),
    ]

    graph = helper.make_graph(
        nodes,
        GRAPH_NAME,
        [
            _vi('input', TensorProto.FLOAT, [1, 10, 30, 30]),
        ],
        [
            _vi('output', TensorProto.FLOAT16, [1, 10, 30, 30]),
        ],
        initializer=[
            _tensor('four_u8', TensorProto.INT8, (), INIT_FOUR_U8_0),
            _tensor('ch_base', TensorProto.FLOAT16, (1, 10), INIT_CH_BASE_1),
            _tensor('pm_updates', TensorProto.FLOAT16, (1, 2), INIT_PM_UPDATES_2),
            _tensor('q10h', TensorProto.FLOAT16, (1, 14), INIT_Q10H_3),
            _tensor('F0', TensorProto.FLOAT16, (2, 2, 2), INIT_F0_4),
            _tensor('F1', TensorProto.FLOAT16, (2, 2), INIT_F1_5),
            _tensor('F2', TensorProto.FLOAT16, (2, 2), INIT_F2_6),
            _tensor('feat_base', TensorProto.FLOAT16, (2, 30), INIT_FEAT_BASE_7),
            _tensor('ch_mask', TensorProto.FLOAT16, (2, 10), INIT_CH_MASK_8),
        ],
        value_info=[
            _vi('key4', TensorProto.INT32, [1, 4]),
            _vi('h3_f', TensorProto.FLOAT, [1, 1]),
            _vi('h3', TensorProto.INT8, [1, 1]),
            _vi('w3_f', TensorProto.FLOAT, [1, 1]),
            _vi('w3', TensorProto.INT8, [1, 1]),
            _vi('v54_f', TensorProto.FLOAT, [1, 1]),
            _vi('v54', TensorProto.INT8, [1, 1]),
            _vi('v19_f', TensorProto.FLOAT, [1, 1]),
            _vi('v19', TensorProto.INT8, [1, 1]),
            _vi('v31_f', TensorProto.FLOAT, [1, 1]),
            _vi('v67', TensorProto.INT8, [1, 1]),
            _vi('v25_f', TensorProto.FLOAT, [1, 1]),
            _vi('v25', TensorProto.INT8, [1, 1]),
            _vi('v48_f', TensorProto.FLOAT, [1, 2]),
            _vi('v49', TensorProto.INT32, [1, 2]),
            _vi('h2', TensorProto.INT8, [1, 1]),
            _vi('h1', TensorProto.INT8, [1, 1]),
            _vi('hs', TensorProto.INT8, [1, 1]),
            _vi('w2', TensorProto.INT8, [1, 1]),
            _vi('w1', TensorProto.INT8, [1, 1]),
            _vi('ws', TensorProto.INT8, [1, 1]),
            _vi('v30', TensorProto.INT8, [1, 1]),
            _vi('v57', TensorProto.INT8, [1, 1]),
            _vi('v58', TensorProto.INT8, [1, 1]),
            _vi('v31', TensorProto.INT8, [1, 1]),
            _vi('v70', TensorProto.INT8, [1, 1]),
            _vi('v71', TensorProto.INT8, [1, 1]),
            _vi('v27', TensorProto.INT8, [1, 1]),
            _vi('v51', TensorProto.INT8, [1, 1]),
            _vi('v52', TensorProto.INT8, [1, 1]),
            _vi('v55', TensorProto.INT8, [1, 1]),
            _vi('v56', TensorProto.INT8, [1, 1]),
            _vi('v53', TensorProto.INT8, [1, 1]),
            _vi('v59', TensorProto.INT8, [1, 1]),
            _vi('v60', TensorProto.INT8, [1, 1]),
            _vi('v61', TensorProto.INT8, [1, 14]),
            _vi('v62', TensorProto.INT32, [1, 14]),
            _vi('v77', TensorProto.FLOAT16, [2, 30]),
            _vi('v29', TensorProto.INT8, [1, 1]),
            _vi('v64', TensorProto.INT8, [1, 1]),
            _vi('v65', TensorProto.INT8, [1, 1]),
            _vi('v68', TensorProto.INT8, [1, 1]),
            _vi('v69', TensorProto.INT8, [1, 1]),
            _vi('v66', TensorProto.INT8, [1, 1]),
            _vi('v72', TensorProto.INT8, [1, 1]),
            _vi('v73', TensorProto.INT8, [1, 1]),
            _vi('v74', TensorProto.INT8, [1, 14]),
            _vi('v75', TensorProto.INT32, [1, 14]),
            _vi('v78', TensorProto.FLOAT16, [2, 30]),
            _vi('v50', TensorProto.FLOAT16, [1, 10]),
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
