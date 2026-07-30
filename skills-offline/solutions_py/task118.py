from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task118'
TASK_NUM = 118
KAGGLE = {'score': 19.876036, 'date': '2026-07-14'}
MEMORY_BYTES = 76
PARAMS = 92
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = ''
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task118_tied'
OPSETS = [('', 20)]


# coord: FLOAT[30, 2], 60 value(s)
INIT_COORD_0 = [0.475972443819046, 0.31335505843162537, 0.9208180904388428, 0.2886139154434204, 1.365663766860962, 0.26387277245521545,
 1.8105093240737915, 0.2391316145658493, 2.2553551197052, 0.21439047157764435, 2.7002005577087402, 0.1896493285894394,
 3.1450462341308594, 0.16490817070007324, 3.5898919105529785, 0.1401670277118683, 4.034737586975098,
 0.11542588472366333, 4.479583263397217, 0.09068474173545837, 4.924428939819336, 0.06594359874725342, 5.369274616241455,
 0.04120245575904846, 5.814120292663574, 0.016461282968521118, 6.258965969085693, -0.008279860019683838,
 6.7038116455078125, -0.033021003007888794, 7.148657321929932, -0.05776214599609375, 7.593502998352051,
 -0.0825032889842987, 8.038348197937012, -0.10724443197250366, 8.483193397521973, -0.13198557496070862,
 8.92803955078125, -0.15672671794891357, 9.372884750366211, -0.18146786093711853, 9.817730903625488,
 -0.20620903372764587, 10.26257610321045, -0.23095014691352844, 10.707422256469727, -0.2556913197040558,
 11.152267456054688, -0.28043249249458313, 11.597113609313965, -0.3051736056804657, 12.041958808898926,
 -0.32991477847099304, 12.486804962158203, -0.3546558916568756, 12.931650161743164, -0.37939706444740295,
 13.376496315002441, -0.4041381776332855]

# mode_diff: FLOAT[2, 2, 2], 8 value(s)
INIT_MODE_DIFF_1 = [0.1636635959148407, 2.9426703453063965, 0.0, 0.0, 2.072855234146118, -3.1485753059387207, -0.1636635959148407,
 -2.9426703453063965]

# color: FLOAT[10, 2], 20 value(s)
INIT_COLOR_2 = [1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0]

# prob: FLOAT[1, 4], 4 value(s)
INIT_PROB_3 = [0.0, 0.0, 0.0, 0.0]


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
            ['prob'],
            ['key_2d'],
            name='',
            domain='',
            dtype=6,
            sample_size=5,
            seed=2.0,
        ),
        # 1: TfIdfVectorizer inputs=1 outputs=1
        _node(
            'TfIdfVectorizer',
            ['key_2d'],
            ['r0_feature'],
            name='r0_feature',
            domain='',
            max_gram_length=5,
            max_skip_count=3,
            min_gram_length=1,
            mode='TFIDF',
            ngram_counts=[0, 2, 18, 111, 239],
            ngram_indexes=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1,
 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0,
 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
            pool_int64s=[1, 3, 0, 0, 0, 3, 1, 0, 2, 0, 2, 1, 2, 3, 3, 1, 3, 3, 0, 1, 1, 0, 2, 2, 0, 3, 0, 0, 3, 1, 0, 3, 3,
 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 2, 1, 2, 3, 1, 3, 0, 1, 3, 1, 1, 3, 2, 1, 3, 3, 2, 0, 1,
 2, 1, 1, 2, 1, 2, 2, 2, 2, 2, 3, 0, 2, 3, 1, 3, 0, 0, 3, 0, 1, 3, 0, 2, 3, 1, 0, 3, 1, 3, 3, 2, 0,
 3, 2, 2, 3, 3, 1, 3, 3, 2, 3, 3, 3, 0, 0, 3, 1, 0, 2, 2, 2, 0, 2, 3, 0, 0, 3, 1, 3, 0, 3, 2, 3, 1,
 1, 1, 0, 1, 1, 1, 2, 1, 1, 1, 3, 1, 1, 3, 1, 1, 3, 0, 1, 1, 3, 0, 3, 1, 3, 1, 1, 1, 3, 1, 2, 1, 3,
 1, 3, 1, 3, 3, 0, 2, 0, 1, 1, 2, 0, 3, 3, 2, 1, 1, 0, 2, 3, 1, 1, 2, 3, 1, 2, 3, 0, 1, 1, 3, 1, 1,
 1, 3, 1, 1, 2, 3, 1, 1, 3, 3, 1, 3, 0, 3, 1, 3, 1, 3, 1, 3, 2, 3, 2, 1, 1, 3, 2, 2, 2, 3, 3, 0, 2,
 3, 3, 1, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 3, 1, 3, 0, 2, 3, 0, 0, 1, 1, 3, 1, 2, 1, 3, 0, 1, 1,
 1, 3, 1, 1, 3, 1, 3, 2, 0, 3, 2, 0, 1, 1, 3, 2, 0, 2, 2, 3, 2, 3, 1, 1, 2, 3, 1, 0, 0, 3, 3, 1, 1,
 1, 0, 3, 1, 1, 1, 2, 3, 1, 2, 1, 3, 3, 1, 3, 0, 3, 3, 1, 3, 1, 3, 3, 2, 0, 3, 3, 3, 2, 1, 1, 0, 3,
 2, 1, 3, 0, 3, 2, 2, 2, 2, 3, 3, 1, 3, 2, 3, 3, 3, 0, 2],
            weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ),
        # 2: TfIdfVectorizer inputs=1 outputs=1
        _node(
            'TfIdfVectorizer',
            ['key_2d'],
            ['c0_feature'],
            name='c0_feature',
            domain='',
            max_gram_length=5,
            max_skip_count=1,
            min_gram_length=1,
            mode='TFIDF',
            ngram_counts=[0, 4, 20, 116, 248],
            ngram_indexes=[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1,
 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0,
 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0],
            pool_int64s=[0, 1, 2, 3, 0, 0, 0, 2, 2, 1, 2, 2, 2, 3, 3, 0, 3, 1, 3, 2, 0, 0, 3, 0, 2, 1, 0, 2, 2, 0, 3, 0, 0,
 3, 1, 0, 3, 2, 0, 3, 3, 1, 0, 0, 1, 0, 1, 1, 0, 2, 1, 1, 1, 1, 1, 3, 1, 2, 1, 1, 2, 3, 1, 3, 1, 1,
 3, 2, 1, 3, 3, 2, 0, 1, 2, 0, 2, 2, 0, 3, 2, 1, 3, 2, 2, 2, 2, 2, 3, 2, 3, 1, 3, 0, 0, 3, 0, 1, 3,
 0, 3, 3, 1, 2, 3, 2, 2, 3, 2, 3, 3, 3, 0, 3, 3, 3, 0, 0, 0, 0, 0, 0, 3, 1, 0, 1, 3, 1, 0, 2, 2, 2,
 0, 3, 1, 3, 0, 3, 2, 3, 1, 0, 0, 3, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 3, 1, 1, 2, 1, 3, 1, 3, 0, 1, 1,
 3, 0, 3, 1, 3, 1, 1, 1, 3, 1, 2, 1, 3, 1, 3, 1, 3, 3, 0, 2, 0, 2, 2, 2, 0, 3, 3, 2, 1, 1, 0, 2, 3,
 1, 1, 2, 3, 1, 2, 3, 0, 1, 1, 3, 1, 0, 0, 3, 1, 1, 3, 3, 1, 2, 1, 3, 1, 3, 0, 3, 1, 3, 1, 3, 1, 3,
 3, 3, 2, 1, 1, 3, 2, 1, 3, 3, 2, 3, 1, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 2, 3, 1, 2, 0, 3, 2, 3, 1, 1,
 1, 1, 1, 3, 1, 1, 3, 1, 2, 1, 2, 0, 1, 2, 1, 3, 0, 1, 1, 1, 3, 1, 1, 3, 1, 3, 2, 0, 3, 2, 0, 1, 1,
 3, 2, 0, 2, 2, 2, 3, 1, 0, 0, 3, 3, 1, 2, 1, 3, 3, 1, 3, 0, 3, 3, 1, 3, 1, 3, 3, 1, 3, 3, 0, 3, 2,
 1, 1, 0],
            weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ),
        # 3: TfIdfVectorizer inputs=1 outputs=1
        _node(
            'TfIdfVectorizer',
            ['key_2d'],
            ['r1_feature'],
            name='r1_feature',
            domain='',
            max_gram_length=5,
            max_skip_count=2,
            min_gram_length=1,
            mode='TFIDF',
            ngram_counts=[0, 2, 26, 137, 229],
            ngram_indexes=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1,
 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1],
            pool_int64s=[1, 2, 0, 0, 0, 1, 0, 2, 0, 3, 1, 1, 1, 2, 1, 3, 2, 0, 2, 2, 2, 3, 3, 0, 3, 2, 0, 0, 0, 0, 0, 3, 0,
 1, 2, 0, 1, 3, 0, 2, 1, 0, 2, 2, 0, 2, 3, 0, 3, 0, 0, 3, 1, 0, 3, 3, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1,
 1, 2, 1, 1, 3, 1, 2, 1, 1, 3, 1, 1, 3, 2, 1, 3, 3, 2, 0, 1, 2, 1, 2, 2, 1, 3, 2, 2, 3, 2, 3, 0, 2,
 3, 1, 3, 0, 0, 3, 0, 2, 3, 1, 0, 3, 1, 1, 3, 1, 2, 3, 1, 3, 3, 2, 2, 3, 2, 3, 3, 3, 0, 3, 3, 1, 3,
 3, 2, 3, 3, 3, 0, 0, 0, 0, 0, 0, 3, 1, 0, 2, 3, 0, 0, 3, 1, 3, 1, 0, 0, 3, 1, 1, 1, 0, 1, 1, 1, 1,
 1, 1, 1, 2, 1, 1, 3, 1, 1, 3, 0, 3, 1, 3, 1, 1, 1, 3, 1, 3, 2, 1, 1, 0, 2, 3, 0, 0, 2, 3, 1, 1, 3,
 1, 0, 0, 3, 1, 1, 1, 3, 1, 1, 2, 3, 1, 3, 0, 3, 1, 3, 1, 3, 1, 3, 2, 3, 3, 1, 3, 3, 3, 3, 0, 0, 0,
 0, 0, 0, 0, 0, 3, 1, 3, 0, 2, 3, 0, 0, 1, 3, 1, 1, 3, 1, 3, 2, 0, 3, 2, 3, 1, 1, 2, 3, 1, 0, 0, 3,
 3, 1, 1, 1, 0, 3, 3, 1, 3, 2],
            weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ),
        # 4: TfIdfVectorizer inputs=1 outputs=1
        _node(
            'TfIdfVectorizer',
            ['key_2d'],
            ['c1_feature'],
            name='c1_feature',
            domain='',
            max_gram_length=5,
            max_skip_count=1,
            min_gram_length=1,
            mode='TFIDF',
            ngram_counts=[0, 2, 22, 115, 211],
            ngram_indexes=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1,
 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1,
 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
            pool_int64s=[2, 3, 0, 0, 0, 2, 1, 2, 1, 3, 2, 0, 2, 1, 2, 2, 2, 3, 3, 2, 3, 3, 0, 1, 2, 0, 1, 3, 0, 2, 1, 0, 2,
 3, 0, 3, 2, 0, 3, 3, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 2, 1, 1, 2, 3, 1, 3, 2, 1, 3, 3, 2, 0,
 1, 2, 0, 3, 2, 1, 1, 2, 1, 2, 2, 1, 3, 2, 2, 3, 2, 3, 0, 2, 3, 1, 3, 0, 0, 3, 0, 2, 3, 0, 3, 3, 2,
 0, 3, 2, 1, 3, 2, 2, 3, 2, 3, 3, 3, 1, 3, 3, 3, 0, 0, 0, 0, 0, 0, 3, 1, 0, 2, 3, 0, 0, 3, 1, 3, 1,
 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 3, 1, 1, 3, 0, 1, 1, 3, 1, 1, 1, 3, 2, 0, 2, 0, 3, 3, 2, 1,
 1, 0, 2, 3, 0, 0, 2, 3, 1, 1, 3, 0, 1, 1, 3, 1, 0, 0, 3, 1, 1, 1, 3, 1, 1, 2, 3, 1, 3, 1, 3, 1, 3,
 2, 3, 2, 0, 3, 3, 2, 1, 1, 3, 3, 1, 3, 0, 0, 3, 1, 3, 0, 2, 3, 0, 0, 1, 3, 0, 1, 1, 1, 3, 1, 1, 3,
 1, 3, 2, 0, 3, 2, 3, 1, 1, 2, 3, 1, 0, 0, 3, 3, 1, 1, 1, 0, 3, 1, 3, 0, 3, 3, 1, 3, 1, 3, 3, 2, 0,
 3, 3, 3, 2, 1, 1, 0, 3, 3, 1, 3, 2],
            weights=[1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ),
        # 5: TfIdfVectorizer inputs=1 outputs=1
        _node(
            'TfIdfVectorizer',
            ['key_2d'],
            ['r2_feature'],
            name='r2_feature',
            domain='',
            max_gram_length=5,
            max_skip_count=1,
            min_gram_length=1,
            mode='TFIDF',
            ngram_counts=[0, 3, 33, 114, 146],
            ngram_indexes=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1],
            pool_int64s=[1, 2, 3, 0, 1, 0, 2, 0, 3, 1, 0, 1, 1, 1, 2, 1, 3, 2, 0, 2, 1, 2, 2, 2, 3, 3, 0, 3, 1, 3, 2, 3, 3,
 0, 0, 0, 0, 0, 3, 0, 1, 2, 0, 1, 3, 0, 2, 1, 0, 3, 0, 0, 3, 1, 0, 3, 3, 1, 0, 1, 1, 1, 0, 1, 1, 1,
 1, 1, 3, 1, 2, 3, 1, 3, 1, 1, 3, 3, 2, 0, 3, 2, 1, 1, 2, 1, 2, 2, 2, 3, 3, 0, 2, 3, 0, 3, 3, 1, 0,
 3, 1, 2, 3, 2, 0, 3, 2, 1, 3, 2, 2, 3, 3, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 3, 1, 1, 3, 0, 3, 2, 1,
 1, 0, 3, 1, 0, 0, 3, 2, 0, 3, 3, 2, 1, 1, 3, 1, 0, 0, 3, 3, 2, 0, 3, 3, 3, 2, 1, 1, 0],
            weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ),
        # 6: TfIdfVectorizer inputs=1 outputs=1
        _node(
            'TfIdfVectorizer',
            ['key_2d'],
            ['c3_feature'],
            name='c3_feature',
            domain='',
            max_gram_length=5,
            max_skip_count=0,
            min_gram_length=1,
            mode='TFIDF',
            ngram_counts=[0, 3, 25, 94, 126],
            ngram_indexes=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0],
            pool_int64s=[0, 1, 3, 0, 0, 0, 1, 0, 2, 0, 3, 1, 0, 1, 1, 1, 3, 2, 0, 2, 1, 2, 2, 3, 1, 0, 0, 0, 0, 0, 3, 0, 1,
 2, 0, 1, 3, 0, 3, 1, 0, 3, 2, 0, 3, 3, 1, 0, 0, 1, 1, 1, 1, 1, 3, 1, 2, 1, 1, 3, 1, 1, 3, 2, 1, 3,
 3, 2, 2, 3, 2, 3, 0, 3, 0, 1, 3, 0, 2, 3, 0, 3, 3, 1, 2, 3, 1, 3, 3, 2, 2, 3, 3, 1, 0, 0, 3, 1, 0,
 3, 1, 3, 1, 1, 1, 0, 1, 1, 1, 1, 1, 3, 1, 3, 1, 3, 2, 0, 2, 0, 3, 3, 3, 1, 3, 1, 0, 0, 3, 1, 3, 3,
 1, 3, 1, 3],
            weights=[1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ),
        # 7: TfIdfVectorizer inputs=1 outputs=1
        _node(
            'TfIdfVectorizer',
            ['key_2d'],
            ['Lvec'],
            name='Lvec',
            domain='',
            max_gram_length=5,
            max_skip_count=0,
            min_gram_length=1,
            mode='TFIDF',
            ngram_counts=[0, 4, 12, 33, 37],
            ngram_indexes=[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            pool_int64s=[0, 1, 2, 3, 0, 2, 0, 3, 2, 0, 3, 0, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 3, 1, 2, 1, 1, 3, 2, 2, 3, 3, 1,
 0, 0, 0, 0, 0, 0, 3, 1, 3, 1, 1, 3, 1, 2, 1, 3, 0, 1, 1, 2, 3, 1, 1, 2, 3, 1, 3, 1, 3, 3, 2, 1, 1,
 0],
            weights=[0.20000000298023224, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ),
        # 8: Einsum inputs=70 outputs=1
        _node(
            'Einsum',
            ['Lvec', 'mode_diff', 'Lvec', 'mode_diff', 'mode_diff', 'r0_feature', 'coord', 'r0_feature',
             'coord', 'mode_diff', 'mode_diff', 'c0_feature', 'coord', 'c0_feature', 'coord', 'Lvec',
             'mode_diff', 'Lvec', 'mode_diff', 'mode_diff', 'r1_feature', 'coord', 'r1_feature', 'coord',
             'mode_diff', 'mode_diff', 'c1_feature', 'coord', 'c1_feature', 'coord', 'Lvec', 'mode_diff',
             'Lvec', 'mode_diff', 'mode_diff', 'r2_feature', 'coord', 'r2_feature', 'coord', 'mode_diff',
             'mode_diff', 'c1_feature', 'coord', 'c1_feature', 'coord', 'Lvec', 'mode_diff', 'Lvec',
             'mode_diff', 'mode_diff', 'mode_diff', 'mode_diff', 'mode_diff', 'mode_diff', 'coord', 'mode_diff',
             'mode_diff', 'mode_diff', 'mode_diff', 'coord', 'mode_diff', 'mode_diff', 'c3_feature', 'coord',
             'c3_feature', 'coord', 'input', 'color', 'color', 'color'],
            ['output'],
            name='',
            domain='',
            equation='nb,abd,nd,bef,bgi,ne,hf,ng,hi,djk,dlm,nj,wk,nl,wm,nq,pqr,nr,qst,quv,ns,ht,nu,hv,rxy,rAB,nx,wy,nA,wB,nD,CDE,nE,DFG,DHI,nF,hG,nH,hI,EJK,ELM,nJ,wK,nL,wM,nO,NOP,nP,OQR,OST,YQQ,YQQ,YQQ,YQQ,hR,ZSS,ZSS,ZSS,ZSS,hT,PUV,PWX,nU,wV,nW,wX,nchw,cz,cz,oz->nohw',
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
            _tensor('coord', TensorProto.FLOAT, (30, 2), INIT_COORD_0),
            _tensor('mode_diff', TensorProto.FLOAT, (2, 2, 2), INIT_MODE_DIFF_1),
            _tensor('color', TensorProto.FLOAT, (10, 2), INIT_COLOR_2),
            _tensor('prob', TensorProto.FLOAT, (1, 4), INIT_PROB_3),
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
