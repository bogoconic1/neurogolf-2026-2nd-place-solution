from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task208'
TASK_NUM = 208
KAGGLE = {'score': 19.969562, 'date': '2026-07-14'}
MEMORY_BYTES = 60
PARAMS = 93
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = 'task208_seeded_rng_scaled_projective_k6'
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task208_seeded_rng_scaled_projective_k6'
OPSETS = [('', 21)]


# Pcoord: FLOAT[2, 30], 60 value(s)
INIT_PCOORD_0 = [-1.0, -4.843054082770415e-11, 1.0, 2.0, 3.0, 3.9995615482330322, 5.0, 6.000010013580322, 7.0, 7.999884605407715,
 8.999617576599121, 10.0, 10.999982833862305, 11.999512672424316, 12.999890327453613, 13.999015808105469,
 14.999704360961914, 16.0, 16.999773025512695, 17.998746871948242, 19.0, -188.99569702148438, -8.973052024841309,
 -111.07079315185547, 43.780921936035156, -190.99440002441406, -74.36836242675781, 90.90036010742188,
 -2.823763847351074, 244.5760498046875, 2.0, 2.0, 2.000000238418579, 2.0, 2.0, 1.9997159242630005, 2.0,
 1.9999791383743286, 2.0, 1.999983787536621, 1.99993896484375, 2.0, 2.0001304149627686, 1.999971628189087,
 2.000004768371582, 1.999933123588562, 2.0, 2.0, 2.0, 2.0, 2.0, -41.999656677246094, -79.99996185302734,
 -144.5672607421875, -133.2535858154297, 163.70103454589844, 125.71998596191406, -305.118896484375, 303.6840515136719,
 -10.165346145629883]

# Pcolor: FLOAT[2, 10], 20 value(s)
INIT_PCOLOR_1 = [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.000114440917969, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
 1.9994065761566162]

# D: FLOAT[2, 2, 2], 8 value(s)
INIT_D_2 = [-1.1590300346142612e-05, -3.2105644809377054e-09, 1.378769431958915e-09, 0.13153371214866638, 5.656230314343702e-06,
 0.500006377696991, -0.5000265836715698, 1.1955939349661548e-08]

# logits: FLOAT[1, 5], 5 value(s)
INIT_LOGITS_3 = [0.0, 0.0, 0.0, 0.0, 0.0]


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
            name='',
            domain='',
            dtype=6,
            sample_size=5,
            seed=27.0,
        ),
        # 1: TfIdfVectorizer inputs=1 outputs=1
        _node(
            'TfIdfVectorizer',
            ['tokens'],
            ['T'],
            name='',
            domain='',
            mode='TF',
            min_gram_length=1,
            max_gram_length=5,
            max_skip_count=4,
            ngram_counts=[0, 3, 19, 118, 202],
            ngram_indexes=[0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1,
 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1,
 0, 1, 0, 0, 0, 0, 0, 1, 1],
            pool_int64s=[0, 2, 3, 0, 1, 0, 2, 1, 1, 1, 4, 2, 0, 2, 3, 3, 1, 3, 4, 0, 1, 2, 0, 1, 3, 0, 2, 0, 0, 3, 4, 0, 4,
 2, 1, 0, 4, 1, 1, 4, 1, 2, 0, 1, 3, 0, 1, 3, 4, 1, 4, 1, 1, 4, 4, 2, 0, 1, 2, 1, 4, 2, 2, 1, 2, 3,
 3, 2, 4, 3, 2, 4, 4, 3, 0, 0, 3, 0, 3, 3, 0, 4, 3, 1, 1, 3, 1, 4, 3, 3, 1, 3, 3, 2, 3, 4, 4, 4, 1,
 4, 4, 2, 1, 4, 2, 3, 4, 3, 1, 4, 3, 4, 4, 4, 0, 4, 4, 1, 0, 1, 3, 4, 0, 3, 1, 1, 0, 3, 3, 0, 0, 3,
 3, 2, 0, 4, 1, 1, 0, 4, 2, 1, 1, 0, 3, 3, 1, 0, 4, 2, 1, 3, 0, 3, 2, 0, 0, 2, 2, 0, 1, 3, 2, 1, 3,
 0, 2, 1, 4, 4, 3, 1, 1, 2, 3, 1, 2, 3, 3, 1, 2, 4, 3, 3, 0, 4, 3, 3, 1, 2, 4, 1, 4, 3, 4, 2, 2, 1,
 4, 3, 4, 4, 0, 3, 1, 1, 2, 0, 3, 1, 2, 3, 1, 0, 3, 3, 2, 1, 0, 4, 2, 1, 2, 0, 1, 3, 4, 2, 1, 3, 0,
 3, 2, 1, 4, 4, 3, 3, 3, 1, 2, 4, 4, 1, 4, 3, 1, 4, 2, 2, 1, 3],
        ),
        # 2: TfIdfVectorizer inputs=1 outputs=1
        _node(
            'TfIdfVectorizer',
            ['tokens'],
            ['Bot'],
            name='',
            domain='',
            mode='TF',
            min_gram_length=1,
            max_gram_length=5,
            max_skip_count=4,
            ngram_counts=[0, 4, 32, 137, 217],
            ngram_indexes=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1,
 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1],
            pool_int64s=[0, 1, 2, 3, 0, 1, 0, 3, 0, 4, 1, 2, 1, 3, 2, 0, 2, 1, 2, 3, 2, 4, 3, 1, 3, 2, 3, 3, 4, 3, 4, 4, 0,
 1, 0, 0, 1, 3, 0, 2, 2, 0, 3, 3, 1, 0, 3, 1, 1, 4, 1, 2, 0, 1, 2, 4, 1, 3, 0, 1, 3, 2, 1, 3, 4, 1,
 4, 3, 1, 4, 4, 2, 0, 0, 2, 0, 1, 2, 0, 2, 2, 0, 3, 2, 1, 3, 2, 1, 4, 2, 3, 3, 3, 0, 0, 3, 0, 3, 3,
 1, 4, 3, 2, 1, 3, 2, 2, 3, 3, 0, 3, 3, 1, 3, 4, 2, 3, 4, 4, 4, 1, 4, 4, 2, 3, 4, 3, 4, 4, 4, 0, 4,
 4, 1, 4, 4, 3, 0, 0, 2, 2, 0, 1, 3, 4, 0, 4, 1, 1, 1, 0, 3, 3, 1, 2, 0, 3, 1, 4, 3, 1, 2, 0, 0, 2,
 2, 0, 1, 3, 2, 1, 3, 0, 2, 2, 1, 3, 3, 0, 0, 1, 3, 0, 2, 0, 3, 1, 1, 2, 3, 1, 2, 4, 3, 3, 0, 3, 3,
 3, 1, 0, 3, 3, 1, 2, 3, 4, 4, 0, 4, 1, 1, 4, 4, 3, 4, 4, 0, 3, 1, 1, 2, 1, 0, 3, 3, 2, 1, 2, 0, 3,
 3, 2, 0, 1, 3, 4, 3, 0, 0, 1, 0, 3, 0, 2, 0, 1, 3, 3, 1, 2, 4, 4, 2, 2, 1, 3, 4, 3, 4, 4, 0],
        ),
        # 3: TfIdfVectorizer inputs=1 outputs=1
        _node(
            'TfIdfVectorizer',
            ['tokens'],
            ['Left'],
            name='',
            domain='',
            mode='TF',
            min_gram_length=1,
            max_gram_length=5,
            max_skip_count=4,
            ngram_counts=[0, 3, 31, 118, 214],
            ngram_indexes=[1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0,
 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0],
            pool_int64s=[0, 1, 2, 0, 0, 0, 2, 0, 4, 1, 0, 1, 2, 1, 4, 2, 0, 2, 1, 2, 2, 2, 3, 3, 1, 4, 1, 4, 2, 4, 4, 0, 1,
 2, 0, 1, 3, 0, 1, 4, 0, 3, 3, 0, 3, 4, 0, 4, 1, 1, 0, 3, 1, 2, 0, 1, 2, 3, 1, 2, 4, 1, 4, 1, 1, 4,
 3, 1, 4, 4, 2, 0, 3, 2, 1, 3, 2, 1, 4, 2, 3, 3, 2, 4, 1, 2, 4, 2, 3, 0, 0, 3, 0, 4, 3, 1, 0, 3, 1,
 2, 3, 2, 4, 4, 1, 4, 4, 2, 1, 4, 2, 3, 4, 3, 1, 4, 4, 3, 0, 3, 1, 1, 0, 3, 1, 2, 0, 3, 3, 0, 1, 2,
 0, 3, 1, 2, 4, 2, 1, 2, 4, 4, 1, 3, 0, 3, 1, 4, 1, 1, 1, 4, 2, 4, 1, 4, 4, 3, 2, 0, 0, 2, 2, 0, 3,
 3, 2, 1, 3, 0, 2, 1, 4, 4, 3, 0, 2, 0, 3, 1, 0, 2, 3, 1, 1, 2, 3, 1, 2, 3, 3, 3, 0, 4, 3, 3, 1, 0,
 4, 1, 4, 2, 4, 1, 4, 3, 4, 2, 2, 1, 4, 4, 3, 1, 0, 3, 1, 2, 3, 0, 3, 3, 0, 4, 1, 1, 4, 3, 2, 1, 2,
 0, 3, 3, 1, 4, 4, 3, 1, 2, 0, 0, 2, 2, 2, 0, 1, 3, 4, 2, 1, 4, 1, 1, 2, 1, 4, 2, 4, 2, 1, 4, 4, 3,
 3, 0, 0, 1, 0, 3, 0, 2, 0, 1, 3, 1, 2, 4, 4, 4, 1, 4, 2, 3, 4, 2, 2, 1, 3],
        ),
        # 4: TfIdfVectorizer inputs=1 outputs=1
        _node(
            'TfIdfVectorizer',
            ['tokens'],
            ['Right'],
            name='',
            domain='',
            mode='TF',
            min_gram_length=1,
            max_gram_length=5,
            max_skip_count=4,
            ngram_counts=[0, 4, 42, 153, 233],
            ngram_indexes=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0,
 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
            pool_int64s=[0, 2, 3, 4, 0, 0, 0, 1, 0, 2, 0, 3, 0, 4, 1, 1, 1, 3, 1, 4, 2, 0, 2, 2, 2, 3, 2, 4, 3, 1, 3, 2, 3,
 3, 4, 0, 4, 1, 4, 3, 4, 4, 0, 1, 3, 0, 1, 4, 0, 2, 0, 0, 3, 1, 0, 3, 3, 0, 3, 4, 1, 0, 2, 1, 0, 3,
 1, 1, 4, 1, 2, 0, 1, 2, 3, 1, 3, 2, 1, 4, 0, 1, 4, 3, 1, 4, 4, 2, 0, 0, 2, 0, 1, 2, 0, 2, 2, 0, 3,
 2, 1, 4, 2, 4, 3, 2, 4, 4, 3, 0, 0, 3, 0, 2, 3, 0, 4, 3, 1, 0, 3, 1, 1, 3, 2, 1, 3, 3, 2, 4, 1, 0,
 4, 1, 1, 4, 2, 1, 4, 2, 4, 4, 3, 1, 4, 3, 3, 4, 3, 4, 4, 4, 0, 0, 0, 2, 2, 0, 1, 3, 4, 0, 2, 0, 1,
 0, 3, 1, 1, 0, 3, 3, 0, 1, 0, 3, 3, 1, 2, 0, 3, 1, 2, 4, 2, 1, 4, 1, 1, 1, 4, 2, 4, 1, 4, 4, 3, 2,
 0, 0, 2, 2, 0, 3, 3, 2, 1, 4, 2, 2, 1, 4, 4, 3, 1, 0, 2, 3, 3, 0, 4, 3, 3, 1, 0, 4, 2, 1, 4, 4, 4,
 3, 1, 0, 3, 1, 1, 2, 1, 0, 3, 3, 2, 1, 2, 0, 3, 3, 1, 2, 4, 2, 4, 1, 4, 4, 3, 1, 2, 0, 0, 2, 2, 2,
 1, 4, 1, 1, 2, 1, 4, 4, 3, 3, 0, 2, 0, 1, 3, 3, 1, 0, 2, 4, 2, 1, 4, 0],
        ),
        # 5: TfIdfVectorizer inputs=1 outputs=1
        _node(
            'TfIdfVectorizer',
            ['tokens'],
            ['Mark'],
            name='',
            domain='',
            mode='TF',
            min_gram_length=1,
            max_gram_length=5,
            max_skip_count=4,
            ngram_counts=[0, 1, 11, 83, 151],
            ngram_indexes=[0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1,
 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1],
            pool_int64s=[4, 0, 1, 0, 4, 1, 0, 3, 1, 4, 2, 0, 0, 2, 0, 3, 3, 0, 3, 4, 1, 0, 2, 1, 1, 4, 1, 2, 4, 1, 3, 0, 1,
 3, 2, 2, 0, 1, 2, 1, 3, 2, 1, 4, 2, 2, 1, 2, 3, 3, 2, 4, 3, 2, 4, 4, 3, 0, 3, 3, 1, 0, 3, 2, 1, 3,
 3, 0, 3, 3, 1, 4, 1, 1, 4, 1, 4, 4, 2, 2, 4, 4, 0, 0, 3, 1, 1, 1, 0, 3, 3, 1, 4, 2, 3, 1, 4, 2, 4,
 2, 0, 0, 2, 2, 1, 3, 0, 2, 1, 4, 4, 2, 2, 1, 3, 2, 4, 2, 4, 3, 1, 0, 2, 3, 1, 1, 2, 3, 3, 1, 0, 4,
 1, 4, 3, 4, 2, 1, 4, 4, 2, 2, 1, 4, 2, 4, 2, 4, 3, 4, 4, 0, 3, 1, 1, 2, 0, 3, 3, 0, 4, 0, 4, 1, 1,
 4, 1, 3, 2, 1, 3, 2, 1, 3, 0, 3, 3, 0, 0, 1, 0, 3, 3, 1, 0, 2, 4, 1, 4, 3, 1, 4, 2, 2, 1, 3, 4, 3,
 4, 4, 0],
        ),
        # 6: Einsum inputs=52 outputs=1
        _node(
            'Einsum',
            ['input', 'Pcoord', 'Pcoord', 'Pcoord', 'D', 'Pcoord', 'T', 'D', 'Pcoord', 'Bot', 'Pcoord',
             'Pcoord', 'Pcoord', 'D', 'Pcoord', 'T', 'D', 'Pcoord', 'Bot', 'Pcoord', 'Pcoord', 'Pcoord', 'D',
             'Pcoord', 'Left', 'D', 'Pcoord', 'Right', 'Pcoord', 'Pcoord', 'Pcoord', 'D', 'Pcoord', 'Left', 'D',
             'Pcoord', 'Right', 'D', 'Pcoord', 'D', 'D', 'Pcolor', 'Pcolor', 'D', 'Pcolor', 'Pcolor', 'D',
             'Pcolor', 'Mark', 'D', 'Pcolor', 'Mark'],
            ['output'],
            name='',
            domain='',
            equation='...ihw,rM,aM,aM,agj,gh,...j,akl,kh,...l,rN,bN,bN,bmn,mh,...n,bpq,ph,...q,rO,cO,cO,cst,sw,...t,cuv,uw,...v,rP,dP,dP,dxy,xw,...y,dzA,zw,...A,rJK,KL,Jef,eBC,Bo,Ci,eDE,Do,Ei,fFG,Fo,...G,fHI,Ho,...I->...ohw',
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
            _tensor('Pcoord', TensorProto.FLOAT, (2, 30), INIT_PCOORD_0),
            _tensor('Pcolor', TensorProto.FLOAT, (2, 10), INIT_PCOLOR_1),
            _tensor('D', TensorProto.FLOAT, (2, 2, 2), INIT_D_2),
            _tensor('logits', TensorProto.FLOAT, (1, 5), INIT_LOGITS_3),
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
