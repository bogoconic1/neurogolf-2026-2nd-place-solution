from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task251'
TASK_NUM = 251
KAGGLE = {'score': 19.900134, 'date': '2026-07-15'}
MEMORY_BYTES = 70
PARAMS = 94
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = ''
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task251_rng_tfidf_gated'
OPSETS = [('', 20)]


# P: FLOAT[2, 30], 60 value(s)
INIT_P_0 = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, -0.0393691211938858, -2.023275136947632,
 -1.622431433645488e-08, -10.32257080078125, -14.807141304016113, -0.00016331773076672107, -0.42754003405570984,
 -1.3216841807661694e-06, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
 1.0, 1.0, 1.0, -0.0027839671820402145, -0.1833789050579071, -4.3955544981599814e-08, -0.6605810523033142,
 -1.986697793006897, -1.0759578799479641e-05, 1.0388308763504028, -1.4087436284171417e-05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0]

# V: FLOAT[2, 10], 20 value(s)
INIT_V_1 = [1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# T: FLOAT[2, 3, 2], 12 value(s)
INIT_T_2 = [0.0, -2.0, 0.0, -1.0, 2.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.5]

# zero: INT32[1], 1 value(s)
INIT_ZERO_3 = [0]

# LIKE: FLOAT16[1], 1 value(s)
INIT_LIKE_4 = [0.0]


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
        # 0: RandomUniformLike inputs=1 outputs=1
        _node(
            'RandomUniformLike',
            ['LIKE'],
            ['rnd'],
            name='',
            domain='',
            dtype=10,
            high=30000.0,
            low=100.0,
            seed=1.0,
        ),
        # 1: Cast inputs=1 outputs=1
        _node(
            'Cast',
            ['rnd'],
            ['key'],
            name='',
            domain='',
            to=6,
        ),
        # 2: Concat inputs=4 outputs=1
        _node(
            'Concat',
            ['zero', 'key', 'key', 'key'],
            ['tokens'],
            name='',
            domain='',
            axis=0,
        ),
        # 3: TfIdfVectorizer inputs=1 outputs=1
        _node(
            'TfIdfVectorizer',
            ['tokens'],
            ['ZR0'],
            name='',
            domain='',
            max_gram_length=4,
            max_skip_count=1,
            min_gram_length=1,
            mode='TFIDF',
            ngram_counts=[0, 9, 25, 46],
            ngram_indexes=[2, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1],
            pool_int64s=[0, 22692, 13813, 16029, 20398, 1698, 2098, 15855, 19652, 0, 22692, 22692, 22692, 0, 20398, 0, 1698,
 0, 2098, 2098, 2098, 0, 15855, 15855, 15855, 0, 22692, 22692, 0, 16029, 16029, 16029, 16029, 16029,
 0, 2098, 2098, 2098, 2098, 2098, 0, 19652, 19652, 19652, 19652, 19652],
            weights=[0.7899215221405029, 0.9236293435096741, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ),
        # 4: TfIdfVectorizer inputs=1 outputs=1
        _node(
            'TfIdfVectorizer',
            ['tokens'],
            ['ZC0'],
            name='',
            domain='',
            max_gram_length=4,
            max_skip_count=1,
            min_gram_length=1,
            mode='TFIDF',
            ngram_counts=[0, 9, 35, 74],
            ngram_indexes=[2, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1,
 0, 0, 1, 1],
            pool_int64s=[0, 22692, 13813, 16029, 20398, 1698, 2098, 15855, 19652, 0, 22692, 0, 13813, 0, 16029, 16029,
 16029, 0, 20398, 0, 1698, 1698, 1698, 0, 2098, 2098, 2098, 0, 15855, 15855, 15855, 0, 19652, 19652,
 19652, 0, 22692, 22692, 22692, 22692, 22692, 0, 13813, 13813, 13813, 13813, 13813, 0, 16029, 16029,
 16029, 16029, 16029, 0, 20398, 20398, 0, 1698, 1698, 1698, 1698, 1698, 0, 2098, 2098, 0, 15855,
 15855, 0, 19652, 19652, 19652, 19652, 19652, 0, 22692, 22692, 22692, 0, 19652, 19652, 19652],
            weights=[0.8401011228561401, 0.9594425559043884, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ),
        # 5: TfIdfVectorizer inputs=1 outputs=1
        _node(
            'TfIdfVectorizer',
            ['tokens'],
            ['ZR1'],
            name='',
            domain='',
            max_gram_length=4,
            max_skip_count=1,
            min_gram_length=1,
            mode='TFIDF',
            ngram_counts=[0, 22, 78, 186],
            ngram_indexes=[2, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1,
 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0,
 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1],
            pool_int64s=[0, 100, 4032, 22692, 13813, 20398, 20410, 28047, 15630, 1133, 1698, 20167, 11564, 2098, 12582,
 20634, 17710, 27920, 25400, 15855, 2849, 19652, 0, 4032, 0, 22692, 0, 13813, 0, 16029, 0, 6646, 0,
 1506, 0, 20398, 0, 20410, 0, 28047, 0, 11566, 0, 15630, 0, 24945, 0, 1133, 0, 1698, 0, 15938, 0,
 20167, 0, 330, 0, 11564, 11564, 11564, 0, 2098, 0, 12582, 0, 20634, 0, 17710, 0, 27920, 0, 25400,
 0, 15855, 0, 19652, 0, 12538, 0, 100, 100, 0, 4032, 4032, 0, 13813, 13813, 0, 16029, 16029, 16029,
 16029, 16029, 0, 6646, 6646, 0, 1506, 1506, 1506, 1506, 1506, 0, 20398, 20398, 0, 20410, 20410, 0,
 28047, 28047, 28047, 28047, 28047, 0, 11566, 11566, 11566, 11566, 11566, 0, 15630, 15630, 0, 24945,
 24945, 0, 1133, 1133, 1133, 1133, 1133, 0, 1698, 1698, 1698, 1698, 1698, 0, 15938, 15938, 0, 20167,
 20167, 20167, 20167, 20167, 0, 330, 330, 330, 330, 330, 0, 2098, 2098, 2098, 2098, 2098, 0, 20634,
 20634, 20634, 20634, 20634, 0, 27920, 27920, 27920, 27920, 27920, 0, 25400, 25400, 0, 15855, 15855,
 15855, 15855, 15855, 0, 19652, 19652, 19652, 19652, 19652, 0, 15855, 15855, 15855],
            weights=[1.015941858291626, 1.6671470403671265, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ),
        # 6: TfIdfVectorizer inputs=1 outputs=1
        _node(
            'TfIdfVectorizer',
            ['tokens'],
            ['ZC1'],
            name='',
            domain='',
            max_gram_length=4,
            max_skip_count=1,
            min_gram_length=1,
            mode='TFIDF',
            ngram_counts=[0, 30, 82, 193],
            ngram_indexes=[2, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1,
 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1,
 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1],
            pool_int64s=[0, 100, 4032, 22692, 13813, 16029, 6646, 1506, 20398, 20410, 28047, 11566, 15630, 24945, 1133,
 1698, 15938, 20167, 330, 11564, 2098, 12582, 20634, 17710, 27920, 25400, 15855, 2849, 19652, 12538,
 0, 4032, 0, 22692, 22692, 22692, 0, 6646, 6646, 6646, 0, 20398, 0, 28047, 0, 11566, 0, 15630, 0,
 24945, 0, 1133, 0, 15938, 0, 11564, 11564, 11564, 0, 2098, 2098, 2098, 0, 12582, 0, 27920, 27920,
 27920, 0, 25400, 25400, 25400, 0, 15855, 0, 2849, 0, 19652, 0, 12538, 12538, 12538, 0, 100, 100, 0,
 4032, 4032, 0, 22692, 22692, 22692, 22692, 22692, 0, 16029, 16029, 0, 6646, 6646, 6646, 6646, 6646,
 0, 1506, 1506, 0, 20398, 20398, 20398, 20398, 20398, 0, 20410, 20410, 0, 28047, 28047, 28047,
 28047, 28047, 0, 11566, 11566, 0, 15630, 15630, 0, 24945, 24945, 0, 1133, 1133, 1133, 1133, 1133,
 0, 1698, 1698, 1698, 1698, 1698, 0, 15938, 15938, 15938, 15938, 15938, 0, 20167, 20167, 0, 330,
 330, 0, 11564, 11564, 11564, 11564, 11564, 0, 2098, 2098, 2098, 2098, 2098, 0, 12582, 12582, 0,
 20634, 20634, 0, 17710, 17710, 0, 27920, 27920, 0, 25400, 25400, 25400, 25400, 25400, 0, 2849,
 2849, 0, 12538, 12538, 12538, 12538, 12538, 0, 1133, 1133, 1133, 0, 15938, 15938, 15938, 0, 25400,
 25400, 25400],
            weights=[0.951712965965271, 1.1507222652435303, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ),
        # 7: Einsum inputs=46 outputs=1
        _node(
            'Einsum',
            ['P', 'P', 'T', 'P', 'P', 'P', 'P', 'ZR0', 'P', 'T', 'ZR0', 'P', 'P', 'P', 'P', 'ZC0', 'T', 'V',
             'V', 'input', 'P', 'T', 'ZC0', 'P', 'P', 'P', 'V', 'P', 'P', 'P', 'P', 'ZR1', 'T', 'P', 'T', 'ZR1',
             'P', 'P', 'P', 'P', 'ZC1', 'T', 'P', 'T', 'ZC1', 'P'],
            ['output'],
            name='',
            domain='',
            equation='dN,aN,bef,dN,aO,bO,bO,e,fR,bgh,g,hR,aP,cP,cP,i,cij,dF,dF,nFRC,jC,ckl,k,lC,mQ,dQ,do,dQ,mS,pS,pS,r,prs,sR,ptu,t,uR,mT,qT,qT,v,qvw,wC,qxy,x,yC->noRC',
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
            _tensor('P', TensorProto.FLOAT, (2, 30), INIT_P_0),
            _tensor('V', TensorProto.FLOAT, (2, 10), INIT_V_1),
            _tensor('T', TensorProto.FLOAT, (2, 3, 2), INIT_T_2),
            _tensor('zero', TensorProto.INT32, (1,), INIT_ZERO_3),
            _tensor('LIKE', TensorProto.FLOAT16, (1,), INIT_LIKE_4),
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
