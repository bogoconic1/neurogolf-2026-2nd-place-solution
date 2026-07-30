from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task119'
TASK_NUM = 119
KAGGLE = {'score': 20.14797, 'date': '2026-07-15'}
MEMORY_BYTES = 36
PARAMS = 92
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = 'task119_sharedM_projective'
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task119_sharedM_projective_m36_p96'
OPSETS = [('', 14)]


# prob: FLOAT[1, 4], 4 value(s)
INIT_PROB_0 = [1.0, 1.0, 1.0, 1.0]

# P: FLOAT[2, 30], 60 value(s)
INIT_P_1 = [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 0.32429659366607666, 1.681910514831543,
 -0.2382236272096634, -7.650504112243652, 0.5676254034042358, -0.2395433634519577, -2.729562997817993,
 3.567715883255005, -0.14034707844257355, -0.320563942193985, -1.554932713508606, 0.3161238431930542,
 -4.413186550140381, -1.893680453300476, -3.669494152069092, 1.7397091388702393, -1.5517096519470215, -2.79563307762146,
 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.210275411605835, -2.1677212715148926,
 -0.9266166090965271, -0.13783708214759827, -0.05563170462846756, -0.5909659266471863, -0.23614895343780518,
 -3.7566027641296387, -1.1125760078430176, -1.068403959274292, -0.5476298928260803, -1.200142741203308,
 -0.3551996052265167, -0.43829211592674255, -0.3127945363521576, 3.8868305683135986, -0.4694088399410248,
 -0.30058300495147705]

# B: FLOAT[2, 2, 2], 8 value(s)
INIT_B_2 = [0.0, 1.0, -1.0, -0.02500000037252903, 0.0, 0.0, 0.0, 1.0]

# E: FLOAT[2, 10], 20 value(s)
INIT_E_3 = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -8.024999618530273, 0.0, -2.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


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
            ['tokens'],
            name='',
            domain='',
            dtype=6,
            sample_size=5,
            seed=92.0,
        ),
        # 1: TfIdfVectorizer inputs=1 outputs=1
        _node(
            'TfIdfVectorizer',
            ['tokens'],
            ['cp'],
            name='',
            domain='',
            max_gram_length=5,
            max_skip_count=1,
            min_gram_length=1,
            mode='TF',
            ngram_counts=[0, 1, 19, 109, 253],
            ngram_indexes=[1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            pool_int64s=[1, 2, 2, 0, 1, 1, 0, 1, 2, 1, 3, 2, 1, 3, 0, 3, 1, 3, 3, 0, 0, 0, 0, 2, 0, 0, 2, 3, 0, 3, 1, 1, 1,
 1, 1, 2, 1, 1, 3, 1, 1, 3, 3, 2, 2, 2, 3, 0, 1, 3, 3, 1, 0, 0, 1, 0, 0, 2, 0, 1, 1, 0, 3, 0, 0, 3,
 3, 1, 0, 1, 1, 0, 2, 1, 1, 2, 1, 2, 3, 1, 3, 2, 2, 1, 0, 2, 2, 0, 2, 3, 0, 2, 3, 2, 3, 1, 0, 3, 1,
 3, 3, 2, 1, 3, 3, 0, 3, 3, 2, 0, 0, 1, 3, 0, 2, 1, 0, 0, 2, 2, 0, 0, 3, 0, 3, 1, 0, 1, 0, 1, 1, 1,
 1, 1, 2, 1, 0, 1, 3, 2, 1, 2, 1, 3, 2, 2, 3, 0, 2, 3, 2, 1, 3, 0, 0, 0, 1, 0, 0, 3, 1, 0, 1, 0, 2,
 0, 1, 1, 2, 0, 1, 2, 0, 0, 2, 2, 3, 0, 2, 3, 2, 0, 3, 1, 0, 0, 3, 1, 2, 1, 0, 2, 2, 1, 2, 0, 3, 1,
 3, 2, 2, 1, 3, 3, 2, 2, 2, 3, 2, 2, 2, 3, 3, 2, 3, 2, 0, 2, 3, 2, 1, 2, 3, 2, 3, 3, 0, 0, 1, 3, 0,
 0, 3, 3, 1, 2, 3, 3, 2, 1, 0, 3, 3, 0, 0, 3, 3, 1, 2, 3, 3, 2, 1, 0, 0, 0, 1, 3, 0, 1, 0, 2, 2, 0,
 1, 1, 2, 1, 0, 3, 0, 0, 3, 1, 0, 1, 0, 2, 3, 3, 1, 2, 3, 0, 0, 3, 1, 0, 0, 1, 2, 0, 3, 0, 2, 1, 0,
 2, 0, 2, 3, 2, 3, 1, 1, 2, 1, 0, 1, 3, 3, 2, 1, 2, 1, 3, 2, 2, 2, 2, 3, 2, 0, 2, 3, 2, 1, 0, 3, 2,
 2, 3, 3, 3, 3, 0, 0, 1],
        ),
        # 2: TfIdfVectorizer inputs=1 outputs=1
        _node(
            'TfIdfVectorizer',
            ['tokens'],
            ['cm'],
            name='',
            domain='',
            max_gram_length=5,
            max_skip_count=1,
            min_gram_length=1,
            mode='TF',
            ngram_counts=[0, 1, 21, 111, 239],
            ngram_indexes=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            pool_int64s=[2, 0, 0, 0, 3, 1, 2, 2, 1, 2, 2, 2, 3, 3, 0, 3, 1, 3, 2, 3, 3, 0, 1, 0, 0, 2, 3, 1, 1, 1, 1, 1, 3,
 1, 2, 2, 2, 1, 0, 2, 1, 3, 2, 2, 3, 3, 1, 2, 3, 3, 2, 0, 0, 3, 0, 1, 2, 0, 1, 3, 0, 2, 1, 0, 2, 2,
 0, 3, 1, 0, 3, 3, 1, 0, 0, 1, 0, 2, 1, 1, 0, 1, 3, 1, 1, 3, 3, 2, 3, 0, 2, 3, 2, 3, 0, 0, 3, 1, 0,
 3, 1, 1, 3, 2, 0, 3, 2, 1, 3, 3, 3, 0, 0, 0, 1, 0, 0, 1, 3, 0, 1, 0, 2, 0, 1, 2, 0, 0, 2, 0, 3, 0,
 2, 2, 3, 0, 3, 0, 3, 1, 0, 1, 0, 1, 3, 2, 2, 3, 0, 0, 3, 3, 0, 2, 2, 3, 3, 0, 0, 0, 0, 3, 1, 0, 1,
 0, 0, 0, 2, 1, 0, 0, 2, 2, 0, 0, 2, 3, 2, 0, 3, 0, 0, 0, 3, 1, 0, 1, 0, 2, 2, 1, 1, 2, 2, 1, 1, 3,
 3, 1, 3, 3, 2, 1, 3, 3, 3, 2, 1, 0, 2, 2, 1, 3, 2, 2, 2, 3, 2, 2, 3, 2, 0, 2, 3, 2, 3, 3, 1, 1, 2,
 3, 2, 2, 3, 3, 3, 2, 1, 0, 0, 0, 1, 3, 0, 0, 3, 1, 0, 0, 1, 1, 2, 1, 1, 0, 1, 0, 2, 2, 0, 2, 0, 3,
 0, 1, 2, 0, 3, 0, 2, 1, 0, 2, 0, 2, 3, 2, 3, 0, 3, 0, 0, 3, 1, 0, 1, 0, 0, 1, 0, 2, 2, 3, 1, 1, 2,
 1, 0, 1, 3, 3, 2, 1, 2, 1, 3, 2, 2, 2, 2, 3, 2, 0, 3, 0, 2, 2, 0, 3, 1, 1, 2, 2, 3, 2, 2, 3, 3],
        ),
        # 3: Einsum inputs=48 outputs=1
        _node(
            'Einsum',
            ['input', 'E', 'E', 'P', 'P', 'P', 'cp', 'B', 'B', 'B', 'E', 'B', 'B', 'B', 'P', 'P', 'P', 'cp',
             'B', 'B', 'B', 'E', 'B', 'B', 'B', 'P', 'P', 'P', 'cm', 'B', 'B', 'B', 'E', 'B', 'B', 'B', 'P',
             'P', 'cm', 'B', 'B', 'B', 'E', 'B', 'B', 'B', 'P', 'P'],
            ['output'],
            name='',
            domain='',
            equation='ncrs,fc,fq,fz,tz,tz,ni,igh,jgm,jmj,hM,tjk,twk,wab,ao,ar,bs,nI,Ipy,JpC,JCJ,yP,tJK,tWK,WAB,AO,Ar,Bs,nl,lFG,uFH,uHu,GS,tuv,txv,xde,dr,es,nL,LNQ,UNR,URU,QT,tUV,tXV,XDE,Dr,Es->nqrs',
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
            _tensor('prob', TensorProto.FLOAT, (1, 4), INIT_PROB_0),
            _tensor('P', TensorProto.FLOAT, (2, 30), INIT_P_1),
            _tensor('B', TensorProto.FLOAT, (2, 2, 2), INIT_B_2),
            _tensor('E', TensorProto.FLOAT, (2, 10), INIT_E_3),
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
