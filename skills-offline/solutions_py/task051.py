from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task051'
TASK_NUM = 51
KAGGLE = {'score': 19.989365, 'date': '2026-07-13'}
MEMORY_BYTES = 56
PARAMS = 94
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 8
PRODUCER_NAME = ''
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'g'
OPSETS = [('', 15)]


# P: FLOAT[2, 30], 60 value(s)
INIT_P_0 = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, -30.0,
 -30.0, -30.0, -30.0, -30.0, -30.0, -30.0, -30.0, -30.0, 103.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.7913460731506348, 2.7913460731506348, 2.7913460731506348,
 2.7913460731506348, 2.7913460731506348, 2.7913460731506348, 2.7913460731506348, 2.7913460731506348, 2.7913460731506348,
 -35.12211608886719]

# C: FLOAT[2, 10], 20 value(s)
INIT_C_1 = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

# A: FLOAT[2, 2, 2], 8 value(s)
INIT_A_2 = [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]

# NEG: FLOAT[2], 2 value(s)
INIT_NEG_3 = [-1.0, 1.0]

# prob_state: FLOAT[1, 4], 4 value(s)
INIT_PROB_STATE_4 = [1.0, 1.0, 1.0, 1.0]


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
        # 0: Einsum inputs=5 outputs=1
        _node(
            'Einsum',
            ['input', 'C', 'A', 'NEG', 'P'],
            ['H'],
            name='',
            domain='',
            equation='nchw,rc,xry,x,sh->ns',
        ),
        # 1: Einsum inputs=5 outputs=1
        _node(
            'Einsum',
            ['input', 'C', 'A', 'NEG', 'P'],
            ['W'],
            name='',
            domain='',
            equation='nchw,rc,xry,x,sw->ns',
        ),
        # 2: Multinomial inputs=1 outputs=1
        _node(
            'Multinomial',
            ['prob_state'],
            ['tokens'],
            name='state_tokens',
            domain='',
            dtype=6,
            sample_size=4,
            seed=4529.0,
        ),
        # 3: TfIdfVectorizer inputs=1 outputs=1
        _node(
            'TfIdfVectorizer',
            ['tokens'],
            ['K'],
            name='decode_K',
            domain='',
            max_gram_length=4,
            max_skip_count=3,
            min_gram_length=1,
            mode='TFIDF',
            ngram_counts=[0, 2, 24, 102],
            ngram_indexes=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1,
 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0],
            pool_int64s=[2, 3, 0, 1, 0, 2, 0, 3, 1, 0, 1, 1, 1, 2, 2, 1, 2, 2, 2, 3, 3, 2, 3, 3, 0, 0, 3, 0, 1, 0, 0, 1, 2,
 0, 2, 0, 0, 3, 0, 1, 0, 2, 1, 1, 1, 1, 1, 2, 1, 1, 3, 1, 2, 0, 1, 2, 1, 1, 2, 2, 1, 3, 1, 1, 3, 2,
 1, 3, 3, 2, 0, 2, 2, 0, 3, 2, 3, 1, 2, 3, 2, 3, 0, 3, 3, 1, 2, 3, 2, 0, 3, 2, 1, 3, 2, 2, 3, 2, 3,
 3, 3, 0, 0, 0, 3, 0, 0, 1, 2, 0, 0, 2, 0, 3, 1, 0, 1, 3, 1, 0, 2, 2, 1, 1, 1, 3, 1, 1, 2, 1, 1, 2,
 1, 3, 2, 0, 2, 2, 2, 1, 3, 1, 2, 3, 2, 0, 3, 0, 0, 1, 3, 0, 1, 2, 3, 1, 2, 1, 3, 2, 1, 3, 3, 2, 2,
 0, 3, 2, 3, 1, 3, 3, 0, 3],
            weights=[1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ),
        # 4: TfIdfVectorizer inputs=1 outputs=1
        _node(
            'TfIdfVectorizer',
            ['tokens'],
            ['db0'],
            name='decode_db0',
            domain='',
            max_gram_length=4,
            max_skip_count=3,
            min_gram_length=1,
            mode='TFIDF',
            ngram_counts=[0, 4, 4, 4],
            ngram_indexes=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            pool_int64s=[0, 1, 2, 3, 0, 0, 3, 0, 0, 1, 0, 2, 0, 1, 2, 0, 0, 2, 0, 3, 0, 2, 2, 0, 0, 3, 0, 0, 0, 3, 0, 3, 1,
 0, 2, 2, 1, 2, 2, 3, 3, 1, 2, 1, 3, 2, 1, 3, 3, 2, 2, 0, 3, 3, 0, 3],
            weights=[1.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ),
        # 5: TfIdfVectorizer inputs=1 outputs=1
        _node(
            'TfIdfVectorizer',
            ['tokens'],
            ['db1'],
            name='decode_db1',
            domain='',
            max_gram_length=4,
            max_skip_count=3,
            min_gram_length=1,
            mode='TFIDF',
            ngram_counts=[0, 4, 4, 4],
            ngram_indexes=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            pool_int64s=[0, 1, 2, 3, 0, 0, 3, 0, 0, 1, 2, 0, 0, 2, 0, 3, 0, 2, 2, 2, 0, 3, 0, 3, 1, 0, 1, 3, 1, 0, 2, 2, 1,
 1, 3, 3, 1, 2, 2, 3, 2, 3, 2, 0, 3, 0, 0, 1],
            weights=[1.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ),
        # 6: Einsum inputs=72 outputs=1
        _node(
            'Einsum',
            ['P', 'P', 'P', 'A', 'A', 'NEG', 'A', 'A', 'P', 'P', 'A', 'NEG', 'NEG', 'NEG', 'db1', 'A', 'NEG',
             'H', 'A', 'NEG', 'NEG', 'db0', 'A', 'A', 'NEG', 'H', 'A', 'NEG', 'H', 'NEG', 'A', 'W', 'A', 'A',
             'NEG', 'W', 'H', 'NEG', 'A', 'A', 'A', 'W', 'NEG', 'H', 'NEG', 'A', 'W', 'P', 'P', 'P', 'NEG', 'K',
             'A', 'K', 'NEG', 'A', 'C', 'C', 'C', 'P', 'P', 'P', 'P', 'C', 'A', 'NEG', 'C', 'C', 'A', 'NEG',
             'C', 'input'],
            ['output'],
            name='',
            domain='',
            equation='jp,jp,lV,kmA,mFG,m,ikl,cij,bt,bt,BaD,B,l,O,Zm,abc,C,...C,zZC,c,K,ZK,jzO,jJK,j,...x,JLM,M,...M,I,OHI,...I,kns,nqr,r,...r,...Q,Q,NPQ,KNO,ORS,...S,S,...v,v,suv,...x,qw,Hw,Rw,g,Zg,bfg,Ze,e,bde,do,bo,fo,Lh,Ph,uh,Zh,To,xTU,T,UY,Xo,xWX,W,WY,...Yhw->...ohw',
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
            _tensor('C', TensorProto.FLOAT, (2, 10), INIT_C_1),
            _tensor('A', TensorProto.FLOAT, (2, 2, 2), INIT_A_2),
            _tensor('NEG', TensorProto.FLOAT, (2,), INIT_NEG_3),
            _tensor('prob_state', TensorProto.FLOAT, (1, 4), INIT_PROB_STATE_4),
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
