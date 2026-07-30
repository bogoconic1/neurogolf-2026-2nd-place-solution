from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task213'
TASK_NUM = 213
KAGGLE = {'score': 20.355609, 'date': '2026-07-14'}
MEMORY_BYTES = 20
PARAMS = 84
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = 'task213_multinomial_tfidf_count_S'
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task213_multinomial_tfidf_count_S'
OPSETS = [('', 20)]


# W: FLOAT[1, 10], 10 value(s)
INIT_W_0 = [-1.0, -1.0, -1.0, -1.0, -1.0, 0.0, -1.0, -1.0, -1.0, -1.0]

# B: FLOAT[2, 30], 60 value(s)
INIT_B_1 = [-0.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -8.0, -8.0, -9.0, -9.0, -9.0, -1.199999957179898e-07, -1.199999957179898e-07,
 -1.199999957179898e-07, -17.0, -17.0, -17.0, -18.0, -18.0, -18.0, -18.0, -18.0, -18.0, 34.0, 34.0, 34.0, 34.0, 34.0,
 54.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 9.99999993922529e-09, 9.99999993922529e-09,
 9.99999993922529e-09, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -2.0, -2.0, -2.0, -2.0, -2.0, -3.0]

# P1: FLOAT[2, 2], 4 value(s)
INIT_P1_2 = [0.0, -1.0, 3.0, 0.15000000596046448]

# P2: FLOAT[2, 2], 4 value(s)
INIT_P2_3 = [0.0, -1.0, 3.0, -2.200000047683716]

# Tp: FLOAT[2, 2], 4 value(s)
INIT_TP_4 = [1.0, 1.0, 0.0, 1.0]

# C3: FLOAT[2], 2 value(s)
INIT_C3_5 = [1.0, 17.0]


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
            ['W'],
            ['seq'],
            name='',
            domain='',
            dtype=6,
            sample_size=3,
            seed=481.0,
        ),
        # 1: TfIdfVectorizer inputs=1 outputs=1
        _node(
            'TfIdfVectorizer',
            ['seq'],
            ['S'],
            name='',
            domain='',
            max_gram_length=3,
            max_skip_count=2,
            min_gram_length=1,
            mode='TF',
            ngram_counts=[0, 10, 96],
            ngram_indexes=[1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1,
 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1,
 0, 1, 1, 1, 1, 1, 1],
            pool_int64s=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 3, 0, 4, 0, 6, 1, 1, 1, 3, 1, 5, 1, 7, 3, 0, 3, 1, 3, 3, 3, 4, 3,
 5, 3, 6, 3, 9, 4, 0, 4, 1, 4, 6, 4, 9, 5, 1, 5, 3, 5, 4, 5, 5, 5, 7, 6, 3, 6, 9, 7, 0, 7, 1, 7, 2,
 7, 3, 7, 4, 7, 5, 7, 6, 7, 7, 7, 8, 7, 9, 8, 2, 8, 6, 8, 7, 8, 8, 8, 9, 9, 4, 9, 5, 9, 8, 0, 3, 4,
 0, 3, 6, 1, 3, 5, 1, 7, 1, 3, 0, 6, 3, 5, 3, 4, 9, 5, 5, 3, 1, 5, 4, 1, 5, 5, 3, 5, 5, 7, 5, 6, 9,
 5, 8, 8, 6, 5, 3, 7, 6, 3, 7, 7, 0, 7, 9, 4, 7, 9, 5, 7, 9, 8, 8, 7, 2],
        ),
        # 2: Einsum inputs=63 outputs=1
        _node(
            'Einsum',
            ['input', 'input', 'W', 'B', 'P1', 'B', 'B', 'P2', 'B', 'B', 'P1', 'B', 'B', 'P2', 'B', 'S', 'B',
             'S', 'Tp', 'B', 'S', 'Tp', 'Tp', 'B', 'S', 'Tp', 'Tp', 'Tp', 'B', 'B', 'B', 'B', 'Tp', 'B', 'C3',
             'B', 'C3', 'Tp', 'B', 'S', 'B', 'S', 'Tp', 'B', 'S', 'Tp', 'Tp', 'B', 'S', 'Tp', 'Tp', 'Tp', 'B',
             'B', 'B', 'B', 'Tp', 'B', 'C3', 'B', 'C3', 'Tp', 'B'],
            ['output'],
            name='',
            domain='',
            equation='nkhw,nkij,nk,ah,ab,br,ch,cd,dr,ej,ef,fs,gj,gl,ls,nm,mr,no,op,pr,nq,qu,uv,vr,nx,xy,yz,zA,Ar,BF,Br,CS,CD,Dr,G,Gr,H,HI,Ir,nJ,Js,nK,KL,Ls,nM,MN,NO,Os,nP,PQ,QR,RT,Ts,UV,Us,WY,WX,Xs,Z,Zs,t,tE,Es->nkrs',
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
            _tensor('W', TensorProto.FLOAT, (1, 10), INIT_W_0),
            _tensor('B', TensorProto.FLOAT, (2, 30), INIT_B_1),
            _tensor('P1', TensorProto.FLOAT, (2, 2), INIT_P1_2),
            _tensor('P2', TensorProto.FLOAT, (2, 2), INIT_P2_3),
            _tensor('Tp', TensorProto.FLOAT, (2, 2), INIT_TP_4),
            _tensor('C3', TensorProto.FLOAT, (2,), INIT_C3_5),
        ],
        value_info=[
            _vi('seq', TensorProto.INT32, [1, 3]),
            _vi('S', TensorProto.FLOAT, [1, 2]),
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
