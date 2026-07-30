from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task330'
TASK_NUM = 330
KAGGLE = {'score': 19.900134, 'date': '2026-07-15'}
MEMORY_BYTES = 72
PARAMS = 92
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = 'neurogolf'
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task330_scalar_replay'
OPSETS = [('', 15)]


# U: FLOAT[2, 30], 60 value(s)
INIT_U_0 = [0.569724977016449, 0.569724977016449, 0.569724977016449, 0.569724977016449, 0.569724977016449, 0.569724977016449,
 0.569724977016449, 0.569724977016449, 0.569724977016449, 0.569724977016449, 0.569724977016449, 0.569724977016449,
 0.569724977016449, 0.569724977016449, 0.569724977016449, 0.569724977016449, 0.569724977016449, 0.569724977016449,
 0.569724977016449, 0.569724977016449, 0.569724977016449, 0.569724977016449, 0.569724977016449, 0.569724977016449,
 0.569724977016449, 0.569724977016449, 0.569724977016449, 0.569724977016449, 0.569724977016449, 0.569724977016449,
 2.099147081375122, -3.1972968578338623, 2.1691341400146484, -2.1199944019317627, -1.3822171688079834,
 -1.2131147384643555, -0.9993545413017273, -0.5924599766731262, -0.5207444429397583, -0.41526204347610474, 0.0, 0.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# R: FLOAT[3, 2], 6 value(s)
INIT_R_1 = [1.0, 0.0, 0.0, 1.0, 1.0, 1.0]

# V: FLOAT[10, 2], 20 value(s)
INIT_V_2 = [1.0, 0.0, 1.0, -1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# M: FLOAT[2, 3], 6 value(s)
INIT_M_3 = [1.0, -1.0, 0.0, -0.5, -0.5, 0.5]


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
            ['U'],
            ['tok'],
            name='tok',
            domain='',
            dtype=6,
            sample_size=3,
            seed=201.0,
        ),
        # 1: TfIdfVectorizer inputs=1 outputs=1
        _node(
            'TfIdfVectorizer',
            ['tok'],
            ['left'],
            name='left',
            domain='',
            max_gram_length=3,
            max_skip_count=1,
            min_gram_length=1,
            mode='TFIDF',
            ngram_counts=[0, 5, 187],
            ngram_indexes=[1, 1, 1, 2, 1, 1, 2, 2, 0, 2, 2, 1, 0, 2, 2, 0, 2, 0, 2, 2, 0, 0, 2, 2, 2, 2, 0, 2, 1, 0, 0, 1, 0,
 1, 1, 0, 0, 1, 1, 2, 0, 2, 1, 0, 2, 2, 2, 0, 1, 0, 0, 2, 1, 2, 0, 2, 2, 1, 2, 1, 1, 0, 1, 1, 1, 1,
 2, 2, 0, 1, 2, 1, 0, 1, 2, 1, 1, 1, 2, 0, 0, 0, 1, 1, 1, 1, 2, 0, 0, 2, 1, 1, 2, 2, 1, 1, 0, 2, 0,
 2, 1, 0, 2, 0, 0, 1, 2, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 2, 2, 1, 1, 1, 1, 2, 0, 1, 0, 1, 1, 0, 1, 2,
 1, 2, 1, 2],
            pool_int64s=[1, 3, 10, 11, 26, 0, 0, 0, 4, 0, 9, 0, 14, 0, 15, 0, 18, 0, 23, 1, 14, 2, 0, 2, 2, 2, 11, 2, 14, 2,
 20, 2, 28, 3, 0, 3, 8, 4, 0, 4, 1, 4, 5, 4, 6, 4, 12, 4, 13, 4, 15, 4, 16, 4, 20, 5, 0, 5, 1, 5, 9,
 5, 20, 5, 23, 6, 2, 6, 18, 6, 24, 6, 28, 7, 0, 7, 18, 8, 3, 8, 20, 9, 5, 9, 7, 9, 14, 10, 2, 10, 7,
 11, 29, 12, 4, 12, 6, 12, 8, 12, 22, 13, 0, 13, 28, 14, 16, 14, 18, 14, 20, 14, 23, 15, 0, 15, 20,
 16, 4, 16, 5, 16, 6, 16, 8, 16, 18, 16, 20, 16, 27, 17, 17, 17, 23, 20, 10, 20, 23, 21, 7, 21, 8,
 21, 20, 21, 27, 22, 0, 22, 2, 22, 28, 23, 0, 24, 6, 24, 11, 24, 18, 25, 15, 25, 25, 25, 28, 26, 15,
 27, 4, 27, 5, 27, 7, 27, 12, 27, 13, 27, 28, 28, 19, 28, 21, 28, 23, 0, 2, 0, 0, 12, 4, 0, 14, 20,
 0, 17, 23, 2, 14, 23, 2, 20, 2, 2, 22, 28, 2, 23, 0, 4, 0, 10, 4, 13, 20, 4, 14, 1, 5, 8, 20, 5, 9,
 14, 5, 12, 1, 5, 28, 23, 6, 24, 2, 6, 25, 28, 7, 6, 18, 9, 27, 5, 11, 4, 15, 12, 0, 0, 13, 5, 0,
 14, 0, 18, 15, 20, 0, 16, 21, 8, 16, 21, 20, 16, 24, 18, 16, 27, 22, 17, 0, 17, 20, 0, 23, 21, 0,
 7, 21, 1, 27, 22, 0, 2, 24, 16, 6, 25, 25, 0, 27, 7, 7, 27, 12, 22, 28, 0, 21, 28, 14, 16, 28, 27,
 19],
            weights=[4.0, -2.0101113319396973, -3.404780387878418, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ),
        # 2: TfIdfVectorizer inputs=1 outputs=1
        _node(
            'TfIdfVectorizer',
            ['tok'],
            ['right'],
            name='right',
            domain='',
            max_gram_length=3,
            max_skip_count=1,
            min_gram_length=1,
            mode='TFIDF',
            ngram_counts=[0, 4, 188],
            ngram_indexes=[1, 2, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 2, 2, 1, 2, 1, 1, 1, 0, 0, 2, 0, 1, 1, 2,
 2, 2, 2, 1, 0, 1, 0, 1, 1, 0, 1, 2, 2, 1, 2, 1, 1, 2, 0, 2, 1, 0, 0, 1, 1, 2, 1, 0, 0, 0, 0, 1, 1,
 1, 2, 1, 2, 1, 2, 1, 2, 0, 0, 2, 1, 1, 0, 1, 0, 2, 0, 0, 0, 1, 0, 2, 1, 1, 2, 0, 1, 1, 1, 1, 2, 0,
 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 2, 0, 1, 1, 1, 1, 2, 1, 1, 2, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 2, 0,
 0, 2, 0, 2, 1, 0, 1, 2],
            pool_int64s=[6, 11, 18, 27, 0, 2, 0, 4, 0, 7, 0, 9, 0, 12, 0, 15, 0, 18, 0, 20, 0, 21, 0, 23, 1, 12, 2, 0, 2, 2,
 2, 14, 2, 20, 2, 22, 2, 28, 3, 0, 4, 1, 4, 5, 4, 6, 4, 10, 4, 13, 4, 16, 4, 20, 5, 0, 5, 1, 5, 8,
 5, 9, 5, 20, 5, 23, 5, 28, 6, 2, 6, 18, 6, 28, 7, 0, 7, 26, 8, 0, 8, 3, 9, 5, 9, 14, 10, 2, 10, 7,
 11, 15, 12, 0, 12, 1, 12, 3, 12, 4, 12, 8, 12, 22, 13, 0, 13, 20, 13, 28, 14, 1, 14, 16, 14, 18,
 14, 20, 14, 23, 15, 0, 15, 20, 16, 4, 16, 5, 16, 8, 16, 18, 16, 20, 16, 24, 17, 0, 17, 17, 17, 23,
 20, 2, 20, 23, 21, 1, 21, 7, 21, 8, 21, 20, 21, 27, 22, 0, 23, 0, 23, 15, 24, 11, 25, 0, 25, 15,
 25, 25, 25, 28, 26, 10, 26, 15, 26, 26, 26, 29, 27, 19, 28, 14, 28, 21, 28, 27, 0, 0, 9, 0, 2, 0,
 0, 14, 20, 0, 17, 23, 0, 25, 15, 1, 12, 14, 2, 0, 0, 2, 14, 23, 2, 22, 2, 2, 22, 28, 2, 23, 0, 4,
 0, 10, 4, 12, 6, 5, 0, 0, 5, 8, 20, 5, 9, 14, 5, 12, 1, 5, 28, 23, 6, 24, 2, 7, 0, 0, 7, 26, 29, 9,
 27, 5, 10, 9, 7, 11, 4, 15, 12, 0, 0, 12, 3, 8, 13, 5, 0, 15, 20, 0, 16, 21, 8, 16, 21, 20, 16, 27,
 4, 16, 27, 22, 17, 0, 17, 20, 0, 23, 20, 10, 2, 22, 0, 2, 23, 26, 15, 24, 2, 11, 24, 16, 6, 27, 7,
 7, 27, 12, 22, 27, 13, 28, 28, 0, 21, 28, 14, 16],
            weights=[1.0, -0.2967260479927063, -0.4677557051181793, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ),
        # 3: Einsum inputs=29 outputs=1
        _node(
            'Einsum',
            ['input', 'V', 'M', 'R', 'R', 'V', 'V', 'left', 'R', 'M', 'M', 'M', 'M', 'M', 'M', 'R', 'R', 'R',
             'right', 'M', 'M', 'R', 'R', 'U', 'U', 'R', 'R', 'U', 'U'],
            ['output'],
            name='decode',
            domain='',
            equation='nqhw,qx,xc,ci,cj,oi,oj,za,pz,zp,zp,zp,zp,Xp,Yp,pA,pB,py,yb,xa,xb,ak,al,kh,lh,br,bs,rw,sw->nohw',
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
            _tensor('U', TensorProto.FLOAT, (2, 30), INIT_U_0),
            _tensor('R', TensorProto.FLOAT, (3, 2), INIT_R_1),
            _tensor('V', TensorProto.FLOAT, (10, 2), INIT_V_2),
            _tensor('M', TensorProto.FLOAT, (2, 3), INIT_M_3),
        ],
        value_info=[
            _vi('tok', TensorProto.INT32, [2, 3]),
            _vi('left', TensorProto.FLOAT, [2, 3]),
            _vi('right', TensorProto.FLOAT, [2, 3]),
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
