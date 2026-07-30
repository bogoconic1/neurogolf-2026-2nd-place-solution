from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task148'
TASK_NUM = 148
KAGGLE = {'score': 19.996054, 'date': '2026-07-15'}
MEMORY_BYTES = 40
PARAMS = 109
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 9
PRODUCER_NAME = ''
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task148_3root_len4_tiedpower'
OPSETS = [('', 20)]


# U: FLOAT[2, 10], 20 value(s)
INIT_U_0 = [1.0, 0.0, 1.0, 0.0, 3.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, -2.0, 0.0, 0.0, 0.0, -1.0, 0.0]

# P: FLOAT[2, 30], 60 value(s)
INIT_P_1 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 4.931999683380127, 3.6649997234344482, 2.3979997634887695, 1.1309998035430908,
 -0.1360001564025879, -1.4030001163482666, -2.6700000762939453, -3.937000036239624, -4.604499816894531,
 -5.672500133514404, -6.874000072479248, -8.208999633789062, -10.345000267028809, -12.34749984741211,
 -15.017499923706055, -19.690000534057617, -23.69499969482422, -29.03499984741211, -37.04499816894531,
 -45.05500030517578, -57.06999969482422, -73.08999633789062, -89.11000061035156, -105.1300048828125,
 -121.15000915527344, -137.17001342773438, -153.1900177001953, -169.21002197265625, -185.2300262451172,
 -201.25003051757812]

# PA: FLOAT[3, 2], 6 value(s)
INIT_PA_2 = [1.0, -1.0, 0.0, 0.5, -1.0, 1.0]

# CA: FLOAT[3, 2], 6 value(s)
INIT_CA_3 = [0.0, -0.3333333432674408, 0.0, -0.3333333432674408, 1.0, -0.3333333432674408]

# CB: FLOAT[2, 2], 4 value(s)
INIT_CB_4 = [1.0, 1.0, -3.0, 0.0]

# Ffac: FLOAT[2, 2, 2], 8 value(s)
INIT_FFAC_5 = [1.0, -2.0, 2.0, 0.0, 0.0, 2.0, -2.0, 0.0]

# logits: FLOAT[1, 5], 5 value(s)
INIT_LOGITS_6 = [0.0, 0.0, 0.0, 0.0, 0.0]


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
            sample_size=4,
            seed=2.0,
        ),
        # 1: TfIdfVectorizer inputs=1 outputs=1
        _node(
            'TfIdfVectorizer',
            ['tokens'],
            ['root0'],
            name='',
            domain='',
            max_gram_length=4,
            max_skip_count=0,
            min_gram_length=1,
            mode='TFIDF',
            ngram_counts=[0, 3, 27, 129],
            ngram_indexes=[0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1,
 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0,
 1, 0, 0, 1, 1, 0],
            pool_int64s=[4, 0, 3, 4, 2, 1, 2, 2, 2, 0, 3, 3, 2, 4, 4, 3, 0, 3, 3, 3, 4, 3, 1, 1, 3, 4, 3, 1, 4, 2, 4, 2, 1,
 0, 0, 0, 0, 1, 2, 1, 2, 2, 4, 2, 0, 2, 0, 3, 0, 0, 3, 4, 4, 0, 4, 0, 2, 1, 1, 1, 1, 1, 2, 3, 3, 0,
 3, 0, 3, 3, 4, 4, 4, 4, 1, 0, 4, 2, 3, 2, 1, 2, 4, 2, 4, 2, 2, 3, 1, 1, 1, 1, 3, 1, 3, 4, 3, 4, 1,
 4, 0, 4, 0, 4, 4, 4, 3, 1, 3, 1, 3, 1, 4, 4, 4, 4, 4, 4, 3, 0, 4, 1, 3, 4, 1, 4, 1, 4, 3, 1, 4, 2,
 1, 4, 0, 0, 0, 0, 1, 2, 2, 4, 2, 0, 3, 4, 4, 2, 1, 4, 0, 0, 3, 4, 4, 0, 2, 1, 1, 1, 2, 3, 3, 0, 3,
 3, 4, 4, 1, 0, 0, 4, 2, 3, 3, 2, 1, 2, 4, 2, 2, 3, 1, 1, 3, 1, 3, 4, 1, 4, 0, 4, 4, 4, 3, 1, 3, 1,
 4, 4, 4, 2, 4, 3, 0, 3, 4, 1, 3, 2, 4, 2, 0, 4, 3, 2, 2, 4, 1, 4, 3],
            weights=[1.0, 16.020000457763672, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ),
        # 2: TfIdfVectorizer inputs=1 outputs=1
        _node(
            'TfIdfVectorizer',
            ['tokens'],
            ['root1'],
            name='',
            domain='',
            max_gram_length=4,
            max_skip_count=0,
            min_gram_length=1,
            mode='TFIDF',
            ngram_counts=[0, 5, 41, 161],
            ngram_indexes=[0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1,
 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1,
 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1],
            pool_int64s=[1, 4, 2, 0, 3, 1, 4, 4, 2, 4, 0, 0, 0, 1, 2, 2, 2, 3, 2, 2, 3, 2, 0, 4, 4, 3, 0, 1, 1, 3, 3, 4, 1,
 0, 4, 2, 4, 3, 1, 4, 3, 1, 4, 2, 4, 2, 1, 0, 0, 0, 0, 0, 1, 0, 1, 4, 0, 3, 2, 4, 2, 0, 4, 4, 2, 3,
 0, 0, 0, 0, 3, 4, 4, 0, 4, 0, 2, 1, 1, 2, 3, 3, 0, 3, 0, 3, 3, 4, 4, 4, 4, 1, 0, 0, 4, 0, 4, 2, 3,
 3, 2, 3, 2, 1, 2, 4, 2, 4, 2, 2, 1, 3, 4, 4, 0, 4, 0, 4, 4, 4, 2, 4, 2, 4, 1, 4, 3, 1, 3, 1, 3, 1,
 4, 4, 4, 4, 4, 2, 4, 3, 4, 3, 0, 4, 1, 3, 4, 3, 2, 3, 2, 2, 2, 0, 0, 4, 1, 4, 1, 4, 3, 1, 4, 2, 1,
 4, 0, 0, 0, 0, 0, 1, 4, 0, 1, 2, 2, 0, 3, 2, 3, 4, 2, 0, 3, 3, 0, 0, 3, 4, 0, 0, 3, 4, 4, 0, 2, 1,
 1, 1, 2, 3, 3, 0, 3, 0, 0, 4, 2, 3, 3, 2, 1, 2, 4, 2, 2, 1, 3, 4, 1, 1, 1, 1, 3, 4, 2, 4, 1, 4, 3,
 1, 3, 1, 4, 4, 4, 2, 4, 3, 0, 2, 4, 2, 0, 4, 3, 2, 2, 2, 0, 0, 0, 4, 1, 4, 3],
            weights=[1.0, 16.020000457763672, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ),
        # 3: TfIdfVectorizer inputs=1 outputs=1
        _node(
            'TfIdfVectorizer',
            ['tokens'],
            ['root2'],
            name='',
            domain='',
            max_gram_length=4,
            max_skip_count=0,
            min_gram_length=1,
            mode='TFIDF',
            ngram_counts=[0, 3, 31, 124],
            ngram_indexes=[1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1,
 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1,
 1, 1, 0, 1],
            pool_int64s=[1, 4, 2, 4, 0, 1, 2, 0, 3, 3, 2, 2, 3, 2, 0, 3, 0, 0, 2, 3, 3, 3, 4, 4, 1, 0, 4, 3, 1, 1, 3, 1, 4,
 2, 4, 0, 0, 0, 0, 0, 0, 0, 1, 3, 2, 3, 4, 2, 0, 2, 0, 3, 3, 0, 0, 4, 4, 0, 1, 1, 1, 1, 1, 2, 3, 3,
 0, 3, 0, 3, 3, 4, 4, 0, 0, 4, 0, 4, 2, 3, 3, 2, 3, 2, 1, 2, 4, 2, 3, 1, 1, 1, 1, 3, 1, 3, 4, 4, 0,
 4, 0, 4, 4, 1, 4, 4, 4, 4, 4, 2, 4, 3, 4, 3, 0, 4, 3, 2, 3, 2, 2, 1, 4, 3, 4, 0, 0, 0, 0, 0, 1, 4,
 0, 1, 2, 2, 0, 3, 2, 3, 4, 2, 0, 3, 4, 4, 2, 1, 3, 0, 0, 3, 4, 0, 0, 3, 4, 4, 0, 2, 3, 3, 0, 3, 3,
 4, 4, 1, 0, 0, 4, 2, 3, 3, 2, 1, 3, 1, 1, 3, 1, 3, 4, 1, 4, 0, 4, 4, 1, 4, 4, 4, 2, 4, 3, 0, 3, 4,
 1, 3, 2, 4, 2, 0, 2, 0, 0, 0, 4, 1, 4, 3],
            weights=[2.0, 16.020000457763672, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ),
        # 4: Einsum inputs=79 outputs=1
        _node(
            'Einsum',
            ['root0', 'CA', 'CB', 'P', 'root1', 'CA', 'CB', 'PA', 'P', 'PA', 'CA', 'root2', 'CA', 'CB', 'P',
             'root2', 'CA', 'CB', 'P', 'U', 'input', 'input', 'U', 'P', 'CA', 'CA', 'PA', 'PA', 'Ffac', 'U',
             'input', 'input', 'U', 'CA', 'CA', 'CA', 'PA', 'PA', 'Ffac', 'P', 'P', 'P', 'input', 'U', 'input',
             'U', 'CA', 'PA', 'CB', 'U', 'CA', 'PA', 'CA', 'CB', 'CB', 'Ffac', 'U', 'CB', 'CA', 'CA', 'CA',
             'CA', 'CA', 'CA', 'PA', 'CB', 'CB', 'P', 'P', 'P', 'P', 'P', 'PA', 'CA', 'Ffac', 'CB', 'CB',
             'Ffac', 'Ffac'],
            ['output'],
            name='',
            domain='',
            equation='...c,tc,cd,dh,...f,tf,fg,tB,gh,tB,XB,...j,tj,jk,kh,...m,tm,mn,nh,BA,...AhC,...DhC,ED,BC,iE,iE,tB,tB,BBE,BF,...FhH,...IhH,JI,YJ,pJ,sJ,tW,tW,WTU,Ew,TH,Uw,...Khw,LK,...Nhw,PN,ZM,ZL,MP,Oo,eM,eO,aG,Gl,Ml,Qlb,Qo,xQ,tR,tR,tR,tR,tR,tR,tS,yS,By,Bz,Bz,Bz,Bz,Bz,tq,tr,rMq,uM,ru,mVv,Vmv->...ohw',
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
            _tensor('U', TensorProto.FLOAT, (2, 10), INIT_U_0),
            _tensor('P', TensorProto.FLOAT, (2, 30), INIT_P_1),
            _tensor('PA', TensorProto.FLOAT, (3, 2), INIT_PA_2),
            _tensor('CA', TensorProto.FLOAT, (3, 2), INIT_CA_3),
            _tensor('CB', TensorProto.FLOAT, (2, 2), INIT_CB_4),
            _tensor('Ffac', TensorProto.FLOAT, (2, 2, 2), INIT_FFAC_5),
            _tensor('logits', TensorProto.FLOAT, (1, 5), INIT_LOGITS_6),
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
