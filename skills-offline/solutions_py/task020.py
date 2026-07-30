from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task020'
TASK_NUM = 20
KAGGLE = {'score': 20.072746, 'date': '2026-07-15'}
MEMORY_BYTES = 44
PARAMS = 94
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 9
PRODUCER_NAME = ''
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task020_random_case_renderer'
OPSETS = [('', 14)]


# C: FLOAT[2, 10], 20 value(s)
INIT_C_0 = [1.0, 0.50048828125, 0.333984375, 0.250732421875, 0.20078125596046448, 0.1119791641831398, 0.1436941921710968,
 0.1258544921875, 0.16748046875, 0.10087890923023224, -1.0, 0.50048828125, 0.66796875, 0.752197265625,
 0.8031250238418579, 0.8958333134651184, 0.8621651530265808, 0.8809814453125, 0.83740234375, 0.907910168170929]

# F: FLOAT[2, 30], 60 value(s)
INIT_F_1 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.3551969528198242, -9.377433776855469, 1.0416526794433594,
 -4.3270583152771, -3.232144832611084, 2.5958797931671143, -0.5866072177886963, -3.6867074966430664, 2.052358388900757,
 7.514593601226807, -2.83668851852417, -8.321306228637695, 9.697503089904785, 0.8211553692817688, 0.30000001192092896,
 4.509730339050293, -4.509730339050293, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
 0.31592074036598206, -1.026898980140686, 0.6136523485183716, -2.408984899520874, -6.465945243835449,
 -2.8858683109283447, -1.037555456161499, -4.340938568115234, -6.351940631866455, -5.041463851928711,
 -3.752817153930664, -6.631875514984131, 4.620640754699707, -9.605924606323242, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# Jm: FLOAT[2, 2, 2], 8 value(s)
INIT_JM_2 = [1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0]

# logits: FLOAT[1, 6], 6 value(s)
INIT_LOGITS_3 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


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
            seed=1918079.0,
        ),
        # 1: TfIdfVectorizer inputs=1 outputs=1
        _node(
            'TfIdfVectorizer',
            ['tok'],
            ['R'],
            name='',
            domain='',
            max_gram_length=3,
            max_skip_count=1,
            min_gram_length=1,
            mode='TFIDF',
            ngram_counts=[0, 6, 44],
            ngram_indexes=[1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1,
 1, 1, 1, 1, 0, 1, 1],
            pool_int64s=[0, 1, 2, 3, 4, 5, 0, 0, 0, 1, 0, 2, 0, 3, 0, 4, 1, 0, 1, 3, 1, 4, 2, 2, 2, 5, 3, 2, 3, 4, 4, 1, 4,
 2, 4, 3, 5, 2, 5, 3, 5, 4, 5, 5, 0, 1, 0, 0, 2, 2, 0, 4, 1, 1, 0, 3, 2, 3, 2, 3, 0, 4, 4, 1, 2, 4,
 1, 4, 4, 3, 3, 5, 1, 0, 5, 3, 2, 5, 3, 5, 5, 4, 0, 5, 4, 2, 5, 5, 0],
            weights=[1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ),
        # 2: TfIdfVectorizer inputs=1 outputs=1
        _node(
            'TfIdfVectorizer',
            ['tok'],
            ['CenterC'],
            name='',
            domain='',
            max_gram_length=3,
            max_skip_count=1,
            min_gram_length=1,
            mode='TFIDF',
            ngram_counts=[0, 5, 47],
            ngram_indexes=[1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1,
 1, 1, 0, 0, 0, 0, 0, 0, 1, 0],
            pool_int64s=[0, 1, 3, 4, 5, 0, 0, 0, 1, 0, 2, 0, 3, 0, 4, 1, 0, 1, 3, 1, 4, 1, 5, 2, 0, 2, 2, 2, 3, 2, 5, 3, 2,
 4, 0, 4, 2, 4, 3, 4, 4, 4, 5, 5, 0, 5, 5, 0, 1, 0, 0, 1, 5, 0, 2, 0, 0, 3, 4, 0, 4, 0, 0, 4, 1, 0,
 4, 4, 1, 0, 3, 1, 2, 5, 1, 4, 3, 3, 0, 4, 4, 3, 3, 5, 1, 0, 5, 3, 2, 5, 4, 0, 5, 4, 5, 5, 5, 0],
            weights=[1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ),
        # 3: TfIdfVectorizer inputs=1 outputs=1
        _node(
            'TfIdfVectorizer',
            ['tok'],
            ['D'],
            name='',
            domain='',
            max_gram_length=3,
            max_skip_count=1,
            min_gram_length=1,
            mode='TFIDF',
            ngram_counts=[0, 3, 47],
            ngram_indexes=[1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1,
 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0],
            pool_int64s=[0, 1, 4, 0, 1, 0, 2, 0, 3, 0, 4, 1, 0, 1, 2, 1, 4, 1, 5, 2, 0, 2, 1, 2, 2, 2, 5, 3, 1, 3, 2, 4, 2,
 4, 4, 4, 5, 5, 1, 5, 2, 5, 3, 5, 4, 5, 5, 0, 1, 0, 0, 1, 5, 0, 2, 0, 0, 2, 2, 0, 3, 4, 0, 4, 0, 0,
 4, 1, 0, 4, 3, 1, 0, 3, 1, 2, 5, 1, 4, 3, 2, 1, 4, 3, 0, 4, 3, 1, 0, 4, 1, 0, 4, 1, 2, 4, 1, 4, 4,
 2, 1, 4, 3, 3, 5, 1, 0, 5, 4, 2, 5, 4, 5, 5, 5, 0],
            weights=[1.0, 0.20000000298023224, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ),
        # 4: TfIdfVectorizer inputs=1 outputs=1
        _node(
            'TfIdfVectorizer',
            ['tok'],
            ['P'],
            name='',
            domain='',
            max_gram_length=3,
            max_skip_count=1,
            min_gram_length=1,
            mode='TFIDF',
            ngram_counts=[0, 4, 50],
            ngram_indexes=[1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1,
 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0],
            pool_int64s=[0, 1, 3, 4, 0, 0, 0, 1, 0, 3, 0, 5, 1, 2, 1, 3, 1, 5, 2, 2, 3, 1, 3, 2, 3, 3, 3, 4, 4, 0, 4, 1, 4,
 2, 4, 3, 4, 4, 5, 0, 5, 1, 5, 2, 5, 3, 5, 4, 5, 5, 0, 1, 0, 0, 1, 5, 0, 2, 0, 0, 2, 2, 0, 3, 4, 0,
 4, 0, 0, 4, 1, 0, 4, 3, 0, 4, 4, 1, 0, 3, 1, 2, 5, 1, 4, 3, 2, 0, 5, 2, 1, 4, 2, 3, 2, 4, 1, 0, 4,
 1, 2, 4, 2, 1, 5, 3, 2, 5, 4, 0, 5, 5, 0],
            weights=[1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ),
        # 5: Einsum inputs=70 outputs=1
        _node(
            'Einsum',
            ['D', 'F', 'F', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'D',
             'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'Jm', 'Jm', 'Jm', 'Jm', 'CenterC', 'Jm', 'CenterC',
             'Jm', 'Jm', 'F', 'R', 'Jm', 'F', 'R', 'Jm', 'F', 'R', 'Jm', 'F', 'R', 'Jm', 'C', 'P', 'Jm', 'C',
             'P', 'Jm', 'C', 'C', 'Jm', 'C', 'C', 'input', 'F', 'F', 'Jm', 'F', 'CenterC', 'Jm', 'F',
             'CenterC'],
            ['output'],
            name='',
            domain='',
            equation='hw,wV,wV,wT,wT,wT,wU,wU,wU,wU,wU,fR,fR,fR,fR,fR,fQ,fQ,fQ,hf,fS,fS,bt,bt,bt,bt,st,st,at,xwu,svx,gfd,seg,hq,epq,ho,eno,uyA,yr,hA,dlm,lr,hm,dij,ir,hj,uBC,Br,hC,bOP,Oc,hP,bLM,Lc,hM,aHI,Hc,Ik,aJK,Jc,Kk,Nkrz,nz,pz,vDE,Dz,hE,vFG,Fz,hG->Ncrz',
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
            _tensor('C', TensorProto.FLOAT, (2, 10), INIT_C_0),
            _tensor('F', TensorProto.FLOAT, (2, 30), INIT_F_1),
            _tensor('Jm', TensorProto.FLOAT, (2, 2, 2), INIT_JM_2),
            _tensor('logits', TensorProto.FLOAT, (1, 6), INIT_LOGITS_3),
        ],
        value_info=[
            _vi('tok', TensorProto.INT32, [1, 3]),
            _vi('R', TensorProto.FLOAT, [1, 2]),
            _vi('CenterC', TensorProto.FLOAT, [1, 2]),
            _vi('D', TensorProto.FLOAT, [1, 2]),
            _vi('P', TensorProto.FLOAT, [1, 2]),
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
