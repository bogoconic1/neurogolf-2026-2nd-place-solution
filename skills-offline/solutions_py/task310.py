from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task310'
TASK_NUM = 310
KAGGLE = {'score': 19.478539, 'date': '2026-07-12'}
MEMORY_BYTES = 88
PARAMS = 162
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = 'task310_rng_tfidf_final'
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'rng_final'
OPSETS = [('', 18)]


# Feat3: FLOAT[30, 4], 120 value(s)
INIT_FEAT3_0 = [0.0, 0.0, 0.5, -0.0, 0.7937005162239075, 0.0, 0.39685025811195374, -0.0, 0.0, -0.6299605369567871, 0.0, -0.0, 0.5,
 -0.5, 0.0, -0.0, 0.0, 0.0, 0.19842512905597687, -0.19842512905597687, 0.31498026847839355, 0.0, 0.15749013423919678,
 -0.15749013423919678, 0.0, -0.25, 0.0, -0.125, 0.19842512905597687, -0.19842512905597687, 0.0, -0.09921256452798843,
 0.0, 0.0, 0.07874506711959839, -0.0, 0.125, 0.0, 0.0625, -0.0, 0.0, -0.09921256452798843, 0.0, -0.0,
 0.07874506711959839, -0.07874506711959839, 0.0, -0.0, 0.0, 0.0, 0.03125, -0.03125, 0.04960628226399422, 0.0,
 0.02480314113199711, -0.02480314113199711, 0.0, -0.039372533559799194, 0.0, -0.019686266779899597, 0.03125, -0.03125,
 0.0, -0.015625, 0.0, 0.0, 0.012401570565998554, -0.0, 0.019686266779899597, 0.0, 0.009843133389949799, -0.0, 0.0,
 -0.015625, 0.0, -0.0, 0.012401570565998554, -0.012401570565998554, 0.0, -0.0, 0.0, 0.0, 0.004921566694974899,
 -0.004921566694974899, 0.0078125, 0.0, 0.00390625, -0.00390625, 0.0, -0.006200785282999277, 0.0,
 -0.0031003926414996386, 0.004921566694974899, -0.004921566694974899, 0.0, -0.0024607833474874496, 0.0, 0.0,
 0.001953125, -0.0, 0.0031003926414996386, 0.0, 0.0015501963207498193, -0.0, 0.0, -0.0024607833474874496, 0.0, -0.0,
 0.001953125, -0.001953125, 0.0, -0.0, 0.0, 0.0, 0.0007750981603749096, -0.0007750981603749096, 0.0012303916737437248,
 0.0, 0.0006151958368718624, -0.0006151958368718624]

# Sel0: FLOAT[4, 2], 8 value(s)
INIT_SEL0_1 = [-1.0, 1.0, -1.0, 0.0, 2.0, 0.0, 0.0, 0.0]

# Sel1: FLOAT[4, 2], 8 value(s)
INIT_SEL1_2 = [0.0, 0.0, 0.0, -1.0, 2.0, 0.0, 0.0, 0.0]

# Sel2: FLOAT[4, 2], 8 value(s)
INIT_SEL2_3 = [0.0, 0.0, -1.0, 0.0, 2.0, 0.0, 2.0, -2.0]

# like: FLOAT16[2], 2 value(s)
INIT_LIKE_4 = [0.0, 0.0]

# Q: FLOAT[2, 2, 2, 2], 16 value(s)
INIT_Q_5 = [0.0, 0.22796311974525452, 0.1350395530462265, 0.0, 0.051967184990644455, 0.0, 0.3007991313934326, 0.09048011898994446,
 0.051967184990644455, 0.22796311974525452, 0.0, 0.09048011898994446, 0.0, 0.08506958186626434, 0.3007991313934326,
 0.0]


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
            ['like'],
            ['ru'],
            name='ru',
            domain='',
            high=65504.0,
            low=0.0,
            seed=123.0,
        ),
        # 1: Cast inputs=1 outputs=1
        _node(
            'Cast',
            ['ru'],
            ['tok'],
            name='tok',
            domain='',
            to=6,
        ),
        # 2: TfIdfVectorizer inputs=1 outputs=1
        _node(
            'TfIdfVectorizer',
            ['tok'],
            ['tgt_onehot\\/x'],
            name='tf_tgt_onehot\\/x',
            domain='',
            max_gram_length=2,
            max_skip_count=0,
            min_gram_length=1,
            mode='TFIDF',
            ngram_counts=[0, 35],
            ngram_indexes=[2, 6, 4, 3, 2, 9, 8, 2, 3, 5, 8, 6, 7, 4, 6, 7, 3, 5, 6, 8, 6, 2, 3, 3, 9, 8, 6, 3, 1, 6, 1, 2, 6,
 4, 5, 3, 2, 6, 8, 4, 3, 5, 4, 5, 2, 5, 3, 6, 3, 8, 5, 9, 3, 2, 1, 9, 6, 6, 3, 7, 6, 5, 5, 8, 3],
            pool_int64s=[2521, 5143, 10030, 11185, 14515, 18453, 26969, 27116, 29438, 31467, 32775, 34741, 35590, 37715,
 43413, 44451, 45267, 49151, 49849, 51518, 53204, 53355, 53646, 54538, 54901, 55130, 58117, 58181,
 58796, 61048, 61354, 61538, 61597, 63357, 63503, 63, 11734, 61538, 27116, 34741, 61048, 51518,
 32775, 36253, 63357, 11185, 58181, 13673, 16531, 37715, 10030, 36117, 62024, 10488, 14515, 22977,
 30986, 29092, 29063, 5143, 53204, 20416, 28299, 10999, 16143, 63503, 49151, 18453, 54901, 35245,
 24420, 53355, 2521, 58796, 61354, 25270, 55919, 58117, 43413, 8959, 61597, 45267, 53646, 35590,
 44451, 18266, 49849, 16130, 53290, 12419, 31467, 55130, 26969, 54538, 29438],
            weights=[1.0, 8.765456199645996, 0.8399473428726196, 10.666666984558105, 0.12941932678222656, 4.0, 6.34960412979126, 0.10870542377233505, 5.521891117095947, 1.6435229778289795, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ),
        # 3: TfIdfVectorizer inputs=1 outputs=1
        _node(
            'TfIdfVectorizer',
            ['tok'],
            ['rfeat'],
            name='tf_rfeat',
            domain='',
            max_gram_length=2,
            max_skip_count=0,
            min_gram_length=1,
            mode='TFIDF',
            ngram_counts=[0, 50],
            ngram_indexes=[0, 2, 1, 0, 2, 1, 3, 0, 2, 0, 2, 1, 2, 3, 2, 0, 2, 2, 0, 1, 0, 2, 1, 1, 3, 2, 0, 1, 0, 1, 1, 1, 3,
 1, 0, 2, 2, 3, 0, 1, 1, 1, 3, 2, 0, 1, 0, 1, 2, 3, 3, 3, 3, 3, 3],
            pool_int64s=[63, 11734, 61538, 34741, 61048, 51518, 32775, 36253, 63357, 11185, 58181, 13673, 37715, 10030,
 36117, 10488, 14515, 22977, 29092, 29063, 5143, 53204, 20416, 10999, 16143, 63503, 18453, 54901,
 35245, 24420, 53355, 58796, 61354, 25270, 58117, 43413, 8959, 61597, 45267, 53646, 35590, 18266,
 49849, 16130, 12419, 31467, 55130, 26969, 54538, 29438, 63, 11734, 11185, 58181, 5143, 53204,
 12419, 31467, 55130, 26969],
            weights=[1.0, -1.0, 0.5, -0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ),
        # 4: TfIdfVectorizer inputs=1 outputs=1
        _node(
            'TfIdfVectorizer',
            ['tok'],
            ['cfeat'],
            name='tf_cfeat',
            domain='',
            max_gram_length=2,
            max_skip_count=0,
            min_gram_length=1,
            mode='TFIDF',
            ngram_counts=[0, 50],
            ngram_indexes=[0, 2, 1, 0, 2, 1, 3, 0, 2, 0, 1, 0, 1, 1, 2, 0, 2, 2, 0, 2, 2, 2, 0, 2, 1, 2, 3, 0, 1, 1, 2, 2, 0,
 1, 0, 1, 0, 1, 0, 2, 0, 2, 1, 3, 0, 2, 1, 3, 0, 1, 3, 3, 3, 3, 3, 3],
            pool_int64s=[63, 11734, 61538, 34741, 61048, 51518, 32775, 36253, 63357, 11185, 58181, 13673, 16531, 37715,
 36117, 10488, 14515, 22977, 29092, 29063, 5143, 20416, 10999, 16143, 63503, 18453, 54901, 35245,
 24420, 53355, 58796, 25270, 58117, 43413, 8959, 61597, 45267, 53646, 35590, 44451, 18266, 49849,
 16130, 53290, 12419, 31467, 55130, 26969, 54538, 29438, 63, 11734, 34741, 61048, 13673, 16531,
 35245, 24420, 8959, 61597, 18266, 49849],
            weights=[1.0, -1.0, 0.5, -0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ),
        # 5: TfIdfVectorizer inputs=1 outputs=1
        _node(
            'TfIdfVectorizer',
            ['tok'],
            ['scale'],
            name='tf_scale',
            domain='',
            max_gram_length=2,
            max_skip_count=0,
            min_gram_length=1,
            mode='TFIDF',
            ngram_counts=[0, 24],
            ngram_indexes=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            pool_int64s=[2521, 10030, 12419, 16531, 20416, 25270, 28299, 29063, 30986, 31467, 32775, 35590, 36253, 44451,
 45267, 51518, 53355, 53646, 55919, 58796, 61048, 61354, 62024, 63357, 63, 11734, 61538, 27116,
 34741, 61048, 51518, 32775, 36253, 63357, 11185, 58181, 13673, 16531, 37715, 10030, 36117, 62024,
 10488, 14515, 22977, 30986, 29092, 29063, 5143, 53204, 20416, 28299, 10999, 16143, 63503, 49151,
 18453, 54901, 35245, 24420, 53355, 2521, 58796, 61354, 25270, 55919, 58117, 43413, 8959, 61597,
 45267, 53646, 35590, 44451, 18266, 49849, 16130, 53290, 12419, 31467, 55130, 26969, 54538, 29438],
            weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ),
        # 6: Einsum inputs=95 outputs=1
        _node(
            'Einsum',
            ['tgt_onehot\\/x', 'tgt_onehot\\/x', 'tgt_onehot\\/x', 'scale', 'scale', 'scale', 'scale', 'scale',
             'scale', 'scale', 'scale', 'scale', 'scale', 'cfeat', 'Sel2', 'cfeat', 'Sel1', 'cfeat', 'Sel0',
             'rfeat', 'Sel0', 'rfeat', 'Sel2', 'rfeat', 'Sel1', 'Q', 'Q', 'Q', 'Q', 'Q', 'Q', 'Q', 'Q', 'Q',
             'Q', 'Q', 'Q', 'Sel2', 'Q', 'Q', 'Q', 'Q', 'Q', 'Q', 'Feat3', 'Sel0', 'Feat3', 'Feat3', 'Sel1',
             'Q', 'Q', 'Q', 'Sel0', 'input', 'input', 'input', 'Feat3', 'Sel2', 'Feat3', 'Feat3', 'Sel0',
             'Sel1', 'Q', 'Q', 'Q', 'Q', 'Q', 'Q', 'Q', 'Q', 'Q', 'Q', 'Q', 'Q', 'Q', 'Q', 'Q', 'Q', 'Q', 'Q',
             'Q', 'Q', 'Q', 'Sel1', 'Feat3', 'Sel0', 'Feat3', 'Sel2', 'Feat3', 'Sel2', 'Sel1', 'Feat3', 'Feat3',
             'Sel0', 'Feat3'],
            ['output'],
            name='output',
            domain='',
            equation='Z,Z,Z,...,...,...,...,...,...,...,...,...,...,f,fO,m,mN,n,nM,o,ox,u,uz,B,By,TkJO,lOJk,lOJT,TkJO,lOJk,lOJT,TkJO,lOJk,lOJT,TkJO,lOJk,lOJT,gT,SjIN,kNIj,kNIS,SjIN,kNIj,kNIS,wg,iR,wi,wA,AS,RaHM,jMHa,jMHR,va,...ZXw,...ZhV,...chw,hF,FE,hG,hU,UC,GD,Capx,bxpa,bxpC,Dbqy,dyqb,dyqD,Dbqy,dyqb,dyqD,Edtz,eztd,eztE,Edtz,eztd,eztE,Edtz,eztd,eztE,Edtz,eztd,eztE,KI,sK,YH,sY,LJ,sL,Pt,Qq,rP,rQ,Wp,rW->...crs',
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
            _tensor('Feat3', TensorProto.FLOAT, (30, 4), INIT_FEAT3_0),
            _tensor('Sel0', TensorProto.FLOAT, (4, 2), INIT_SEL0_1),
            _tensor('Sel1', TensorProto.FLOAT, (4, 2), INIT_SEL1_2),
            _tensor('Sel2', TensorProto.FLOAT, (4, 2), INIT_SEL2_3),
            _tensor('like', TensorProto.FLOAT16, (2,), INIT_LIKE_4),
            _tensor('Q', TensorProto.FLOAT, (2, 2, 2, 2), INIT_Q_5),
        ],
        value_info=[
            _vi('ru', TensorProto.FLOAT16, [2]),
            _vi('tok', TensorProto.INT32, [2]),
            _vi('tgt_onehot\\/x', TensorProto.FLOAT, [10]),
            _vi('rfeat', TensorProto.FLOAT, [4]),
            _vi('cfeat', TensorProto.FLOAT, [4]),
            _vi('scale', TensorProto.FLOAT, [1]),
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
