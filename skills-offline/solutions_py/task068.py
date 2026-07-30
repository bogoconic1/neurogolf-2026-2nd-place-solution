from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task068'
TASK_NUM = 68
KAGGLE = {'score': 20.002788, 'date': '2026-07-15'}
MEMORY_BYTES = 32
PARAMS = 116
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 8
PRODUCER_NAME = 'gpt-5.6-sol/swarm-20plus'
PRODUCER_VERSION = 'task068-tie'
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task068_linux_fitted'
OPSETS = [('', 12)]


# B: FLOAT[2, 30], 60 value(s)
INIT_B_0 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.1414213627576828, 0.2828427255153656, 0.4242640733718872, 0.5656854510307312,
 0.7071068286895752, 0.8485281467437744, 0.9899495244026184, 1.1313709020614624, 1.2727922201156616, 1.4142136573791504,
 1.5556349754333496, 1.6970562934875488, 1.8384777307510376, 1.9798990488052368, 2.1213204860687256, 2.262741804122925,
 2.404163122177124, 2.5455844402313232, 2.6870059967041016, 2.828427314758301, 2.9698486328125, 3.111269950866699,
 3.2526912689208984, 3.3941125869750977, 3.535534143447876, 3.676955461502075, 3.8183767795562744, 3.9597980976104736,
 4.101219654083252]

# C2: FLOAT[2, 10], 20 value(s)
INIT_C2_1 = [1.0001338720321655, 0.9998725056648254, 0.9998641610145569, 1.0001338720321655, 1.0001338720321655, 1.0001357793807983,
 0.9998704791069031, 0.9998641610145569, 0.9999337196350098, 1.0001357793807983, -0.9998661875724792,
 -0.7778814435005188, -0.5556309819221497, -0.3332887589931488, -0.11109624803066254, 0.11109602451324463,
 0.3332901895046234, 0.5554801821708679, 0.7776721715927124, 0.9998641610145569]

# Rfac: FLOAT[2, 2, 2], 8 value(s)
INIT_RFAC_2 = [1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0]

# P: FLOAT[1, 4], 4 value(s)
INIT_P_3 = [1.0, 1.0, 1.0, 1.0]

# Wcore: FLOAT[3, 2, 2, 2], 24 value(s)
INIT_WCORE_4 = [-0.6146237850189209, 0.003710597986355424, -0.28259479999542236, -0.00041116602369584143, 0.5975890159606934,
 -0.014516404829919338, 0.270956814289093, -0.00019565530237741768, -0.385531485080719, 0.062442339956760406,
 0.6456156373023987, 0.0004139975644648075, 0.2605036497116089, -0.08964162319898605, -0.7789259552955627,
 0.0004020482301712036, -0.028134731575846672, 0.20330815017223358, -0.0723203793168068, 0.0008053867495618761,
 -0.05878064036369324, -0.2181425243616104, -0.02410501055419445, 0.00016281026182696223]


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
            ['P'],
            ['tokens'],
            name='linux_0_Multinomial',
            domain='',
            dtype=6,
            sample_size=4,
            seed=392.0,
        ),
        # 1: TfIdfVectorizer inputs=1 outputs=1
        _node(
            'TfIdfVectorizer',
            ['tokens'],
            ['row'],
            name='linux_1_TfIdfVectorizer',
            domain='',
            max_gram_length=4,
            max_skip_count=0,
            min_gram_length=1,
            mode='TFIDF',
            ngram_counts=[0, 3, 31, 85],
            ngram_indexes=[1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1,
 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1],
            pool_int64s=[0, 2, 3, 0, 0, 0, 1, 0, 2, 0, 3, 1, 0, 1, 1, 1, 2, 1, 3, 2, 0, 2, 1, 3, 0, 3, 1, 3, 2, 3, 3, 0, 0,
 2, 0, 1, 0, 0, 1, 2, 0, 2, 0, 0, 3, 3, 1, 1, 0, 1, 1, 1, 1, 1, 3, 1, 3, 2, 2, 1, 0, 2, 1, 1, 2, 2,
 2, 2, 2, 3, 2, 3, 2, 3, 1, 1, 3, 1, 3, 3, 2, 1, 3, 3, 1, 0, 0, 0, 2, 0, 0, 1, 0, 0, 0, 2, 3, 0, 1,
 1, 2, 1, 0, 0, 2, 1, 1, 1, 1, 1, 2, 1, 3, 1, 2, 2, 2, 2, 1, 0, 0, 2, 1, 1, 3, 2, 1, 3, 3, 2, 2, 2,
 0, 2, 2, 3, 2, 2, 3, 3, 0, 3, 1, 1, 0, 3, 1, 3, 2, 3, 2, 1, 2, 3, 2, 3, 0],
            weights=[1.0, 0.1414213627576828, 0.1414213627576828, 0.1414213627576828, 1.0, 0.1414213627576828, 0.1414213627576828, 0.1414213627576828, 0.1414213627576828, 1.0, 0.1414213627576828, 1.0, 0.1414213627576828, 0.1414213627576828, 0.1414213627576828, 1.0, 1.0, 1.0, 0.1414213627576828, 1.0, 1.0, 0.1414213627576828, 0.1414213627576828, 0.1414213627576828, 0.1414213627576828, 0.1414213627576828, 1.0, 1.0, 0.1414213627576828, 1.0, 1.0, 1.0, 0.1414213627576828, 1.0, 1.0, 0.1414213627576828, 0.1414213627576828, 1.0, 0.1414213627576828, 1.0, 1.0, 1.0, 1.0, 0.1414213627576828, 0.1414213627576828, 0.1414213627576828, 0.1414213627576828, 1.0, 0.1414213627576828, 1.0, 0.1414213627576828, 1.0, 0.1414213627576828],
        ),
        # 2: TfIdfVectorizer inputs=1 outputs=1
        _node(
            'TfIdfVectorizer',
            ['tokens'],
            ['col'],
            name='linux_2_TfIdfVectorizer',
            domain='',
            max_gram_length=4,
            max_skip_count=0,
            min_gram_length=1,
            mode='TFIDF',
            ngram_counts=[0, 4, 26, 104],
            ngram_indexes=[1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1,
 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1],
            pool_int64s=[0, 1, 2, 3, 0, 1, 0, 2, 0, 3, 1, 0, 1, 1, 1, 2, 2, 0, 2, 1, 2, 3, 3, 1, 3, 3, 0, 0, 0, 0, 0, 1, 0,
 1, 0, 0, 1, 1, 0, 1, 2, 0, 1, 3, 0, 2, 3, 1, 0, 1, 1, 1, 3, 1, 2, 0, 1, 2, 1, 1, 2, 2, 1, 3, 1, 2,
 0, 2, 2, 1, 0, 2, 1, 1, 2, 1, 2, 2, 2, 2, 2, 3, 0, 2, 3, 2, 3, 0, 2, 3, 1, 1, 3, 1, 2, 3, 2, 1, 3,
 2, 3, 3, 3, 0, 0, 0, 0, 2, 0, 0, 2, 3, 0, 1, 3, 1, 0, 3, 3, 1, 1, 0, 0, 1, 1, 0, 1, 2, 1, 1, 1, 1,
 1, 2, 0, 2, 1, 2, 1, 3, 1, 2, 2, 2, 2, 1, 0, 0, 2, 1, 1, 3, 2, 3, 3, 0, 3, 0, 2, 0, 3, 1, 2, 0, 3,
 1, 3, 2, 3, 2, 1, 2, 3, 2, 3, 0],
            weights=[1.0, 0.1414213627576828, 0.1414213627576828, 0.1414213627576828, 0.1414213627576828, 1.0, 0.1414213627576828, 1.0, 0.1414213627576828, 1.0, 1.0, 0.1414213627576828, 1.0, 0.1414213627576828, 1.0, 0.1414213627576828, 0.1414213627576828, 1.0, 1.0, 0.1414213627576828, 0.1414213627576828, 1.0, 0.1414213627576828, 1.0, 0.1414213627576828, 0.1414213627576828, 0.1414213627576828, 1.0, 1.0, 0.1414213627576828, 0.1414213627576828, 1.0, 0.1414213627576828, 0.1414213627576828, 1.0, 0.1414213627576828, 1.0, 0.1414213627576828, 1.0, 0.1414213627576828, 1.0, 0.1414213627576828, 1.0, 0.1414213627576828, 0.1414213627576828, 0.1414213627576828, 0.1414213627576828, 1.0, 1.0, 0.1414213627576828, 0.1414213627576828, 0.1414213627576828, 0.1414213627576828, 1.0, 0.1414213627576828, 0.1414213627576828, 1.0, 0.1414213627576828, 0.1414213627576828],
        ),
        # 3: Einsum inputs=36 outputs=1
        _node(
            'Einsum',
            ['row', 'row', 'row', 'row', 'col', 'col', 'col', 'col', 'Rfac', 'Rfac', 'Rfac', 'Rfac', 'Rfac',
             'Rfac', 'B', 'B', 'B', 'B', 'Rfac', 'Rfac', 'Rfac', 'Rfac', 'Rfac', 'Rfac', 'B', 'B', 'B', 'B',
             'Wcore', 'Wcore', 'Wcore', 'C2', 'C2', 'C2', 'C2', 'input'],
            ['output'],
            name='linux_3_Einsum',
            domain='',
            equation='ar,at,az,ay,av,ai,ax,aj,AJt,AIr,UAc,CFy,CDz,VCg,Dh,Ih,Jh,Fh,UEs,ELx,EKv,VGu,GHi,GTj,Kw,Hw,Lw,Tw,QUMO,QVNP,Qbfb,Mo,No,Od,Pd,ndhw->nohw',
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
            _tensor('B', TensorProto.FLOAT, (2, 30), INIT_B_0),
            _tensor('C2', TensorProto.FLOAT, (2, 10), INIT_C2_1),
            _tensor('Rfac', TensorProto.FLOAT, (2, 2, 2), INIT_RFAC_2),
            _tensor('P', TensorProto.FLOAT, (1, 4), INIT_P_3),
            _tensor('Wcore', TensorProto.FLOAT, (3, 2, 2, 2), INIT_WCORE_4),
        ],
        value_info=[
            _vi('tokens', TensorProto.INT32, [1, 4]),
            _vi('row', TensorProto.FLOAT, [1, 2]),
            _vi('col', TensorProto.FLOAT, [1, 2]),
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
