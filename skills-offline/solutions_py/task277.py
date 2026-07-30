from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task277'
TASK_NUM = 277
KAGGLE = {'score': 20.229315, 'date': '2026-07-14'}
MEMORY_BYTES = 24
PARAMS = 94
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 7
PRODUCER_NAME = ''
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task277_d3_counts'
OPSETS = [('', 13)]


# dummy: FLOAT16[2], 2 value(s)
INIT_DUMMY_0 = [0.0, 0.0]

# U: FLOAT[30, 2], 60 value(s)
INIT_U_1 = [1.0, 3.120206594467163, 1.0, 3.123387336730957, 1.0, 3.3198814392089844, 1.0, 3.602945566177368, 1.0,
 4.312055587768555, 1.0, 8.713648796081543, 1.0, 12.740205764770508, 1.0, 12.885658264160156, 1.0, 19.346742630004883,
 1.0, 19.394067764282227, 1.0, 3.120206594467163, 1.0, 3.123387336730957, 1.0, 3.3198814392089844, 1.0,
 3.602945566177368, 1.0, 4.312055587768555, 1.0, 8.713648796081543, 1.0, 12.740205764770508, 1.0, 12.885658264160156,
 1.0, 19.346742630004883, 1.0, 19.394067764282227, 1.0, 3.120206594467163, 1.0, 3.123387336730957, 1.0,
 3.3198814392089844, 1.0, 3.602945566177368, 1.0, 4.312055587768555, 1.0, 8.713648796081543, 1.0, 12.740205764770508,
 1.0, 12.885658264160156, 1.0, 19.346742630004883, 1.0, 19.394067764282227]

# V: FLOAT[3, 2, 2], 12 value(s)
INIT_V_2 = [24.600473403930664, -3.8126461505889893, -0.940443217754364, 0.14906568825244904, 15.64647102355957,
 0.6278108358383179, -4.2526726722717285, 0.1096549853682518, 0.25, 0.5802803635597229, 0.9971675872802734,
 -0.13076943159103394]

# C: FLOAT[10, 2], 20 value(s)
INIT_C_3 = [1.0, 0.0, -1.0, -1.0, -1.0, 1.0, 0.0, -2.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]


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
            ['dummy'],
            ['rf'],
            name='',
            domain='',
            dtype=10,
            high=60000.0,
            low=1.0,
            seed=1.0,
        ),
        # 1: Cast inputs=1 outputs=1
        _node(
            'Cast',
            ['rf'],
            ['token'],
            name='',
            domain='',
            to=6,
        ),
        # 2: TfIdfVectorizer inputs=1 outputs=1
        _node(
            'TfIdfVectorizer',
            ['token'],
            ['G'],
            name='',
            domain='',
            max_gram_length=2,
            max_skip_count=0,
            min_gram_length=1,
            mode='TF',
            ngram_counts=[0, 56],
            ngram_indexes=[1, 2, 0, 2, 1, 0, 0, 2, 2, 2, 1, 2, 0, 2, 0, 1, 0, 2, 1, 0, 0, 1, 1, 2, 2, 2, 2, 1, 2, 0, 0, 0, 2,
 0, 2, 2, 2, 2, 2, 0, 1, 0, 1, 2, 1, 2, 0, 1, 0, 0, 2, 1, 0, 2, 0, 0, 2, 1, 2, 1, 1, 1, 0, 1, 2, 2,
 2, 2, 2, 2, 2, 2, 2, 0, 0, 1, 2],
            pool_int64s=[1, 462, 2075, 2823, 2848, 3208, 4011, 4362, 5518, 7893, 9991, 13138, 14267, 14823, 15747, 16363,
 16495, 19694, 21556, 21920, 23005, 23010, 24960, 25049, 26185, 27519, 28664, 29191, 31165, 31616,
 31782, 31966, 35339, 37898, 37958, 39091, 39235, 40269, 40732, 40758, 42071, 43359, 44165, 45201,
 45336, 45384, 45732, 45989, 49858, 50770, 53082, 53859, 54619, 55826, 56081, 59462, 2823, 40732,
 3208, 31782, 5518, 39235, 14267, 16495, 15747, 2848, 21556, 9991, 23005, 4011, 24960, 42071, 29191,
 53859, 35339, 55826, 37958, 45384, 39091, 4362, 40269, 462, 40758, 56081, 43359, 45201, 44165,
 19694, 45336, 27519, 49858, 2075, 50770, 31616, 54619, 45732, 59462, 21920],
        ),
        # 3: Einsum inputs=17 outputs=1
        _node(
            'Einsum',
            ['input', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'U', 'U', 'G', 'V'],
            ['output'],
            name='',
            domain='',
            equation='nqRC,qp,op,ok,zk,ol,wl,xp,xp,xa,yp,yp,yb,Ra,Cb,d,dab->noRC',
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
            _tensor('dummy', TensorProto.FLOAT16, (2,), INIT_DUMMY_0),
            _tensor('U', TensorProto.FLOAT, (30, 2), INIT_U_1),
            _tensor('V', TensorProto.FLOAT, (3, 2, 2), INIT_V_2),
            _tensor('C', TensorProto.FLOAT, (10, 2), INIT_C_3),
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
