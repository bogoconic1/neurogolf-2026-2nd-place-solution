from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task170'
TASK_NUM = 170
KAGGLE = {'score': 19.918596, 'date': '2026-07-15'}
MEMORY_BYTES = 70
PARAMS = 91
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = ''
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task170_cases30_selector_cost161'
OPSETS = [('', 20)]


# LIKE: FLOAT16[1], 1 value(s)
INIT_LIKE_0 = [0.0]

# CASES: STRING[30], 30 value(s)
INIT_CASES_1 = ['-72057594072924161,-18018796558090241,-4294967297', '-1128098933440769,-288230651029618689,-1',
 '-4398066696707,-1,-18295873486192641', '-1099515101249,-8589934593,-6597069766657',
 '-105906177,-72057602627862529,-9007220729577473', '-4397072385,-16777217,-27021597764222977',
 '-5764607525293000705,-22553221180424193,-1125899906842625', '-281475081699329,-4503599629467649,-1126999418470401',
 '-288230376158396417,-65537,-2267759509505', '-285873044324353,-144115256795332609,-137439215617',
 '-86376449,-4785074637635585,-4194305', '-4398085701633,-1,-72339069016735745',
 '-281612470583297,-1,-288230376152760321', '-4535540383745,-1048577,-281474976710657',
 '-17185308961,-144115188075855873,-288230376151711745', '-70516737,-4194305,-144116300472385537',
 '-8660254721,-274911461377,-1116691496961', '-53739521,-563224831328257,-288230380446679041',
 '-70583105,-1,-562949953421313', '-158331294580737,-73192294339575809,-100663297',
 '-38928385,-288230376168554497,-137439215617', '-288230376190902273,-1236950581249,-281474976710657',
 '-70582273,-20971521,-144678138029277185', '-102039809,-563087396569089,-1', '-144115188146307329,-1,-4456449',
 '-274912903169,-9008298766434305,-288230376151711745', '-39190529,-1,-360287970191802369',
 '-107020289,-76561210845167617,-131073', '-137477948417,-1,-1126999418470401',
 '-38928389,-72057594037927937,-286010462175233']

# C: INT64[10, 3], 30 value(s)
INIT_C_2 = [4294967296, 0, 0, 65536, 0, 0, 1, 0, 0, 0, 4294967296, 0, 0, 65536, 0, 281474976710656, 0, 0, 0, 1, 0, 0, 0,
 4294967296, 0, 0, 65536, 0, 0, 1]

# P: INT64[30], 30 value(s)
INIT_P_3 = [8, 4, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


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
            ['rng_f'],
            name='',
            domain='',
            dtype=10,
            high=28.605682373046875,
            low=-26.08678436279297,
            seed=5402863.0,
        ),
        # 1: Cast inputs=1 outputs=1
        _node(
            'Cast',
            ['rng_f'],
            ['case_i'],
            name='',
            domain='',
            to=6,
        ),
        # 2: Gather inputs=2 outputs=1
        _node(
            'Gather',
            ['CASES', 'case_i'],
            ['case_string'],
            name='',
            domain='',
            axis=0,
        ),
        # 3: StringSplit inputs=1 outputs=2
        _node(
            'StringSplit',
            ['case_string'],
            ['parts', 'lengths'],
            name='',
            domain='',
            delimiter=',',
            maxsplit=2,
        ),
        # 4: Cast inputs=1 outputs=1
        _node(
            'Cast',
            ['parts'],
            ['packed'],
            name='',
            domain='',
            to=7,
        ),
        # 5: Einsum inputs=7 outputs=1
        _node(
            'Einsum',
            ['packed', 'C', 'P', 'P', 'P', 'P', 'P'],
            ['output'],
            name='',
            domain='',
            equation='bk,ck,r,r,r,r,w->bcrw',
        ),
    ]

    graph = helper.make_graph(
        nodes,
        GRAPH_NAME,
        [
            _vi('input', TensorProto.FLOAT, [1, 10, 30, 30]),
        ],
        [
            _vi('output', TensorProto.INT64, [1, 10, 30, 30]),
        ],
        initializer=[
            _tensor('LIKE', TensorProto.FLOAT16, (1,), INIT_LIKE_0),
            _tensor('CASES', TensorProto.STRING, (30,), INIT_CASES_1),
            _tensor('C', TensorProto.INT64, (10, 3), INIT_C_2),
            _tensor('P', TensorProto.INT64, (30,), INIT_P_3),
        ],
        value_info=[
            _vi('rng_f', TensorProto.FLOAT16, [1]),
            _vi('case_i', TensorProto.INT32, [1]),
            _vi('case_string', TensorProto.STRING, [1]),
            _vi('parts', TensorProto.STRING, [1, 3]),
            _vi('lengths', TensorProto.INT64, [1]),
            _vi('packed', TensorProto.INT64, [1, 3]),
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
