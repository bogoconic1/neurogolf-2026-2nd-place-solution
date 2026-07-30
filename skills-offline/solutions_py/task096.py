from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task096'
TASK_NUM = 96
KAGGLE = {'score': 19.36521, 'date': '2026-07-15'}
MEMORY_BYTES = 160
PARAMS = 120
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = 'neurogolf'
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task096_field5_int32_channels'
OPSETS = [('', 20)]


# LIKE: FLOAT16[1], 1 value(s)
INIT_LIKE_0 = [0.0]

# CASES: STRING[29], 29 value(s)
INIT_CASES_1 = ['0,0,-321056768,-319018752,1827405824,0,-62914560,0,-525261824,-1879048192',
 '-6315602,-292558104,-274816542,-270850551,-16801368,-18882116,1073741824,-403331648,-1885340424,-361787920',
 '-18896470,-6347696,1073741824,-135136258,-2120278,-361788423,-16784964,-274792734,-1885340498,-309335320',
 '0,-308019200,0,1677721600,-1425014784,0,-830472192,0,0,0',
 '-6299208,-1885339716,-275780888,-528117078,-6314246,1073741824,-18896470,-268732849,-284196864,-274799645',
 '-22102,-1897930496,-274760658,-18896402,-274816542,-286265216,-522088293,1073741824,-277880600,-18896470',
 '0,-62914560,0,-202244096,1827405824,-325254144,-319018752,0,-1879048192,0',
 '0,1827405824,0,-524737536,-62914560,0,-1913389056,0,-319018752,-321056768',
 '0,0,1677721600,-830472192,0,-1416626176,-281018368,-308019200,0,0',
 '0,1677721600,0,0,-281018368,-342884352,-308019200,-1904214016,0,0',
 '0,-830472192,1677721600,-308019200,0,0,-281018368,-1416626176,0,0',
 '0,0,-281018368,1677721600,-1416626176,-308019200,0,0,-830472192,0',
 '-16784196,-274816542,-18898504,-1885339718,-522868499,1073741824,-275779456,-287605688,-277900616,-22034',
 '-2119254,-1885339716,-274799645,-521013875,1073741824,-286265184,-6299204,-277900612,-6299207,-6347968',
 '-18896470,1621426176,-277900612,-274816542,-287606192,-18897480,-286265216,-18896470,-2013265920,-18882130',
 '0,-62914560,0,0,1827405824,-319018752,0,-525261824,-321056768,-1879048192',
 '0,-281018368,-308019200,0,0,0,-830472192,0,-1416626176,1677721600',
 '-6299219,-274783428,-361787911,-1885339718,-23112,-403577634,-274816542,-18896402,-303041536,1073741824',
 '-2119254,1073741824,-494824304,-274816542,-292558104,-16799250,-274760658,-1885339970,-15751168,-18896470',
 '0,-1913389056,-62914560,-201768960,-325254144,-319018752,0,0,1827405824,0',
 '-18897751,-303041536,1073741824,-18896402,-492717937,-274816542,-277900546,-274777042,-18896470,-1885341448',
 '0,-308019200,1677721600,0,0,-281018368,0,-830472192,-1416626176,0',
 '0,-308019200,-1416626176,-830472192,0,0,-281018368,0,0,1677721600',
 '0,-830472192,0,0,-281018368,-308019200,0,1677721600,-1416626176,0',
 '0,0,-308019200,0,0,-830472192,1677721600,-281018368,-1416626176,0',
 '-18896470,-277900546,-292555794,-494813304,-274777042,-18896402,1073741824,-1885339718,-274816542,-18896402',
 '0,-830472192,-281018368,1677721600,-1416626176,-308019200,0,0,0,0',
 '-281018368,-1904214016,-308019200,0,1677721600,0,0,0,-342884352,0',
 '0,1677721600,0,-1904214016,-281018368,-308019200,0,0,0,-342884352']

# V: INT32[3, 30], 90 value(s)
INIT_V_2 = [2, 64, 1, 16, 1, 64, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 512, 2, 1, 2048, 16, 2048,
 1, 2, 512, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4096, 16, 48, 16384, 2, 1, 2, 16384, 48, 16,
 4096, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


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
            high=13.812538146972656,
            low=-16.865692138671875,
            seed=559349184.0,
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
            maxsplit=9,
        ),
        # 4: Cast inputs=1 outputs=1
        _node(
            'Cast',
            ['parts'],
            ['packed'],
            name='',
            domain='',
            to=6,
        ),
        # 5: RegexFullMatch inputs=1 outputs=1
        _node(
            'RegexFullMatch',
            ['case_string'],
            ['is_s0'],
            name='',
            domain='',
            pattern='(?:0,\\-308019200,0,1677721600,\\-1425014784,0,\\-830472192,0,0,0|\\-281018368,\\-1904214016,\\-308019200,0,1677721600,0,0,0,\\-342884352,0|0,1677721600,0,\\-1904214016,\\-281018368,\\-308019200,0,0,0,\\-342884352|0,\\-830472192,\\-281018368,1677721600,\\-1416626176,\\-308019200,0,0,0,0|0,\\-308019200,\\-1416626176,\\-830472192,0,0,\\-281018368,0,0,1677721600|0,\\-281018368,\\-308019200,0,0,0,\\-830472192,0,\\-1416626176,1677721600|0,1677721600,0,0,\\-281018368,\\-342884352,\\-308019200,\\-1904214016,0,0|0,\\-308019200,1677721600,0,0,\\-281018368,0,\\-830472192,\\-1416626176,0|0,0,1677721600,\\-830472192,0,\\-1416626176,\\-281018368,\\-308019200,0,0|0,0,\\-281018368,1677721600,\\-1416626176,\\-308019200,0,0,\\-830472192,0|0,\\-830472192,1677721600,\\-308019200,0,0,\\-281018368,\\-1416626176,0,0|0,\\-830472192,0,0,\\-281018368,\\-308019200,0,1677721600,\\-1416626176,0|0,0,\\-308019200,0,0,\\-830472192,1677721600,\\-281018368,\\-1416626176,0)',
        ),
        # 6: RegexFullMatch inputs=1 outputs=1
        _node(
            'RegexFullMatch',
            ['case_string'],
            ['is_s1'],
            name='',
            domain='',
            pattern='(?:0,\\-1913389056,\\-62914560,\\-201768960,\\-325254144,\\-319018752,0,0,1827405824,0|0,0,\\-321056768,\\-319018752,1827405824,0,\\-62914560,0,\\-525261824,\\-1879048192|0,\\-62914560,0,0,1827405824,\\-319018752,0,\\-525261824,\\-321056768,\\-1879048192|0,\\-62914560,0,\\-202244096,1827405824,\\-325254144,\\-319018752,0,\\-1879048192,0|0,1827405824,0,\\-524737536,\\-62914560,0,\\-1913389056,0,\\-319018752,\\-321056768)',
        ),
        # 7: RegexFullMatch inputs=1 outputs=1
        _node(
            'RegexFullMatch',
            ['case_string'],
            ['is_s2'],
            name='',
            domain='',
            pattern='(?:\\-18896470,\\-277900546,\\-292555794,\\-494813304,\\-274777042,\\-18896402,1073741824,\\-1885339718,\\-274816542,\\-18896402|\\-18896470,1621426176,\\-277900612,\\-274816542,\\-287606192,\\-18897480,\\-286265216,\\-18896470,\\-2013265920,\\-18882130|\\-6299208,\\-1885339716,\\-275780888,\\-528117078,\\-6314246,1073741824,\\-18896470,\\-268732849,\\-284196864,\\-274799645|\\-18896470,\\-6347696,1073741824,\\-135136258,\\-2120278,\\-361788423,\\-16784964,\\-274792734,\\-1885340498,\\-309335320|\\-2119254,1073741824,\\-494824304,\\-274816542,\\-292558104,\\-16799250,\\-274760658,\\-1885339970,\\-15751168,\\-18896470|\\-2119254,\\-1885339716,\\-274799645,\\-521013875,1073741824,\\-286265184,\\-6299204,\\-277900612,\\-6299207,\\-6347968|\\-6299219,\\-274783428,\\-361787911,\\-1885339718,\\-23112,\\-403577634,\\-274816542,\\-18896402,\\-303041536,1073741824|\\-18897751,\\-303041536,1073741824,\\-18896402,\\-492717937,\\-274816542,\\-277900546,\\-274777042,\\-18896470,\\-1885341448|\\-22102,\\-1897930496,\\-274760658,\\-18896402,\\-274816542,\\-286265216,\\-522088293,1073741824,\\-277880600,\\-18896470|\\-6315602,\\-292558104,\\-274816542,\\-270850551,\\-16801368,\\-18882116,1073741824,\\-403331648,\\-1885340424,\\-361787920|\\-16784196,\\-274816542,\\-18898504,\\-1885339718,\\-522868499,1073741824,\\-275779456,\\-287605688,\\-277900616,\\-22034)',
        ),
        # 8: Concat inputs=3 outputs=1
        _node(
            'Concat',
            ['is_s0', 'is_s1', 'is_s2'],
            ['shape_b'],
            name='',
            domain='',
            axis=0,
        ),
        # 9: Cast inputs=1 outputs=1
        _node(
            'Cast',
            ['shape_b'],
            ['shape_i'],
            name='',
            domain='',
            to=6,
        ),
        # 10: Einsum inputs=4 outputs=1
        _node(
            'Einsum',
            ['packed', 'shape_i', 'V', 'V'],
            ['output'],
            name='',
            domain='',
            equation='bc,s,sr,sw->bcrw',
        ),
    ]

    graph = helper.make_graph(
        nodes,
        GRAPH_NAME,
        [
            _vi('input', TensorProto.FLOAT, [1, 10, 30, 30]),
        ],
        [
            _vi('output', TensorProto.INT32, [1, 10, 30, 30]),
        ],
        initializer=[
            _tensor('LIKE', TensorProto.FLOAT16, (1,), INIT_LIKE_0),
            _tensor('CASES', TensorProto.STRING, (29,), INIT_CASES_1),
            _tensor('V', TensorProto.INT32, (3, 30), INIT_V_2),
        ],
        value_info=[
            _vi('parts', TensorProto.STRING, [1, 10]),
            _vi('packed', TensorProto.INT32, [1, 10]),
            _vi('rng_f', TensorProto.FLOAT16, [1]),
            _vi('case_i', TensorProto.INT32, [1]),
            _vi('case_string', TensorProto.STRING, [1]),
            _vi('lengths', TensorProto.INT64, [1]),
            _vi('is_s0', TensorProto.BOOL, [1]),
            _vi('is_s1', TensorProto.BOOL, [1]),
            _vi('is_s2', TensorProto.BOOL, [1]),
            _vi('shape_b', TensorProto.BOOL, [3]),
            _vi('shape_i', TensorProto.INT32, [3]),
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
