from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task062'
TASK_NUM = 62
KAGGLE = {'score': 19.68188, 'date': '2026-07-15'}
MEMORY_BYTES = 111
PARAMS = 93
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = ''
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task062_direct'
OPSETS = [('', 20)]


# p_color_shared: FLOAT16[1, 1, 1, 1], 1 value(s)
INIT_P_COLOR_SHARED_0 = [0.7001953125]

# zero_i8: INT8[1, 1, 1, 1], 1 value(s)
INIT_ZERO_I8_1 = [0]

# negone_i8: INT8[1, 1, 1, 1], 1 value(s)
INIT_NEGONE_I8_2 = [-1]

# P: INT64[2, 30], 60 value(s)
INIT_P_3 = [0, 128, 4, 8, 2, 32, 64, 16, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# CASES: INT64[30, 1], 30 value(s)
INIT_CASES_4 = [-1175439572810616832, -33495689740091392, -432345567952164864, -19780514836006470, -281474993553664, -335493602550016,
 -19144988520873984, -33495689732292608, -38984876988301312, -33495865826148352, -24345755829403648, -4503668347899904,
 -21955048197980160, -3956102967201294, -31631231550517360, -282578783371264, -31648823737610352, -31543270615023616,
 -1873497444684141056, -20266507565924352, -23714542745042944, -33495522236366848, -18014673391599616, -562949987370496,
 -18014673391599616, -18085043209519104, -22517999483503616, -19215358410114048, -62488398312505344,
 -16325548657492992]


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
            ['p_color_shared'],
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
            ['qpacked'],
            name='',
            domain='',
            axis=0,
        ),
        # 3: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p_color_shared'],
            ['c1'],
            name='',
            domain='',
            dtype=3,
            seed=950276.0,
        ),
        # 4: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p_color_shared'],
            ['c4'],
            name='',
            domain='',
            dtype=3,
            seed=115513.0,
        ),
        # 5: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p_color_shared'],
            ['c5'],
            name='',
            domain='',
            dtype=3,
            seed=9399013.0,
        ),
        # 6: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p_color_shared'],
            ['c6'],
            name='',
            domain='',
            dtype=3,
            seed=2462569.0,
        ),
        # 7: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p_color_shared'],
            ['c7'],
            name='',
            domain='',
            dtype=3,
            seed=1873798.0,
        ),
        # 8: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p_color_shared'],
            ['c8'],
            name='',
            domain='',
            dtype=3,
            seed=852839.0,
        ),
        # 9: Bernoulli inputs=1 outputs=1
        _node(
            'Bernoulli',
            ['p_color_shared'],
            ['c9'],
            name='',
            domain='',
            dtype=3,
            seed=5568487.0,
        ),
        # 10: Concat inputs=10 outputs=1
        _node(
            'Concat',
            ['zero_i8', 'c1', 'zero_i8', 'negone_i8', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9'],
            ['Wdyn'],
            name='',
            domain='',
            axis=0,
        ),
        # 11: Cast inputs=1 outputs=1
        _node(
            'Cast',
            ['Wdyn'],
            ['W64'],
            name='',
            domain='',
            to=7,
        ),
        # 12: Einsum inputs=11 outputs=1
        _node(
            'Einsum',
            ['W64', 'qpacked', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P'],
            ['output'],
            name='',
            domain='',
            equation='oabc,uvxyk,kr,ks,ks,ks,ks,ks,ks,ks,ks->uors',
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
            _tensor('p_color_shared', TensorProto.FLOAT16, (1, 1, 1, 1), INIT_P_COLOR_SHARED_0),
            _tensor('zero_i8', TensorProto.INT8, (1, 1, 1, 1), INIT_ZERO_I8_1),
            _tensor('negone_i8', TensorProto.INT8, (1, 1, 1, 1), INIT_NEGONE_I8_2),
            _tensor('P', TensorProto.INT64, (2, 30), INIT_P_3),
            _tensor('CASES', TensorProto.INT64, (30, 1), INIT_CASES_4),
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
