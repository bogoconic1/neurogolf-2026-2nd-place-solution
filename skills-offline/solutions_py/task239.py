from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task239'
TASK_NUM = 239
KAGGLE = {'score': 19.506939, 'date': '2026-07-15'}
MEMORY_BYTES = 182
PARAMS = 61
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = ''
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task239_ordered_rng_string_replay_table31_crosszero'
OPSETS = [('', 20)]


# LIKE: FLOAT16[1], 1 value(s)
INIT_LIKE_0 = [0.0]

# CASES: STRING[30], 30 value(s)
INIT_CASES_1 = ['-910120888434689,-1,-211106232532993,-4398046511105,-279223181191938049,-7881299347898369,-1,-1,-1,-1',
 '-281474976710657,-1,-279223176896970753,-1,-8444249301319681,-1,-1,-1,-1,-1',
 '-910120888434689,-1,-279223181191938049,-1,-7881299347898369,-1,-211106232532993,-1,-1,-4398046511105',
 '-910121018318849,-1,-7881299347898369,-1,-211106232532993,-4398046511105,-279223185218469889,-1,-1,-1',
 '-131941533745153,-8725724278030337,-1,-1,-1,-279223181191938049,-1,-1,-1,-140737488355329',
 '-281474976710657,-8444249301319681,-1,-1,-1,-1,-1,-1,-1,-279223176896970753',
 '-347170796470273,-1,-4398046511105,-279223176896970753,-211106232532993,-1,-1,-1,-8444249301319681,-1',
 '-976366567686145,-1,-279223184413163521,-1,-140737488355329,-1,-1,-1,-7881299347898369,-1',
 '-615726511554561,-211106232532993,-1,-1,-7881299347898369,-1,-1,-1,-1,-270215977642229761',
 '-910120888434689,-211106232532993,-7881299347898369,-4398046511105,-279223181191938049,-1,-1,-1,-1,-1',
 '-347171056238593,-4398046511105,-1,-1,-1,-279223184950034433,-1,-211106232532993,-1,-8444249301319681',
 '-281475211591681,-279223184413163521,-1,-1,-1,-1,-1,-1,-8444249301319681,-1',
 '-343047896039425,-1,-1,-1,-1,-1,-279223185218469889,-8444249301319681,-1,-211106232532993',
 '-615726511554561,-1,-1,-7881299347898369,-1,-211106232532993,-1,-270215977642229761,-1,-1',
 '-910120888434689,-211106232532993,-279223181191938049,-1,-1,-7881299347898369,-1,-1,-1,-4398046511105',
 '-2102266370719745,-1,-140737488355329,-1,-1,-279223181191938049,-1,-1,-6755399441055745,-1',
 '-30511655485441,-1,-246290604621825,-1,-1,-8725724278030337,-279223183339421697,-4398046511105,-1,-1',
 '-844425064349697,-1,-7881299347898369,-1,-1,-1,-279223181191938049,-1,-1,-1',
 '-281475211591681,-1,-1,-8444249301319681,-1,-279223184413163521,-1,-1,-1,-1',
 '-343047766278145,-279223181191938049,-8444249301319681,-1,-1,-1,-1,-1,-211106232532993,-1',
 '-976366325465089,-279223176896970753,-7881299347898369,-140737488355329,-1,-1,-1,-1,-1,-1',
 '-615726511554561,-7881299347898369,-1,-211106232532993,-270215977642229761,-1,-1,-1,-1,-1',
 '-343047766278145,-211106232532993,-1,-1,-279223181191938049,-1,-1,-1,-1,-8444249301319681',
 '-281474976710657,-8444249301319681,-1,-279223176896970753,-1,-1,-1,-1,-1,-1',
 '-343047766278145,-8444249301319681,-1,-279223181191938049,-1,-1,-1,-1,-211106232532993,-1',
 '-844425181790209,-1,-1,-279223184950034433,-1,-1,-1,-1,-1,-7881299347898369',
 '-309916388691969,-279223181191938049,-246290604621825,-6597069766657,-137438953473,-1,-1,-1,-8444249301319681,-1',
 '-343047896039425,-1,-279223185218469889,-1,-8444249301319681,-211106232532993,-1,-1,-1,-1',
 '-844425064349697,-1,-279223181191938049,-1,-7881299347898369,-1,-1,-1,-1,-1',
 '-343047766278145,-1,-211106232532993,-1,-1,-1,-279223181191938049,-1,-1,-8444249301319681']

# P: INT64[30], 30 value(s)
INIT_P_2 = [2, 4, 8, 16, 32, 67108864, 134217728, 268435456, 536870912, 1073741824, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
 0, 0, 0, 0]


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
            maxsplit=9,
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
            ['packed', 'P', 'P', 'P', 'P', 'P', 'P'],
            ['output'],
            name='',
            domain='',
            equation='bc,r,w,w,w,w,w->bcrw',
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
            _tensor('P', TensorProto.INT64, (30,), INIT_P_2),
        ],
        value_info=[
            _vi('rng_f', TensorProto.FLOAT16, [1]),
            _vi('case_i', TensorProto.INT32, [1]),
            _vi('case_string', TensorProto.STRING, [1]),
            _vi('parts', TensorProto.STRING, [1, 10]),
            _vi('lengths', TensorProto.INT64, [1]),
            _vi('packed', TensorProto.INT64, [1, 10]),
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
