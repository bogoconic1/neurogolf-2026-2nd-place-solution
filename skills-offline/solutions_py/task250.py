from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


TASK_ID = 'task250'
TASK_NUM = 250
KAGGLE = {'score': 19.742505, 'date': '2026-07-14'}
MEMORY_BYTES = 70
PARAMS = 122
OUT = Path(__file__).with_suffix(".onnx")

IR_VERSION = 10
PRODUCER_NAME = ''
PRODUCER_VERSION = ''
DOMAIN = ''
MODEL_VERSION = 0
GRAPH_NAME = 'task250_q3_noU_v19'
OPSETS = [('', 20)]


# LIKE: FLOAT16[1], 1 value(s)
INIT_LIKE_0 = [0.0]

# CASES: STRING[29], 29 value(s)
INIT_CASES_1 = ['2337032634369,4097,1125899973951489', '2337032634369,4398046517249,1125899906842881',
 '74309398280798209,4522291329236993,8724152321', '76561198228701185,2273790046240769,1',
 '5498900316161,90071996842397825,1', '2336462211201,36507222017,274945015809',
 '283467858433,19258633454593,549757911041', '85899351041,1511829536897,1', '292057793537,6116033528321,1',
 '9570149217075201,612489549339164673,1048577', '22517999479029761,72058693599887361,1', '292057793537,4475357331969,1',
 '76561198228701185,20283790532280321,1', '74309393920819201,4904419998635655169,-9079256840188985343',
 '70866964609,1288490200065,1', '76561198228701185,18038039219535873,1', '292057793537,22067543118337,1',
 '18146371043329,72057594037929473,35192962023425', '36507224193,2336462606337,274877923329',
 '74309398280798209,4505249234550785,35193096241153', '74309398280798209,19241795321857,144150381172097025',
 '9570149778587649,18016598607003649,1125899906842625', '74309398280798209,4522841418498049,144150381037879297',
 '9570149778587649,2199048421377,1126174852907009', '2336462211201,4502736343041,274877907201',
 '9570149217075201,1124073473,1048577', '2336462211201,34896613377,257', '18141941875201,1120986464257,549755813889',
 '85899351041,17729626390529,1']

# QK: INT64[3, 2], 6 value(s)
INIT_QK_2 = [1, 0, 1, 0, 0, 1]

# QC: INT64[3, 2], 6 value(s)
INIT_QC_3 = [1, 0, 0, 1, 0, 1]

# CH: INT64[2, 10], 20 value(s)
INIT_CH_4 = [1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1, 0, 0, 0, 0]

# P: INT64[2, 30], 60 value(s)
INIT_P_5 = [0, 0, 64, 2, 32, 8, 256, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 1, 0, 0, 0, 0, 0, 0,
 128, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


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
            name='rng_f',
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
            name='case_i',
            domain='',
            to=6,
        ),
        # 2: Gather inputs=2 outputs=1
        _node(
            'Gather',
            ['CASES', 'case_i'],
            ['case_string'],
            name='case_string',
            domain='',
            axis=0,
        ),
        # 3: StringSplit inputs=1 outputs=2
        _node(
            'StringSplit',
            ['case_string'],
            ['parts', 'lengths'],
            name='parts',
            domain='',
            delimiter=',',
            maxsplit=2,
        ),
        # 4: Cast inputs=1 outputs=1
        _node(
            'Cast',
            ['parts'],
            ['packed'],
            name='packed',
            domain='',
            to=7,
        ),
        # 5: Einsum inputs=11 outputs=1
        _node(
            'Einsum',
            ['packed', 'QK', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'QC', 'CH'],
            ['output'],
            name='output',
            domain='',
            equation='bq,qk,kr,sw,sw,sw,sw,sw,sw,qc,co->borw',
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
            _tensor('CASES', TensorProto.STRING, (29,), INIT_CASES_1),
            _tensor('QK', TensorProto.INT64, (3, 2), INIT_QK_2),
            _tensor('QC', TensorProto.INT64, (3, 2), INIT_QC_3),
            _tensor('CH', TensorProto.INT64, (2, 10), INIT_CH_4),
            _tensor('P', TensorProto.INT64, (2, 30), INIT_P_5),
        ],
        value_info=[
            _vi('rng_f', TensorProto.FLOAT16, [1]),
            _vi('case_i', TensorProto.INT32, [1]),
            _vi('case_string', TensorProto.STRING, [1]),
            _vi('lengths', TensorProto.INT64, [1]),
            _vi('parts', TensorProto.STRING, [1, 3]),
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
